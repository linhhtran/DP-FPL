import torch
import torch.nn as nn
from torch.nn import functional as F

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import math

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'DP_FPL',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def orthogonalize(matrix):
    m = matrix.shape[1]
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col

def factorize_ctx(origin, rank):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        v = torch.normal(0, 1, size=(origin.shape[1], rank)).type(origin.dtype) # [ctx_dim, rank]
        u = torch.matmul(origin.to(device), v.to(device)) # [n_ctx, rank]
        orthogonalize(u)
        v = torch.matmul(origin.t().to(device), u.to(device)) # [ctx_dim, rank]
        orthogonalize(v)
        v = v.t() # [rank, ctx_dim]
        residual = origin.to(device) - torch.matmul(u.to(device), v.to(device)) # [n_ctx, ctx_dim]

    return (u, v, residual)

def compute_full_grad(left, right, dtype):
        left_w, left_g = left.data.type(dtype), left.grad.type(dtype) / 10.0
        right_w, right_g = right.data.type(dtype), right.grad.type(dtype) / 10.0

        left_g_right_w = torch.matmul(left_g, right_w)
        m1 = left_g_right_w + torch.matmul(left_w, right_g)
        m2 = torch.matmul(left_w, torch.matmul(left_w.T, left_g_right_w))

        return m1 + m2

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        x = prompts.to(device) + self.positional_embedding.type(self.dtype).to(device) # [100,77,512] + [77, 512]

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DP_FPL.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.factorization = cfg.FACTORIZATION
        self.rank = cfg.RANK
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # global context vector
        global_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # n_ctx = 16, ctx_dim = 512
        nn.init.normal_(global_ctx_vectors, std=0.02)
        self.global_ctx = nn.Parameter(global_ctx_vectors)

        # local u and v context vectors
        if self.factorization != 'fedpgp':
            local_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) # n_ctx = 16, ctx_dim = 512
            nn.init.normal_(local_ctx_vectors, std=0.02)
            self.local_ctx = nn.Parameter(local_ctx_vectors)
        if self.factorization != 'full':
            local_u_ctx_vectors = torch.empty(n_ctx, self.rank, dtype=dtype)
            nn.init.normal_(local_u_ctx_vectors, std=0.02)
            self.local_u_ctx = nn.Parameter(local_u_ctx_vectors)
            local_v_ctx_vectors = torch.empty(self.rank, ctx_dim, dtype=dtype)
            nn.init.normal_(local_v_ctx_vectors, std=0.02)
            self.local_v_ctx = nn.Parameter(local_v_ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # prompts for each class, each of length 16

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("embedding", embedding)

        self.n_cls = n_cls # 100 classes
        self.n_ctx = n_ctx # number of text encoder of text prompts = 16
        self.tokenized_prompts = tokenized_prompts # [100, 77] = [n_cls, clip prompt token limit]
        self.name_lens = name_lens

    def forward(self):
        if self.factorization == 'full':
            client_ctx = self.global_ctx + self.local_ctx
        elif self.factorization == 'fedpgp':
            client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx)
        else:
            local_u_ctx, local_v_ctx, residual = factorize_ctx(self.local_ctx.data, self.rank)
            self.local_u_ctx.data = local_u_ctx
            self.local_v_ctx.data = local_v_ctx
            if self.factorization == 'dpfpl':
                client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx) + residual
            elif self.factorization == 'lora':
                client_ctx = self.global_ctx + torch.matmul(self.local_u_ctx, self.local_v_ctx)

        if client_ctx.dim() == 2:
            client_ctx = client_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        client_prompt = torch.cat(
            [
                self.token_prefix,
                client_ctx,
                self.token_suffix,
            ],
            dim=1,
        )

        return client_prompt


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype)) # [batch, 3, 224, 224] -> [32, 512]

        client_prompt = self.prompt_learner() # [100,77,512] = [n_cls, clip prompt token limit, ctx_dim]
        tokenized_prompts = self.tokenized_prompts
        client_text_features = self.text_encoder(client_prompt, tokenized_prompts) # [100,512] = [n_cls, ctx_dim]

        # normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        client_text_features = client_text_features / client_text_features.norm(dim=-1, keepdim=True)

        # cosine similarity between local text features and image features
        sim = image_features @ client_text_features.t() # [batch, n_cls]
        local_image_logits = sim * self.logit_scale.exp()

        return local_image_logits


# @TRAINER_REGISTRY.register()
class DP_FPL(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DP_FPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DP_FPL.PREC == "fp32" or cfg.TRAINER.DP_FPL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        self.dtype = clip_model.dtype

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        # differential privacy parameters
        max_batch = 0
        for idx in range(0, cfg.DATASET.USERS):
            max_batch = max(max_batch, self.dm.fed_train_loader_x_dict[idx].batch_size)
        if cfg.NOISE > 0:
            q = 1 # random sampling
            delta = 1e-5 # delta
            steps = cfg.OPTIM.ROUND # number of gaussian applications
            sigma = q * math.sqrt(steps * math.log(1/delta)) / cfg.NOISE
            sensitivity = cfg.NORM_THRESH / max_batch # sensitivity
            self.std = sigma * sensitivity

    def forward_pass(self, batch):
        image, label = self.parse_batch_train(batch)
        logits = self.model(image)
        loss = F.cross_entropy(logits.float(), label)

        self.model_zero_grad()
        self.model_backward(loss)

        param_dict = dict(self.model.named_parameters())

        # clip gradient and add noise
        if self.cfg.NOISE > 0:
            grad = param_dict['prompt_learner.global_ctx'].grad.data
            norm = grad.norm(2)
            if norm > self.cfg.NORM_THRESH:
                scale = self.cfg.NORM_THRESH / norm
                scale[scale>1] = 1
                param_dict['prompt_learner.global_ctx'].grad *= scale
            if self.cfg.FACTORIZATION == 'full':
                grad = param_dict['prompt_learner.local_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_ctx'].grad += noise
            else:
                grad = param_dict['prompt_learner.local_u_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_u_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_u_ctx'].grad += noise
                grad = param_dict['prompt_learner.local_v_ctx'].grad.data
                norm = grad.norm(2)
                if norm > self.cfg.NORM_THRESH:
                    scale = self.cfg.NORM_THRESH / norm
                    scale[scale>1] = 1
                    param_dict['prompt_learner.local_v_ctx'].grad *= scale
                noise = torch.normal(0, self.std, size=grad.shape, device=grad.device)
                param_dict['prompt_learner.local_v_ctx'].grad += noise

        if self.cfg.FACTORIZATION == 'lora' or self.cfg.FACTORIZATION == 'dpfpl':
            full_grad = compute_full_grad(param_dict['prompt_learner.local_u_ctx'], param_dict['prompt_learner.local_v_ctx'], self.dtype)
            full_grad = full_grad.type(self.dtype)
            param_dict['prompt_learner.local_ctx'].grad = full_grad

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        return loss_summary

    def backward_pass(self, avg_global_gradient):
        # update global gradient
        param_dict = dict(self.model.named_parameters())
        param_dict['prompt_learner.global_ctx'].grad = avg_global_gradient
        # update
        self.model_update()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

