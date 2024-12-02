from typing import List, Tuple, Dict, Optional, Callable
import argparse

import torch
from Dassl.dassl.utils import set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
from Dassl.dassl.utils import read_image

import os
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


from mia_transforms import build_transform


""" 
DEFAULT START

ARGS AS USED IN federated.py.
It's a combination of stuff foud in
- configs/trainers/DP-FPL/vit_b16.yaml
- Dassl/dassl/config/defaults.py
"""

DEFAULT_TRANSFORM_ARGS = dict(
    input_size=(224, 224),
    transform_choices=("random_resized_crop", "random_flip", "normalize"),
    interpolation="bicubic",
    pixel_mean=(0.48145466, 0.4578275, 0.40821073),
    pixel_std=(0.26862954, 0.26130258, 0.27577711),
    crop_padding=4,
    rrcrop_scale=(0.08, 1.0),
    cutout_n=1,
    cutout_len=16,
    gn_mean=0.0,
    gn_std=0.15,
    randaug_n=2,
    randaug_m=10,
    colorjitter_b=0.4,
    colorjitter_c=0.4,
    colorjitter_s=0.4,
    colorjitter_h=0.1,
    rgs_p=0.2,
    gb_k=21,
    gb_p=0.5,
)


"""
DEFAULTS END
"""


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


class AttackModel(nn.Module):
    def __init__(self, total_classes):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(total_classes, 32)
        self.fc2 = nn.Linear(32, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        dataset: List[Tuple[str, int]],
        input_size: Tuple[int],
        interpolation: str,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        normalize_input: bool,
        transform: Optional[List[Callable] | Tuple[Callable]] = None,
        return_raw_img: bool = False,
    ):
        self.dataset = dataset
        transform = transform if transform is not None else []
        if not isinstance(transform, (list, tuple)):
            transform = [transform]
        self.transform = transform
        self.return_raw_img = return_raw_img
        to_tensor = [
            T.Resize(
                input_size,
                interpolation=INTERPOLATION_MODES[interpolation],
            ),
            T.ToTensor(),
        ]
        if normalize_input:
            to_tensor.append(
                T.Normalize(
                    mean=list(pixel_mean),
                    std=pixel_std,
                )
            )
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # image path
        impath, label = self.dataset[idx]
        img_raw = read_image(impath)
        img = img_raw
        for tfm in self.transform:
            img = tfm(img)
        if self.return_raw_img:
            return (self.to_tensor(img_raw), label)
        else:
            return img, label


def extend_cfg(cfg, args):
    """
    Add new config variables.
    """
    from yacs.config import CfgNode as CN

    # Factorization param
    cfg.FACTORIZATION = args.factorization
    cfg.RANK = args.rank

    # Differential privacy param
    cfg.NORM_THRESH = args.norm_thresh
    cfg.NOISE = args.noise

    # Config for DP_FPL
    cfg.TRAINER.NAME = "DP_FPL"
    cfg.TRAINER.DP_FPL = CN()
    cfg.TRAINER.DP_FPL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.DP_FPL.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.DP_FPL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.ROOT = args.root  # dataset path
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.USEALL = args.useall  # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = (
        args.num_shots
    )  # caltech101, dtd, oxford_flowers, oxford_pets, food101
    cfg.DATASET.PARTITION = args.partition  # cifar10, cifar100
    cfg.DATASET.BETA = args.beta  # cifar10, cifar100
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = (
        6 if args.num_users == 6 else 4
    )  # domainnet, office
    if args.useall:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    else:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.num_shots
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    cfg.OPTIM.ROUND = args.round  # global round
    cfg.OPTIM.MAX_EPOCH = 1  # local epoch
    cfg.OPTIM.LR = args.lr  # learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True

    cfg.SEED = args.seed


def setup_cfg(args):
    cfg = (
        get_cfg_default()
    )  # arguments list, type yacs.config.CfgNode _C from defaults.py
    extend_cfg(cfg, args)  # add more arguments

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)  # load dataset

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)  # load model

    cfg.freeze()

    return cfg


def load_target(args):
    dataset = args.dataset_config_file.split("/")[-1].split(".")[0]
    save_filename = f"checkpoints/{dataset}/{args.num_users}_{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pth.tar"
    if not os.path.exists(save_filename):
        return 0, [{} for i in range(args.num_users)], [], []
    checkpoint = torch.load(
        save_filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    local_weights = checkpoint["local_weights"]
    return local_weights


def load_attack(dataset_name, label):
    print(args.noise)
    save_filename = f"checkpoints/{dataset_name}/mia_{label}_{args.noise}.pth.tar"
    attack_model = torch.load(
        save_filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return attack_model


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # target model
    local_trainer = build_trainer(cfg)
    local_weights = load_target(args)
    dataset_name = args.dataset_config_file.split("/")[-1].split(".")[0]
    max_idx = local_trainer.max_idx
    local_trainer.model.load_state_dict(local_weights[max_idx], strict=False)
    local_trainer.set_model_mode("eval")

    # data
    in_samples = local_trainer.mia_in
    out_samples = local_trainer.mia_out
    label_set = set()
    for sample in in_samples:
        label_set.add(sample.label)
    label_list = list(label_set)

    all_correct = []
    for label in label_list:
        print(f"\n\nTraining attack model for label {label}")

        # get data
        dataset = []
        for sample in in_samples:
            if sample.label == label:
                dataset.append((sample.impath, 1))
        for sample in out_samples:
            if sample.label == label:
                dataset.append((sample.impath, 0))

        # TODO: need to configure the transforms here ahead of time.

        # build transforms here so wee don't do it over and over.
        transforms = build_transform(
            **DEFAULT_TRANSFORM_ARGS,
            dataset_name=dataset_name,
            is_train=False,
        )

        dataset = CustomDataset(
            dataset=dataset,
            input_size=DEFAULT_TRANSFORM_ARGS["input_size"],
            transform=transforms,
            interpolation=DEFAULT_TRANSFORM_ARGS["interpolation"],
            pixel_mean=DEFAULT_TRANSFORM_ARGS["pixel_mean"],
            pixel_std=DEFAULT_TRANSFORM_ARGS["pixel_std"],
            normalize_input="normalize" in DEFAULT_TRANSFORM_ARGS["transform_choices"],
        )

        data_loader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
        )

        # attack model
        attack_model = load_attack(dataset_name, label)

        correct = 0
        total = 0
        for target_in, attack_out in data_loader:
            target_in = target_in.to(local_trainer.device)
            attack_out = attack_out.to(local_trainer.device)
            # target inference
            target_out = local_trainer.model_inference(target_in)
            attack_in = F.softmax(target_out, dim=1)
            # attack inference
            pred = attack_model(attack_in)
            _, predicted = torch.max(pred.data, 1)
            print(f'predicted: {predicted}\t label: {attack_out}')
            correct += (predicted.cpu() == attack_out.cpu()).sum()
            total += target_in.size(0)
        print(f'Success rate: {correct / total}\n')
        all_correct.append(correct / total)
        
        
    print('\nAverage MIA success:', (sum(all_correct) / len(all_correct)).item())
    pickle.dump((sum(all_correct) / len(all_correct)).item(), open(f'outputs/{dataset_name}/mia_acc_{args.noise}.pkl', 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round", type=int, default=100, help="number of communication round"
    )
    parser.add_argument("--num-users", type=int, default=10, help="number of users")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument(
        "--train-batch-size", type=int, default=32, help="number of trainer batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=100, help="number of test batch size"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )

    # parameters of factorization and differential privacy
    parser.add_argument(
        "--factorization",
        type=str,
        default="dpfpl",
        help="Choose from: promptfl, fedotp, fedpgp, dplora, dpfpl",
    )
    parser.add_argument("--rank", type=int, default=8, help="matrix factorization rank")
    parser.add_argument(
        "--norm-thresh", type=float, default=10.0, help="clipping norm threshold"
    )
    parser.add_argument(
        "--noise", type=float, default=0.0, help="differential privacy noise scale"
    )

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument(
        "--iid",
        default=False,
        help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd",
    )
    parser.add_argument(
        "--num-shots", type=int, default=16, help="number of shots in few shot setting"
    )
    parser.add_argument(
        "--useall",
        default=True,
        help="is useall, True for all training samples, False for few shot learning",
    )
    # cifar10, cifar100
    parser.add_argument(
        "--partition",
        type=str,
        default="noniid-labeldir",
        help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"',
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="The parameter for the dirichlet distribution for data partitioning",
    )

    # parameters of learnable prompts
    parser.add_argument(
        "--n_ctx", type=int, default=16, help="number of text encoder of text prompts"
    )

    # parameters of path
    parser.add_argument(
        "--root",
        type=str,
        default="datasets/",
        help="path to dataset",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/trainers/DP-FPL/vit_b16.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/caltech101.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--resume", type=str, default="False", help="resume training or not"
    )

    args = parser.parse_args()
    if torch.cuda.is_available():
        print("Number of gpu:", torch.cuda.device_count())
        main(args)
    else:
        print("No gpu")
