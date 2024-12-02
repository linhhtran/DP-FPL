import argparse
import torch
from Dassl.dassl.utils import set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer

import os
import math
import copy
import pickle
import numpy as np

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
    cfg.TRAINER.NAME = 'DP_FPL'
    cfg.TRAINER.DP_FPL = CN()
    cfg.TRAINER.DP_FPL.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.DP_FPL.PREC = "fp32"  # fp16, fp32, amp
    cfg.TRAINER.DP_FPL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.DATASET.ROOT = args.root # dataset path
    cfg.DATASET.USERS = args.num_users # number of clients
    cfg.DATASET.IID = args.iid  # is iid
    cfg.DATASET.USEALL = args.useall # use all data for training instead of few shot
    cfg.DATASET.NUM_SHOTS = args.num_shots # caltech101, dtd, oxford_flowers, oxford_pets, food101
    cfg.DATASET.PARTITION = args.partition # cifar10, cifar100
    cfg.DATASET.BETA = args.beta # cifar10, cifar100
    cfg.DATALOADER.TRAIN_X.N_DOMAIN = 6 if args.num_users == 6 else 4 # domainnet, office
    if args.useall:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    else:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.num_shots
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    cfg.OPTIM.ROUND = args.round # global round
    cfg.OPTIM.MAX_EPOCH = 1 # local epoch
    cfg.OPTIM.LR = args.lr # learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True

    cfg.SEED = args.seed


def setup_cfg(args):
    cfg = get_cfg_default() # arguments list, type yacs.config.CfgNode _C from defaults.py
    extend_cfg(cfg, args) # add more arguments

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file) # load dataset

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file) # load model

    cfg.freeze()

    return cfg

def save_checkpoint(args, epoch, local_weights, local_acc, neighbor_acc):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_filename = os.path.join(os.getcwd(), f'checkpoints/{dataset}/{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pth.tar')
    state = {
        "epoch": epoch + 1,
        "local_weights": local_weights,
        "local_acc": local_acc,
        "neighbor_acc": neighbor_acc,
    }
    torch.save(state, save_filename)

def load_checkpoint(args):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_filename = os.path.join(os.getcwd(), f'/checkpoints/{dataset}/{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pth.tar')
    if not os.path.exists(save_filename):
        return 0, [{} for i in range(args.num_users)], [], []
    checkpoint = torch.load(save_filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    epoch = checkpoint["epoch"]
    local_weights = checkpoint["local_weights"]
    local_acc = checkpoint["local_acc"]
    neighbor_acc = checkpoint["neighbor_acc"]
    return epoch, local_weights, local_acc, neighbor_acc


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    dirichlet = False
    if args.dataset_config_file.split('/')[-1].split('.')[0] in ['cifar10', 'cifar100']:
        dirichlet = True

    global_gradients = [{} for i in range(args.num_users)]
    local_weights = [{} for i in range(args.num_users)]
    local_weights_g = [[] for i in range(args.num_users)]
    local_weights_l = [[] for i in range(args.num_users)]
    local_weights_u = [[] for i in range(args.num_users)]
    local_weights_v = [[] for i in range(args.num_users)]

    local_trainer = build_trainer(cfg)
    initial_weights = copy.deepcopy(local_trainer.model.state_dict())

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    local_acc_list, neighbor_acc_list, = [], []
    if args.resume == 'True':
        start_epoch, local_weights, local_acc_list, neighbor_acc_list = load_checkpoint(args)
        print('Resume from epoch', start_epoch)
    if start_epoch == max_epoch - 1:
        return
    if args.noise > 0:
        std = local_trainer.std / cfg.DATASET.USERS
    for epoch in range(start_epoch, max_epoch): # global communication loop
        idxs_users = list(range(0,cfg.DATASET.USERS))
        print("------------local train start epoch:", epoch, "-------------")

        # create data iters
        data_iters = []
        for idx in idxs_users:
            local_trainer.set_model_mode("train")
            loader = local_trainer.fed_train_loader_x_dict[idx]
            data_iters.append(iter(loader))
        max_batch = len(loader)

        # loop through batches
        for batch in range(0, max_batch):
            local_trainer.set_model_mode("train")
            for idx in idxs_users:
                if epoch == 0:
                    local_trainer.model.load_state_dict(initial_weights, strict=False)
                else:
                    local_trainer.model.load_state_dict(local_weights[idx], strict=False)
                # train
                local_trainer.train_forward(idx=idx, train_iter=data_iters[idx])

                local_weight = local_trainer.model.state_dict()
                global_gradients[idx] = local_trainer.model.prompt_learner.global_ctx.grad.data
                local_weights_g[idx] = copy.deepcopy(local_weight['prompt_learner.global_ctx'])
                if args.factorization in ['fedotp', 'dplora', 'dpfpl']:
                    local_weights_l[idx] = copy.deepcopy(local_weight['prompt_learner.local_ctx'])
                if args.factorization in ['fedpgp', 'dplora', 'dpfpl']:
                    local_weights_u[idx] = copy.deepcopy(local_weight['prompt_learner.local_u_ctx'])
                    local_weights_v[idx] = copy.deepcopy(local_weight['prompt_learner.local_v_ctx'])

            # average gradient
            avg_global_gradient = sum(global_gradients) / cfg.DATASET.USERS
            if args.noise > 0:
                noise = torch.normal(0, std, size=avg_global_gradient.shape, device=avg_global_gradient.device)
                avg_global_gradient += noise

            # backward and update
            for idx in idxs_users:
                local_weights[idx]['prompt_learner.global_ctx'] = local_weights_g[idx]
                if args.factorization in ['fedotp', 'dplora', 'dpfpl']:
                    local_weights[idx]['prompt_learner.local_ctx'] = local_weights_l[idx]
                if args.factorization in ['fedpgp', 'dplora', 'dpfpl']:
                    local_weights[idx]['prompt_learner.local_u_ctx'] = local_weights_u[idx]
                    local_weights[idx]['prompt_learner.local_v_ctx'] = local_weights_v[idx]

                local_trainer.model.load_state_dict(local_weights[idx], strict=False)
                local_trainer.train_backward(avg_global_gradient=avg_global_gradient)

                local_weight = local_trainer.model.state_dict()
                local_weights_g[idx] = copy.deepcopy(local_weight['prompt_learner.global_ctx'])
                if args.factorization in ['fedotp', 'dplora', 'dpfpl']:
                    local_weights_l[idx] = copy.deepcopy(local_weight['prompt_learner.local_ctx'])
                if args.factorization in ['fedpgp', 'dplora', 'dpfpl']:
                    local_weights_u[idx] = copy.deepcopy(local_weight['prompt_learner.local_u_ctx'])
                    local_weights_v[idx] = copy.deepcopy(local_weight['prompt_learner.local_v_ctx'])

        # test
        print("------------local test start-------------")
        local_trainer.set_model_mode("eval")
        results_local, results_neighbor = [], []
        for idx in idxs_users:
            local_weights[idx]['prompt_learner.global_ctx'] = local_weights_g[idx]
            if args.factorization in ['fedotp', 'dplora', 'dpfpl']:
                local_weights[idx]['prompt_learner.local_ctx'] = local_weights_l[idx]
            if args.factorization in ['fedpgp', 'dplora', 'dpfpl']:
                local_weights[idx]['prompt_learner.local_u_ctx'] = local_weights_u[idx]
                local_weights[idx]['prompt_learner.local_v_ctx'] = local_weights_v[idx]

            local_trainer.model.load_state_dict(local_weights[idx], strict=False)

            results_local.append(local_trainer.test(idx=idx, split='local'))
            if not dirichlet:
                results_neighbor.append(local_trainer.test(idx=idx, split='neighbor'))

        local_acc, neighbor_acc = [], []
        for k in range(len(results_local)):
            local_acc.append(results_local[k][0])
            if not dirichlet:
                neighbor_acc.append(results_neighbor[k][0])
        local_acc_list.append(sum(local_acc)/len(local_acc))
        print(f"Global test local acc:", sum(local_acc)/len(local_acc))
        if not dirichlet:
            neighbor_acc_list.append(sum(neighbor_acc)/len(neighbor_acc))
            print(f"Global test neighbor acc:", sum(neighbor_acc)/len(neighbor_acc))
        print("------------local test finish-------------")
        print(f"Epoch: {epoch}/{max_epoch}\tfinished batch : {batch}/{max_batch}")

        # save checkpoint
        save_checkpoint(args, epoch, local_weights, local_acc_list, neighbor_acc_list)
        dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
        pickle.dump([local_acc_list, neighbor_acc_list],
                    open(os.path.join(os.getcwd(), f'outputs/{dataset_name}/acc_{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pkl'), 'wb'))

    print("maximum test local acc:", max(local_acc_list))
    print("mean of local acc:",np.mean(local_acc_list[-5:]))
    if not dirichlet:
        print("maximum test neighbor acc:", max(neighbor_acc_list))
        print("mean of neighbor acc:",np.mean(neighbor_acc_list[-5:]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=100, help="number of communication round")
    parser.add_argument('--num-users', type=int, default=10, help="number of users")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-batch-size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test-batch-size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    # parameters of factorization and differential privacy
    parser.add_argument('--factorization', type=str, default='dpfpl', help='Choose from: promptfl, fedotp, fedpgp, dplora, dpfpl')
    parser.add_argument('--rank', type=int, default=8, help='matrix factorization rank')
    parser.add_argument('--norm-thresh', type=float, default=10.0, help='clipping norm threshold')
    parser.add_argument('--noise', type=float, default=0.0, help='differential privacy noise scale')

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num-shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=True, help="is useall, True for all training samples, False for few shot learning")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.3, help='The parameter for the dirichlet distribution for data partitioning')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")

    # parameters of path
    parser.add_argument("--root", type=str, default="/datasets", help="path to dataset")
    parser.add_argument("--config-file", type=str, default="configs/trainers/DP-FPL/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar100.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default="False", help="resume training or not")

    args = parser.parse_args()
    main(args)

