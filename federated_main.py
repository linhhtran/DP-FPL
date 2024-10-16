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

def save_checkpoint(args, epoch, local_trainers, local_acc, neighbor_acc, local_err, neighbor_err, local_f1, neighbor_f1):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_filename = os.path.join(os.getcwd(), f'checkpoints/{dataset}/{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pth.tar')
    state = {
        "epoch": epoch + 1,
        "local_trainers": local_trainers,
        "local_acc": local_acc,
        "neighbor_acc": neighbor_acc,
        "local_err": local_err,
        "neighbor_err": neighbor_err,
        "local_f1": local_f1,
        "neighbor_f1": neighbor_f1,
    }
    torch.save(state, save_filename)

def load_checkpoint(args):
    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    save_filename = os.path.join(os.getcwd(), f'/checkpoints/{dataset}/{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pth.tar')
    checkpoint = torch.load(save_filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    epoch = checkpoint["epoch"]
    local_trainers = checkpoint["local_trainers"]
    local_acc = checkpoint["local_acc"]
    neighbor_acc = checkpoint["neighbor_acc"]
    local_err = checkpoint["local_err"]
    neighbor_err = checkpoint["neighbor_err"]
    local_f1 = checkpoint["local_f1"]
    neighbor_f1 = checkpoint["neighbor_f1"]
    return epoch, local_trainers, local_acc, neighbor_acc, local_err, neighbor_err, local_f1, neighbor_f1


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    dataset = args.dataset_config_file.split('/')[-1].split('.')[0]
    directories = ['checkpoints', f'checkpoints/{dataset}', 'outputs', f'outputs/{dataset}']
    for directory in directories:
        if not os.path.exists(os.path.join(os.getcwd(), directory)):
            os.makedirs(os.path.join(os.getcwd(), directory))

    global_gradients = [{} for i in range(args.num_users)]
    local_trainers = []
    initial_trainer = build_trainer(cfg)
    initial_weights = copy.deepcopy(initial_trainer.model.state_dict())
    for i in range(args.num_users):
        local_trainer = build_trainer(cfg)
        local_trainer.model.load_state_dict(initial_weights, strict=False)
        local_trainers.append(local_trainer)

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    local_acc_list, neighbor_acc_list, local_err_list, neighbor_err_list, local_f1_list, neighbor_f1_list = [], [], [], [], [], []
    if args.resume == 'True':
        start_epoch, local_trainers, local_acc_list, neighbor_acc_list, local_err_list, neighbor_err_list, local_f1_list, neighbor_f1_list = load_checkpoint(args)
        print('Resume from epoch', start_epoch)
    if start_epoch == max_epoch - 1:
        return
    if args.noise > 0:
        std = local_trainers[0].std / cfg.DATASET.USERS
    for epoch in range(start_epoch, max_epoch): # global communication loop
        idxs_users = list(range(0,cfg.DATASET.USERS))
        print("------------local train start epoch:", epoch, "-------------")

        # create data iter
        for idx in idxs_users:
            local_trainers[idx].create_data_iter(idx=idx)
        max_batch = local_trainers[0].num_batches

        # loop through batches
        for batch in range(0, max_batch):
            # train
            for idx in idxs_users:
                local_trainers[idx].train_forward(idx=idx)
                global_gradients[idx] = local_trainers[idx].model.prompt_learner.global_ctx.grad.data

            print("------------local train finish epoch:", epoch, "-------------")

            # average gradient
            avg_global_gradient = sum(global_gradients) / cfg.DATASET.USERS
            if args.noise > 0:
                noise = torch.normal(0, std, size=avg_global_gradient.shape, device=avg_global_gradient.device)
                avg_global_gradient += noise

            # backward and update
            for idx in idxs_users:
                local_trainers[idx].train_backward(avg_global_gradient=avg_global_gradient)

            # test
            print("------------local test start-------------")
            results_local, results_neighbor = [], []
            for idx in idxs_users:
                results_local.append(local_trainers[idx].test(idx=idx, split='local'))
                results_neighbor.append(local_trainers[idx].test(idx=idx, split='neighbor'))

            local_acc, neighbor_acc, local_err, neighbor_err, local_f1, neighbor_f1 = [], [], [], [], [], []
            for k in range(len(results_local)):
                local_acc.append(results_local[k][0])
                neighbor_acc.append(results_neighbor[k][0])
                local_err.append(results_local[k][1])
                neighbor_err.append(results_neighbor[k][1])
                local_f1.append(results_local[k][2])
                neighbor_f1.append(results_neighbor[k][2])
            local_acc_list.append(sum(local_acc)/len(local_acc))
            neighbor_acc_list.append(sum(neighbor_acc)/len(neighbor_acc))
            local_err_list.append(sum(local_err)/len(local_err))
            neighbor_err_list.append(sum(neighbor_err)/len(neighbor_err))
            local_f1_list.append(sum(local_f1)/len(local_f1))
            neighbor_f1_list.append(sum(neighbor_f1)/len(neighbor_f1))
            print(f"Global test local acc:", sum(local_acc)/len(local_acc))
            print(f"Global test neighbor acc:", sum(neighbor_acc)/len(neighbor_acc))
            print("------------local test finish-------------")
            print(f"Epoch: {epoch}/{max_epoch}\tfinished batch : {batch}/{max_batch}")

        # delete data iter
        for idx in idxs_users:
            local_trainers[idx].delete_data_iter()

        # update learning rate
        for idx in idxs_users:
            local_trainers[idx].update_lr()
        # save checkpoint
        # uncomment if want to save checkpoint, be aware of disk quota issue
        # save_checkpoint(args, epoch, local_trainers, local_acc_list, neighbor_acc_list, local_err_list, neighbor_err_list, local_f1_list, neighbor_f1_list)
        dataset_name = args.dataset_config_file.split('/')[-1].split('.')[0]
        pickle.dump([local_acc_list, neighbor_acc_list, local_err_list, neighbor_err_list, local_f1_list, neighbor_f1_list],
                    open(os.path.join(os.getcwd(), f'outputs/{dataset_name}/acc_{args.factorization}_{args.rank}_{args.noise}_{args.seed}.pkl'), 'wb'))

    print("maximum test local acc:", max(local_acc_list))
    print("mean of local acc:",np.mean(local_acc_list[-5:]))
    print("maximum test neighbor acc:", max(neighbor_acc_list))
    print("mean of neighbor acc:",np.mean(neighbor_acc_list[-5:]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, default=100, help="number of communication round")
    parser.add_argument('--num-users', type=int, default=10, help="number of users")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--train-batch-size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test-batch-size', type=int, default=100, help="number of test batch size")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    # parameters of factorization and differential privacy
    parser.add_argument('--factorization', type=str, default='dpfpl', help='Choose from: full, fedpgp, lora, dpfpl')
    parser.add_argument('--rank', type=int, default=8, help='matrix factorization rank')
    parser.add_argument('--norm-thresh', type=float, default=10.0, help='clipping norm threshold')
    parser.add_argument('--noise', type=float, default=0.4, help='differential privacy noise scale')

    # parameters of datasets
    # caltech101, oxford_flowers, oxford_pets, food101 and dtd
    parser.add_argument('--iid', default=False, help="is iid, control the iid of caltech101, oxford_flowers, oxford_pets, food101 and dtd")
    parser.add_argument('--num-shots', type=int, default=16, help="number of shots in few shot setting")
    parser.add_argument('--useall', default=False, help="is useall, True for all training samples, False for few shot learning")
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='the data partitioning strategy of cifar10 and cifar100, select from "homo, noniid-labeluni, noniid-labeldir,noniid-labeldir100"')
    parser.add_argument('--beta', type=float, default=0.3, help='The parameter for the dirichlet distribution for data partitioning')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=16, help="number of text encoder of text prompts")

    # parameters of path
    parser.add_argument("--root", type=str, default="/datasets", help="path to dataset")
    parser.add_argument("--config-file", type=str, default="configs/trainers/DP-FPL/vit_b16.yaml", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/caltech101.yaml", help="path to config file for dataset setup")
    parser.add_argument("--resume", type=str, default=None, help="resume training or not")

    args = parser.parse_args()
    if torch.cuda.is_available():
        print('Number of gpu:', torch.cuda.device_count())
    else:
        print('Warning: no gpu')
    main(args)
