import os

def run(root, dataset, users, rank, noise, seed):
    dataset_yaml = f'configs/datasets/{dataset}.yaml'
    os.system(f'srun_main.sh {root} {dataset_yaml} {users} {rank} {noise} {seed}')

# variables
seed_list = [1, 2, 3]
dataset_list = ['caltech101', 'oxford_pets', 'oxford_flowers']
rank_list = [1, 2, 4, 8]
noise_list = [0.0, 0.1, 0.2, 0.4]

root = 'DATA/' # change to your dataset path
users = 10

for seed in seed_list:
    for dataset in dataset_list:
        for rank in rank_list:
            for noise in noise_list:
                run(root, dataset, users, rank, noise, seed)
