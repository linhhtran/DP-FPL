import os

def run(dataset, factorization, rank, noise, seed):
    dataset_yaml = f'configs/datasets/{dataset}.yaml'
    os.system(f'srun_main.sh {dataset_yaml} {factorization} {rank} {noise} {seed}')

# variables
dataset_list = ['caltech101', 'oxford_pets', 'oxford_flowers', 'food101']
rank_list = [1, 2, 4, 8]
noise_list = [0.0, 0.1, 0.2, 0.4]
seed_list = [1, 2, 3]

for seed in seed_list:
    for dataset in dataset_list:
        for noise in noise_list:
            run(dataset, 'full', 0, noise, seed)
            for rank in rank_list:
                run(dataset, 'fedpgp', rank, noise, seed)
                run(dataset, 'lora', rank, noise, seed)
                run(dataset, 'dpfpl', rank, noise, seed)
