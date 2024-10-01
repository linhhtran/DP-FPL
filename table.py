from prettytable import PrettyTable
import pickle
import os

# Variables
dataset_list = ['caltech101', 'oxford_pets', 'oxford_flowers', 'food101']
factorization_list = ['full', 'fedpgp', 'lora', 'dpfpl']
rank_list = [1, 2, 4, 8]
noise_list = [0.0, 0.4, 0.2, 0.1]
seed_list = [1, 2, 3]
rounds = 100

def read_data(dataset, factorization, rank, noise):
    if factorization == 'full':
        rank = 0
    all_seeds_local, all_seeds_base = [], []
    for seed in seed_list:
        file_name = f'/outputs/{dataset}/acc_{factorization}_{rank}_{noise}_{seed}.pkl'
        if os.path.isfile(file_name):
            local, base, _, _, _, _ = pickle.load(open(file_name, 'rb'))
            acc_len = min(rounds, len(local), len(base))
#            local, base = local[:acc_len], base[:acc_len]
            local, base = local[acc_len-10:acc_len], base[acc_len-10:acc_len]
            all_seeds_local.append(sum(local) / len(local))
            all_seeds_base.append(sum(base) / len(base))
    if len(all_seeds_local) == 0:
        local = 0
    else:
        local = round(sum(all_seeds_local) / len(all_seeds_local), 3)
    if len(all_seeds_base) == 0:
        base = 0
    else:
        base = round(sum(all_seeds_base) / len(all_seeds_base), 3)
    return (local, base)

# read all schemes
def read_scheme(dataset, rank, noise):
    local_list, base_list = [], []
    for factorization in factorization_list:
        local, base = read_data(dataset, factorization, rank, noise)
        local_list.append(local)
        base_list.append(base)
    return local_list, base_list

# Make tables
for dataset in dataset_list:
    local1 = PrettyTable(['rank', 'noise', 'full', 'fedpgp', 'lora', 'dpfpl'])
    base1 = PrettyTable(['rank', 'noise', 'full', 'fedpgp', 'lora', 'dpfpl'])
    for rank in rank_list:
        for i in range(len(noise_list)):
            local_list, base_list = read_scheme(dataset, rank, noise_list[i])
            local1.add_row([rank, noise_list[i], local_list[0], local_list[1], local_list[2], local_list[3]])
            base1.add_row([rank, noise_list[i], base_list[0], base_list[1], base_list[2], base_list[3]])
    print(f'========== {dataset} local accuracy ==========')
    print(local1)
    print(f'========== {dataset} base accuracy ==========')
    print(base1)
    print('\n\n')
