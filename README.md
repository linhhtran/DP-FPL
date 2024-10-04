# Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models
The implementation of paper Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models.
The code is based on CoOp reqpository available at https://github.com/KaiyangZhou/CoOp/tree/main.

## Installation
Follow the instruction described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install and set up necessary packages and dependencies.

## Data preparation
Follow the instructions [here](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare the following datasets: Caltech101, OxfordPets, OxfordFlowers.

## How to run

### Training parameters

`--root`: a path to all datasets.

`--dataset-config-file`: which dataset config file to use, default to "configs/datasets/caltech101.yaml".

`--num-users`: number of clients in Federated Prompt Learning, default to 10.

`--rank`: factorization rank, default to 8.

`--noise`: differential privacy noise scale, default to 0.4.

### Example run

You can run one instance of DP-FPL using the following command:

```
python federated_main.py --root DATA/ --dataset-config-file configs/datasets/caltech101.yaml --num-users 10 --rank 8 --noise 0.2 --seed 1
```

You can also run multiple instances with different parameters using the script `python run_main.py`.

