# Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models
The implementation of paper Privacy-Preserving Personalized Federated Prompt Learning for Multimodal Large Language Models.

## How to Run

You can run `federated_main.py` with some specified arguments.

## Data Preparation
Please follow the instructions at CoOP https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md to prepare the following datasets: Caltech101, OxfordPets, OxfordFlowers, Food101.

### Training

`--root` takes as input a path to dataset.

`--config-file` means which config file to use.

You can select variables like shots, users by changing `cfg` or you can change every arguments you like in scripts.

### Running example
`bash scripts/plt_few_shot.sh`

