import argparse
import torch
import numpy as np
import pickle
import random

from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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
    def __init__(self, data_samples):
        label_set = set()
        train_data = []
        test_data = []
        for sample in data_samples:
            label_set.add(sample[2].item())
            if sample[1] == 1:
                train_data.append(sample)
            else:
                test_data.append(sample)
        self.label_set = list(label_set)
        random.shuffle(train_data)
        random.shuffle(test_data)

        # rebalance dataset
        train_data = train_data[:min(len(train_data), len(test_data))]
        test_data = test_data[:min(len(train_data), len(test_data))]
        self.dataset = train_data + test_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Each sample is a tuple (prediction, membership, label)
        return self.dataset[idx]


def main(args):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # data
    train_data = []
    for seed in range(0, 50):
        train_data += pickle.load(
            open(
                f"outputs/{args.dataset}/shadow_{args.noise}_{seed}.pkl",
                "rb",
            )
        )
    total_classes = len(train_data[0][0])
    train_data = CustomDataset(train_data)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    num_classes = len(train_data.label_set)

    # params
    # we need to place more weight on the minority clas. So flip the balance and assign more weight to smaller samples.
    criterion = torch.nn.CrossEntropyLoss()
    max_epoch = 200

    for i in range(num_classes):
        print(f"Train attack model for class {train_data.label_set[i]}", flush=True)
        attack_model = AttackModel(total_classes)
        optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)

        best_so_far = torch.inf
        best_model = None

        for epoch in range(0, max_epoch):
            print(f"Epoch {epoch}/{max_epoch}")
            # train
            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            for input, output, label in train_loader:
                if label == train_data.label_set[i]:
                    # Forward pass
                    pred = attack_model(input)
                    loss = criterion(pred, output)
                    epoch_pred.append(torch.max(pred.data, 1)[1].item())
                    epoch_true.append(output.item())
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

            #log_msg = f'predicted: {epoch_pred}\t true: {epoch_true}\n'
            log_msg = f"Loss: {epoch_loss}"

            if epoch_loss < best_so_far:
                log_msg += " best so far"
                best_so_far = epoch_loss
                best_model = attack_model.cpu()
            print(log_msg, flush=True)

        torch.save(
            best_model,
            f"checkpoints/{args.dataset}/mia_{train_data.label_set[i]}_{args.noise}.pth.tar",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="caltech101", help="dataset")

    parser.add_argument(
        "--noise", type=float, default=0.0, help="differential privacy noise scale"
    )

    args = parser.parse_args()
    if torch.cuda.is_available():
        print("Number of gpu:", torch.cuda.device_count())
        main(args)
    else:
        print("No gpu")
