import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict

from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum
from Dassl.dassl.utils import read_json

from .oxford_pets import OxfordPets


# @DATASET_REGISTRY.register()
class OxfordFlowers(DatasetBase):

    dataset_dir = "oxford_flowers"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.read_data()
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        if cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_dataset(train, num_shots=cfg.DATASET.NUM_SHOTS,
                                                                num_users=cfg.DATASET.USERS,
                                                                is_iid=cfg.DATASET.IID,
                                                                repeat_rate=0)
        elif not cfg.DATASET.USEALL:
            federated_train_x = self.generate_federated_fewshot_dataset(train, num_shots=cfg.DATASET.NUM_SHOTS,
                                                                        num_users=cfg.DATASET.USERS,
                                                                        is_iid=cfg.DATASET.IID,
                                                                        repeat_rate=0)
        federated_test_x = self.generate_federated_dataset(test, num_shots=cfg.DATASET.NUM_SHOTS,
                                                            num_users=cfg.DATASET.USERS,
                                                            is_iid=cfg.DATASET.IID,
                                                            repeat_rate=0)


        super().__init__(total_train_x=train, federated_train_x=federated_train_x, federated_test_x=federated_test_x)

    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        print("Splitting data into 50% train, 20% val, and 30% test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test