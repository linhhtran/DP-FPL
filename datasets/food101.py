import os
import pickle

# from Dassl.dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from Dassl.dassl.data.datasets.base_dataset import DatasetBase, Datum

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


# @DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "food-101"

    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
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