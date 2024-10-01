import os

from data_utils import prepare_data_domainNet

# @DATASET_REGISTRY.register()
class DomainNet():
    dataset_dir = "domainnet"
    def __init__(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATASET.ROOT, self.dataset_dir)
        self.num_classes = 10

        train_set, test_set, classnames, lab2cname = prepare_data_domainNet(cfg, cfg.DATASET.ROOT)

        self.federated_train_x = train_set
        self.federated_test_x = test_set
        self.lab2cname = lab2cname
        self.classnames = classnames
