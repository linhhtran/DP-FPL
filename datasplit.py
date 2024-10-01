import numpy as np
import random
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        data (float): data.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, data, label=0, domain=0, classname=""):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._data = data
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

class CIFAR10_truncated(torch.utils.data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target, self.label, self.lab2cname, self.classnames = self.__build_truncated_dataset__()
        self.data_detailed = self._convert()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)
        label = []
        for i in range(len(target)):
            label.append(cifar_dataobj.classes[target[i]])

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            label = label[self.dataidxs]

        lab2cname = cifar_dataobj.class_to_idx
        classnames = cifar_dataobj.classes

        return data, target, label, lab2cname, classnames

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            data_idx = self.data[i]
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(data=data_idx, label=int(target_idx), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(torch.utils.data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target, self.label, self.lab2cname, self.classnames= self.__build_truncated_dataset__()
        self.data_detailed = self._convert()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)
        label = []
        for i in range(len(target)):
            label.append(cifar_dataobj.classes[target[i]])

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            label = label[self.dataidxs]

        lab2cname = cifar_dataobj.class_to_idx
        classnames = cifar_dataobj.classes

        return data, target, label, lab2cname, classnames

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            data_idx = self.data[i]
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(data=data_idx, label=int(target_idx), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        label = self.label[index]

        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label

    def __len__(self):
        return len(self.data)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    train_data = cifar10_train_ds.data_detailed
    test_data = cifar10_test_ds.data_detailed

    lab2cname = cifar10_train_ds.lab2cname
    classnames = cifar10_train_ds.classnames

    return (X_train, y_train, X_test, y_test, train_data, test_data, lab2cname, classnames)


def load_cifar100_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    train_data = cifar100_train_ds.data_detailed
    test_data = cifar100_test_ds.data_detailed

    lab2cname = cifar100_train_ds.lab2cname
    classnames = cifar100_train_ds.classnames

    return (X_train, y_train, X_test, y_test, train_data, test_data, lab2cname, classnames)

def renormalize(weights, index):
    """
    :param weights: vector of non negative weights summing to 1.
    :type weights: numpy.array
    :param index: index of the weight to remove
    :type index: int
    """
    renormalized_weights = np.delete(weights, index)
    renormalized_weights /= renormalized_weights.sum()

    return renormalized_weights

def partition_data(dataset, datadir, partition, n_parties, beta=0.4):

    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = load_cifar10_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test, data_train, data_test, lab2cname, classnames = load_cifar100_data(datadir)
        y = np.concatenate([y_train, y_test], axis=0)

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if partition == "homo":
        idxs_train = np.random.permutation(n_train)
        idxs_test = np.random.permutation(n_test)

        batch_idxs_train = np.array_split(idxs_train, n_parties)
        batch_idxs_test = np.array_split(idxs_test, n_parties)

        net_dataidx_map_train = {i: batch_idxs_train[i] for i in range(n_parties)}
        net_dataidx_map_test = {i: batch_idxs_test[i] for i in range(n_parties)}

    elif partition == "iid-label100":
        seed = 12345
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples_train = y_train.shape[0]
        n_samples_test = y_test.shape[0]

        selected_indices_train = rng.sample(list(range(n_samples_train)), n_samples_train)
        selected_indices_test = rng.sample(list(range(n_samples_test)), n_samples_test)

        n_samples_by_client_train = int((n_samples_train / n_parties) // 5)
        n_samples_by_client_test = int((n_samples_test / n_parties) // 5)

        indices_by_fine_labels_train = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_train = {k: list() for k in range(n_coarse_labels)}

        indices_by_fine_labels_test = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels_test = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices_train:
            fine_label = y_train[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_train[fine_label].append(idx)
            indices_by_coarse_labels_train[coarse_label].append(idx)

        for idx in selected_indices_test:
            fine_label = y_test[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels_test[fine_label].append(idx)
            indices_by_coarse_labels_test[coarse_label].append(idx)

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_train[fine_label]), n_samples_by_client_train)
                net_dataidx_map_train[client_idx] = np.append(net_dataidx_map_train[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_train[fine_label].remove(idx)

        for client_idx in range(n_parties):
            coarse_idx = client_idx // 5
            fine_idx = fine_labels_by_coarse_labels[coarse_idx]
            for k in range(5):
                fine_label = fine_idx[k]
                sample_idx = rng.sample(list(indices_by_fine_labels_test[fine_label]), n_samples_by_client_test)
                net_dataidx_map_test[client_idx] = np.append(net_dataidx_map_test[client_idx], sample_idx)
                for idx in sample_idx:
                    indices_by_fine_labels_test[fine_label].remove(idx)

    elif partition == "noniid-labeluni":
        if dataset == "cifar10":
            num = 2
            K = 10
        elif dataset == "cifar100":
            num = 10
            K = 100

        # -------------------------------------------#
        # Divide classes + num samples for each user #
        # -------------------------------------------#
        assert (num * n_parties) % K == 0, "equal classes appearance is needed"
        count_per_class = (num * n_parties) // K
        class_dict = {}
        for i in range(K):
            # sampling alpha_i_c
            probs = np.random.uniform(0.4, 0.6, size=count_per_class)
            # normalizing
            probs_norm = (probs / probs.sum()).tolist()
            class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

        # -------------------------------------#
        # Assign each client with data indexes #
        # -------------------------------------#
        class_partitions = defaultdict(list)
        for i in range(n_parties):
            c = []
            for _ in range(num):
                class_counts = [class_dict[i]['count'] for i in range(K)]
                max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
                c.append(np.random.choice(max_class_counts))
                class_dict[c[-1]]['count'] -= 1
            class_partitions['class'].append(c)
            class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

        # -------------------------- #
        # Create class index mapping #
        # -------------------------- #
        data_class_idx_train = {i: np.where(y_train == i)[0] for i in range(K)}
        data_class_idx_test = {i: np.where(y_test == i)[0] for i in range(K)}

        num_samples_train = {i: len(data_class_idx_train[i]) for i in range(K)}
        num_samples_test = {i: len(data_class_idx_test[i]) for i in range(K)}

        # --------- #
        # Shuffling #
        # --------- #
        for data_idx in data_class_idx_train.values():
            random.shuffle(data_idx)
        for data_idx in data_class_idx_test.values():
            random.shuffle(data_idx)

        # ------------------------------ #
        # Assigning samples to each user #
        # ------------------------------ #
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

        for usr_i in range(n_parties):
            for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
                end_idx_train = int(num_samples_train[c] * p)
                end_idx_test = int(num_samples_test[c] * p)

                net_dataidx_map_train[usr_i] = np.append(net_dataidx_map_train[usr_i],
                                                         data_class_idx_train[c][:end_idx_train])
                net_dataidx_map_test[usr_i] = np.append(net_dataidx_map_test[usr_i],
                                                        data_class_idx_test[c][:end_idx_test])
                data_class_idx_train[c] = data_class_idx_train[c][end_idx_train:]
                data_class_idx_test[c] = data_class_idx_test[c][end_idx_test:]

    elif partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        if dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100

        N_train = y_train.shape[0]
        N_test = y_test.shape[0]
        net_dataidx_map_train = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            for k in range(K):
                train_idx_k = np.where(y_train == k)[0]
                test_idx_k = np.where(y_test == k)[0]

                np.random.shuffle(train_idx_k)
                np.random.shuffle(test_idx_k)

                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array \
                    ([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
                proportions = proportions / proportions.sum()
                proportions_train = (np.cumsum(proportions) * len(train_idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions) * len(test_idx_k)).astype(int)[:-1]
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(train_idx_k, proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(test_idx_k, proportions_test))]

                min_size_train = min([len(idx_j) for idx_j in idx_batch_train])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])
                min_size = min(min_size_train, min_size_test)

        for j in range(n_parties):
            np.random.shuffle(idx_batch_train[j])
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_train[j] = idx_batch_train[j]
            net_dataidx_map_test[j] = idx_batch_test[j]

    elif partition == "noniid-labeldir100":
        seed = 12345
        alpha = 10
        n_fine_labels = 100
        n_coarse_labels = 20
        coarse_labels = \
            np.array([
                4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                18, 1, 2, 15, 6, 0, 17, 8, 14, 13
            ])

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        rng = random.Random(rng_seed)
        np.random.seed(rng_seed)

        n_samples = y.shape[0]

        selected_indices = rng.sample(list(range(n_samples)), n_samples)

        n_samples_by_client = n_samples // n_parties

        indices_by_fine_labels = {k: list() for k in range(n_fine_labels)}
        indices_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for idx in selected_indices:
            fine_label = y[idx]
            coarse_label = coarse_labels[fine_label]

            indices_by_fine_labels[fine_label].append(idx)
            indices_by_coarse_labels[coarse_label].append(idx)

        available_coarse_labels = [ii for ii in range(n_coarse_labels)]

        fine_labels_by_coarse_labels = {k: list() for k in range(n_coarse_labels)}

        for fine_label, coarse_label in enumerate(coarse_labels):
            fine_labels_by_coarse_labels[coarse_label].append(fine_label)

        net_dataidx_map = [[] for i in range(n_parties)]

        for client_idx in range(n_parties):
            coarse_labels_weights = np.random.dirichlet(alpha=beta * np.ones(len(fine_labels_by_coarse_labels)))
            weights_by_coarse_labels = dict()

            for coarse_label, fine_labels in fine_labels_by_coarse_labels.items():
                weights_by_coarse_labels[coarse_label] = np.random.dirichlet(alpha=alpha * np.ones(len(fine_labels)))

            for ii in range(n_samples_by_client):
                coarse_label_idx = int(np.argmax(np.random.multinomial(1, coarse_labels_weights)))
                coarse_label = available_coarse_labels[coarse_label_idx]
                fine_label_idx = int(np.argmax(np.random.multinomial(1, weights_by_coarse_labels[coarse_label])))
                fine_label = fine_labels_by_coarse_labels[coarse_label][fine_label_idx]
                sample_idx = int(rng.choice(list(indices_by_fine_labels[fine_label])))

                net_dataidx_map[client_idx] = np.append(net_dataidx_map[client_idx], sample_idx)

                indices_by_fine_labels[fine_label].remove(sample_idx)
                indices_by_coarse_labels[coarse_label].remove(sample_idx)


                if len(indices_by_fine_labels[fine_label]) == 0:
                    fine_labels_by_coarse_labels[coarse_label].remove(fine_label)

                    weights_by_coarse_labels[coarse_label] = renormalize(weights_by_coarse_labels[coarse_label]
                                                                         ,fine_label_idx)

                    if len(indices_by_coarse_labels[coarse_label]) == 0:
                        fine_labels_by_coarse_labels.pop(coarse_label, None)
                        available_coarse_labels.remove(coarse_label)

                        coarse_labels_weights = renormalize(coarse_labels_weights, coarse_label_idx)

        random.shuffle(net_dataidx_map)
        net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i, index in enumerate(net_dataidx_map):
            net_dataidx_map_train[i] = np.append(net_dataidx_map_train[i], index[index < 50_000]).astype(int)
            net_dataidx_map_test[i] = np.append(net_dataidx_map_test[i], index[index >= 50_000 ] -50000).astype(int)

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        if dataset == 'cifar10':
            K = 10
        elif dataset == "cifar100":
            K = 100

        if num == 10:
            net_dataidx_map_train ={i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k_train = np.where(y_train == i)[0]
                idx_k_test = np.where(y_test == i)[0]

                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)

                train_split = np.array_split(idx_k_train, n_parties)
                test_split = np.array_split(idx_k_test, n_parties)
                for j in range(n_parties):
                    net_dataidx_map_train[j] = np.append(net_dataidx_map_train[j], train_split[j])
                    net_dataidx_map_test[j] = np.append(net_dataidx_map_test[j], test_split[j])
        else:
            times = [0 for i in range(10)]
            contain = []
            for i in range(n_parties):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while (j < num):
                    ind = random.randint(0, K - 1)
                    if (ind not in current):
                        j = j + 1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map_train = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
            net_dataidx_map_test = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}

            for i in range(K):
                idx_k_train = np.where(y_train == i)[0]
                idx_k_test = np.where(y_test == i)[0]

                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)

                train_split = np.array_split(idx_k_train, times[i])
                test_split = np.array_split(idx_k_test, times[i])

                ids = 0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map_train[j] = np.append(net_dataidx_map_train[j], train_split[ids])
                        net_dataidx_map_test[j] = np.append(net_dataidx_map_test[j], test_split[ids])
                        ids += 1

    return (data_train, data_test, lab2cname, classnames, net_dataidx_map_train, net_dataidx_map_test)
