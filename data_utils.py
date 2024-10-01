# This code is used to generate non-iid data with Feature Skew

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from collections import Counter

class Datum:
    """Data instance which defines the basic attributes.

    Args:
        data (float): data.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath, label=0, domain=0, classname=""):
        # assert isinstance(impath, str)
        # assert check_isfile(impath)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


def prepare_data_domainNet(cfg, data_base_path):
    data_base_path = data_base_path
    transform_train = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
    ])

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    lab2cname = clipart_trainset.lab2cname
    classnames = clipart_trainset.classnames
    clipart_trainset = clipart_trainset.data_detailed
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False).data_detailed
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train).data_detailed
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False).data_detailed
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train).data_detailed
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False).data_detailed
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train).data_detailed
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False).data_detailed
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train).data_detailed
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False).data_detailed
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train).data_detailed
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False).data_detailed

    train_data_num_list = [len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset), len(real_trainset), len(sketch_trainset)]
    test_data_num_list = [len(clipart_testset), len(infograph_testset), len(painting_testset), len(quickdraw_testset), len(real_testset), len(sketch_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set = [clipart_trainset, infograph_trainset, painting_trainset, quickdraw_trainset, real_trainset, sketch_trainset]
    test_set = [clipart_testset, infograph_testset, painting_testset, quickdraw_testset, real_testset, sketch_testset]
    return train_set, test_set, classnames, lab2cname

def prepare_data_office(cfg, data_base_path):
    data_base_path = data_base_path
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
    ])

    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    lab2cname = amazon_trainset.lab2cname
    classnames = amazon_trainset.classnames
    amazon_trainset = amazon_trainset.data_detailed
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False).data_detailed
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office).data_detailed
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False).data_detailed
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office).data_detailed
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False).data_detailed
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office).data_detailed
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False).data_detailed

    train_data_num_list = [len(amazon_trainset), len(caltech_trainset), len(dslr_trainset), len(webcam_trainset)]
    test_data_num_list = [len(amazon_testset), len(caltech_testset), len(dslr_testset), len(webcam_testset)]
    print("train_data_num_list:", train_data_num_list)
    print("test_data_num_list:", test_data_num_list)

    train_set =  [amazon_trainset, caltech_trainset, dslr_trainset, webcam_trainset]
    test_set = [amazon_testset, caltech_testset, dslr_testset, webcam_testset]
    return train_set, test_set, classnames, lab2cname

class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path, 'office_caltech_10/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)

        self.site_domian = {'amazon':0, 'caltech':1, 'dslr':2, 'webcam':3}
        self.domain = self.site_domian[site]
        self.lab2cname={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.classnames ={'back_pack', 'bike', 'calculator', 'headphones', 'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector'}
        self.target = [self.lab2cname[text] for text in self.label]
        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.target))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.target))
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        self.base_path = base_path
        if train:
            path = os.path.join(self.base_path,'DomainNet/{}_train.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)
        else:
            path = os.path.join(self.base_path,'DomainNet/{}_test.pkl'.format(site))
            self.paths, self.label = np.load(path, allow_pickle=True)

        self.site_domian = {'clipart':0, 'infograph':1, 'painting':2, 'quickdraw':3, 'real':4, 'sketch':5}
        self.domain = self.site_domian[site]
        self.lab2cname = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}
        self.classnames = {'bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass', 'zebra'}
        self.target = [self.lab2cname[text] for text in self.label]
        if train:
            print('Counter({}_train data:)'.format(site), Counter(self.target))
        else:
            print('Counter({}_test data:)'.format(site), Counter(self.target))
        self.label = self.label.tolist()
        self.transform = transform
        self.data_detailed = self._convert()

    def __len__(self):
        return len(self.target)

    def _convert(self):
        data_with_label = []
        for i in range(len(self.target)):
            img_path = os.path.join(self.base_path, self.paths[i])
            data_idx = img_path
            target_idx = self.target[i]
            label_idx = self.label[i]
            item = Datum(impath=data_idx, label=int(target_idx), domain=int(self.domain), classname=label_idx)
            data_with_label.append(item)
        return data_with_label

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.target[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
