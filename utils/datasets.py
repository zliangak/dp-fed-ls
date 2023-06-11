import numpy as np
import json
import os

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.sampling import *


def get_data(model_name, data_root, iid, num_users, train_size=None, train_frac=None):

    if model_name in ['logistic', 'cnn_mnist', 'nn']:

        if not train_size: train_size = 50000
        print('Train set size: {}'.format(train_size))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        raw_set = torchvision.datasets.MNIST(data_root,
                                             train=True,
                                             transform=transform,
                                             download=True
                                             )
        train_indices = np.arange(0, train_size)
        val_indices = np.arange(train_size, train_size+10000)

        train_set = DatasetSplit(raw_set, train_indices)
        val_set = DatasetSplit(raw_set, val_indices)

        test_set = torchvision.datasets.MNIST(data_root,
                                              train=False,
                                              transform=transform,
                                              download=True
                                              )

        print('training set size: {}'.format(len(train_set)))

        if iid:
            dict_users = mnist_iid(train_set, num_users)
        else:
            dict_users = mnist_noniid(train_set, num_users)

    elif model_name in ['cnn_cifar']:

        if not train_size: train_size = 45000

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(24, padding=0),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                              transforms.ToTensor(),
                                              normalize,
                                            ])
        transform_test = transforms.Compose([transforms.CenterCrop(24),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])
        raw_set = torchvision.datasets.CIFAR10(data_root,
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )
        train_indices = np.arange(0, train_size)
        val_indices = np.arange(train_size, train_size+5000)

        train_set = DatasetSplit(raw_set, train_indices)
        val_set = DatasetSplit(raw_set, val_indices)

        test_set = torchvision.datasets.CIFAR10(data_root,
                                                train=False,
                                                download=True,
                                                transform=transform_test
                                                )

        if iid:
            dict_users = cifar_iid(train_set, num_users)
        else:
            dict_users = cifar_noniid(train_set, num_users)

    elif model_name == 'lstm':

        if not train_frac: train_frac = 0.7

        ALL_LETTERS = '''1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -,\'!;"[]?().:<>&{}_'''

        target_train_dict, target_val_dict, shadow_train_dict = split_shakespeare(data_root,
                                                                                  mini=100,
                                                                                  train_frac=train_frac,
                                                                                  val_frac=0.1,
                                                                                  test_frac=0.2)

        train_set = Shakespeare(target_train_dict, ALL_LETTERS)
        val_set = Shakespeare(target_val_dict, ALL_LETTERS)
        test_set = Shakespeare(shadow_train_dict, ALL_LETTERS)

        print('Train set size: {}'.format(len(train_set)))

        if iid:
            raise ValueError('Shakespeare iid not implemented yet!')
        else:
            dict_users, _ = shakespeare_noniid(train_set)

    elif model_name == 'cnn_svhn':

        if not train_size: train_size = 604388
        print('Train set size: {}'.format(train_size))

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        train_set = torchvision.datasets.SVHN(data_root,
                                              split='train',
                                              download=True,
                                              transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.ToTensor(),
                                                normalize,
                                              ]))

        extra_set = torchvision.datasets.SVHN(data_root,
                                              split='extra',
                                              download=True,
                                              transform=transforms.Compose([
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.ToTensor(),
                                                  normalize,
                                              ]))

        train_set = DatasetMerge(train_set, extra_set)
        train_indices = np.arange(0, train_size)
        train_set = DatasetSplit(train_set, train_indices)

        raw_test_set = torchvision.datasets.SVHN(data_root,
                                                 split='test',
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     normalize
                                                 ]))

        val_indices = np.arange(0, 10000)
        test_indices = np.arange(10000, len(raw_test_set))

        val_set = DatasetSplit(raw_test_set, val_indices)
        test_set = DatasetSplit(raw_test_set, test_indices)

        if iid:
            dict_users = svhn_iid(train_set, num_users)
        else:
            dict_users = svhn_noniid(train_set, num_users)


    return train_set, val_set, test_set, dict_users


def split_shakespeare(data_root, train_frac=0.05, val_frac=0.005, test_frac=0.01, mini=64):
    with open(os.path.join(data_root,'all_data.json')) as file:
        all_data = json.load(file)

        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        val_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}

        num_users = 0

        for u in all_data['users']:

            data_x = all_data['user_data'][u]['x']
            data_y = all_data['user_data'][u]['y']
            nums = len(data_y)

            if nums * train_frac < mini:
                continue

            num_users += 1

            train_split = int(nums * train_frac)
            val_split = train_split + int(nums * val_frac)
            test_split = val_split + int(nums * test_frac)

            train_data['users'].append(u)
            train_data['num_samples'].append(train_split)
            train_data['user_data'][u] = {'x': data_x[:train_split], 'y': data_y[:train_split]}

            val_data['users'].append(u)
            val_data['num_samples'].append(val_split - train_split)
            val_data['user_data'][u] = {'x': data_x[train_split:val_split], 'y': data_y[train_split:val_split]}

            test_data['users'].append(u)
            test_data['num_samples'].append(test_split - val_split)
            test_data['user_data'][u] = {'x': data_x[val_split:test_split], 'y': data_y[val_split:test_split]}

    print('Total Number of Users: {}'.format(num_users))

    return train_data, val_data, test_data


class Shakespeare(Dataset):
    def __init__(self, dataset, ALL_LETTERS):

        self.ALL_LETTERS = ALL_LETTERS
        self.NUM_LETTERS = len(ALL_LETTERS)

        self.data = []
        self.labels = []
        self.dict_users = {}
        for u in dataset['users']:
            self.data += dataset['user_data'][u]['x']
            self.labels += dataset['user_data'][u]['y']
            length = len(dataset['user_data'][u]['y'])
            self.dict_users[u] = list(range(len(self.labels) - length, len(self.labels)))

    def letter_to_vec(self, letter):
        '''returns one-hot representation of given letter
        '''
        index = self.ALL_LETTERS.find(letter)
        return index

    def word_to_indices(self, word):
        '''returns a list of character indices
            Args:
                word: string

            Return:
                indices: int list with length len(word)
        '''
        indices = []
        for c in word:
            indices.append(self.ALL_LETTERS.find(c))

        return indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return np.array(self.word_to_indices(x)), np.array(self.letter_to_vec(y))


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class DatasetMerge(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)

    def __len__(self):
        return self.len1 + self.len2

    def __getitem__(self, item):
        if item < self.len1:
            image, label = self.dataset1[item]
        if item >= self.len1:
            image, label = self.dataset2[item-self.len1]
        return image, label