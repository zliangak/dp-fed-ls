import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils.datasets import DatasetSplit, Shakespeare, split_shakespeare
from torch.utils.data import DataLoader
from utils.models import *

import numpy as np


def get_mi_dataloader(model_name, data_root):

    if model_name in ['logistic', 'cnn_mnist', 'nn']:

        n_sample = 10000
        print('MI test size: {}'.format(2 * n_sample))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        raw_train_set = torchvision.datasets.MNIST(data_root,
                                                   train=True,
                                                   transform=transform,
                                                   download=True
                                                   )
        train_indices = np.arange(0, n_sample)
        train_set = DatasetSplit(raw_train_set, train_indices)

        raw_test_set = torchvision.datasets.MNIST(data_root,
                                                  train=False,
                                                  transform=transform,
                                                  download=True
                                                 )
        test_indices = np.arange(0, n_sample)
        test_set = DatasetSplit(raw_test_set, test_indices)

        train_loader = DataLoader(train_set, batch_size=100, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    if model_name in 'cnn_svhn':

        n_sample = 10000
        print('MI test size: {}'.format(2 * n_sample))

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        train_set = torchvision.datasets.SVHN(data_root,
                                              split='train',
                                              download=True,
                                              transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                normalize,
                                              ]))

        train_indices = np.arange(0, n_sample)
        train_set = DatasetSplit(train_set, train_indices)

        raw_test_set = torchvision.datasets.SVHN(data_root,
                                                 split='test',
                                                 download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     normalize
                                                 ]))

        test_indices = np.arange(10000, 10000+n_sample)
        test_set = DatasetSplit(raw_test_set, test_indices)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1)


    if model_name in 'lstm':

        n_sample = 20000
        print('MI test size: {}'.format(2 * n_sample))

        ALL_LETTERS = '''1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -,\'!;"[]?().:<>&{}_'''

        target_train_dict, _, shadow_train_dict = split_shakespeare(data_root,
                                                                    mini=100,
                                                                    train_frac=0.2,
                                                                    val_frac=0.1,
                                                                    test_frac=0.2)

        train_set = Shakespeare(target_train_dict, ALL_LETTERS)
        test_set = Shakespeare(shadow_train_dict, ALL_LETTERS)

        train_indices = np.random.choice(len(train_set), n_sample, replace=False)
        test_indices = np.random.choice(len(test_set), n_sample, replace=False)

        train_set = DatasetSplit(train_set, train_indices)
        test_set = DatasetSplit(test_set, test_indices)

        train_loader = DataLoader(train_set, batch_size=50, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=50, shuffle=False, num_workers=1)

    label = [1] * n_sample + [0] * n_sample

    return train_loader, test_loader, label


def get_roc_auc(model, train_loader, test_loader, label):

    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduce=False)
    loss = []

    for idx, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        score = model(x)
        ls = loss_fn(score, y)
        loss += ls.tolist()

    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        score = model(x)
        ls = loss_fn(score, y)
        loss += ls.tolist()

    max_t = max(loss)
    min_t = min(loss)

    #print(max_t, min_t)

    t1 = np.linspace(min_t, max_t, 5000)
    #ts = np.hstack([t1, t2])
    ts = t1

    TPR_list = []
    FPR_list = []
    F1_list = []
    acc_list = []

    for t in ts:
        T_pos = 0
        T_neg = 0
        F_pos = 0
        F_neg = 0

        for i in range(len(label)):
            if label[i] == 1:
                if loss[i] < t:
                    T_pos += 1
                else:
                    F_neg += 1
            elif label[i] == 0:
                if loss[i] < t:
                    F_pos += 1
                else:
                    T_neg += 1

        TPR_list.append(T_pos / (T_pos + F_neg))
        FPR_list.append(F_pos / (F_pos + T_neg))

        try:
            precision = T_pos / (T_pos + F_pos)
            recall = T_pos / (T_pos + F_neg)
            F1_list.append(2 * (precision * recall) / (recall + precision))
            acc_list.append((T_pos + T_neg) / (T_pos + T_neg + F_pos + F_neg))
        except ZeroDivisionError:
            continue


    F1 = max(F1_list)
    acc = max(acc_list)
    print('Acc {} -- F1 {}'.format(acc, F1))

    AUC = 0
    for i in range(len(FPR_list) - 1):
        AUC += (FPR_list[i + 1] - FPR_list[i]) * TPR_list[i + 1]
    print('AUC:', AUC)

    return AUC, F1, acc