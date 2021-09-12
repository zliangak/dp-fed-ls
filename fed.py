import copy
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.help import *
from utils.datasets import *
from utils.mi import *
from utils.frequency import *

from estimator.ls import LaplacianSmoothing
from estimator.js import JamesStein
from estimator.th import Thresholding

class DpFederatedLearning(object):
    """
    Perform federated learning
    """
    def __init__(self, model, dict_users, frac, dp, G, z, v, data_root,
                 sampling_type, model_log, model_name, wandb, mi, freq, hist, num_bins=64,
                 sampling_freq=None, criterion=torch.nn.CrossEntropyLoss()):
        super(DpFederatedLearning).__init__()
        self.net_t = model
        self.w_t = copy.deepcopy(self.net_t.state_dict())
        self.dict_users = dict_users
        self.num_users = len(dict_users)
        self.frac = frac
        self.dp = dp
        self.criterion = criterion
        self.G = G
        self.z = z
        self.v = v
        self.model_name = model_name
        self.model_log = model_log
        self.sampling_type = sampling_type
        self.wandb = wandb
        self.mi = mi
        self.freq = freq
        self.sampling_freq = sampling_freq
        self.hist = hist
        self.num_bins = num_bins
        self.data_root = data_root
        self.logs = {'train_acc': [],
                     'val_acc': [],
                     'test_acc': [],
                     'test_loss': [],
                     'train_loss': [],
                     'val_loss': [],
                     'best_test_acc': None,
                     'best_test_loss': None,
                     'best_val': -np.inf,
                     'best_model': [],
                     'sigma': [],
                     'local_loss': [],
                     'avg_norm': [],
                     'all_model': [],
                     'mi_auc': [],
                     'mi_F1': [],
                     'mi_acc': [],
                     'freq': {},
                     'hist': {},
                     }

    def train(self, train_set, val_set, test_set, epochs, lr_inner, lr_outer,
              local_ep, local_bs, optim, wd, interval, gamma, estimator, order, sigma_ls):

        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(train_set, batch_size=local_bs, shuffle=False, num_workers=1)
        val_ldr = DataLoader(val_set, batch_size=local_bs, shuffle=False, num_workers=1)
        test_ldr = DataLoader(test_set, batch_size=local_bs, shuffle=False, num_workers=1)

        if self.mi:
            mi_train_loader, mi_test_loader, mi_label = get_mi_dataloader(self.model_name, self.data_root)

        for epoch in range(epochs):

            if self.sampling_type == 'uniform':
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)
            elif self.sampling_type == 'poisson':
                idxs_binomial = np.random.binomial(1, self.frac, self.num_users)
                self.m = sum(idxs_binomial)
                idxs_users = np.where(idxs_binomial == 1)[0]

            local_ws, local_losses, local_norms = [], [], []

            for idx in tqdm(idxs_users, desc='Epoch:%d, lr_outer:%f, lr_inner:%f'%(epoch, lr_outer, lr_inner)):
                local_train_ldr = DataLoader(DatasetSplit(train_set, self.dict_users[idx]), batch_size=local_bs,
                                             shuffle=True, num_workers=1)
                local_w, local_loss, local_norm = self._local_update(train_ldr=local_train_ldr,
                                                                     lr_inner=lr_inner,
                                                                     local_ep=local_ep,
                                                                     optim=optim,
                                                                     wd=wd
                                                                     )
                local_ws.append(copy.deepcopy(local_w))
                local_losses.append(local_loss)
                local_norms.append(local_norm)

            if self.v:
                sigma = self.v / self.m
            else:
                sigma = self.z * self.G / self.m

            #client_weights = self._client_weights(idxs_users)
            client_weights = np.ones(self.m) /self.m
            self._fed_avg(local_ws, client_weights, lr_outer, sigma=sigma, estimator=estimator, order=order,
                          sigma_ls=sigma_ls, epoch=epoch)
            self.net_t.load_state_dict(self.w_t)

            if (epoch+1) == epochs or (epoch +1) % interval == 0:
                loss_train, acc_train = test(self.net_t, train_ldr, criterion=self.criterion)
                loss_val, acc_val = test(self.net_t, val_ldr, criterion=self.criterion)
                loss_test, acc_test = test(self.net_t, test_ldr, criterion=self.criterion)
                self.logs['train_acc'].append(acc_train)
                self.logs['train_loss'].append(loss_train)
                self.logs['val_acc'].append(acc_val)
                self.logs['val_loss'].append(loss_val)
                self.logs['avg_norm'].append(np.mean(local_norms))
                self.logs['local_loss'].append(np.mean(local_losses))
                self.logs['sigma'].append(sigma)
                self.logs['test_loss'].append(loss_test)
                self.logs['test_acc'].append(acc_test)
                if self.wandb:
                    self.wandb.log({'train_acc': acc_train, "round": epoch})
                    self.wandb.log({'train_loss': loss_train, "round": epoch})
                    self.wandb.log({'val_acc': acc_val, "round": epoch})
                    self.wandb.log({'val_loss': loss_val, "round": epoch})
                    self.wandb.log({'avg_norm': np.mean(local_norms), "round": epoch})
                    self.wandb.log({'local_loss': np.mean(local_losses), "round": epoch})
                    self.wandb.log({'sigma': sigma, "round": epoch})
                    self.wandb.log({'test_loss': loss_test, "round": epoch})
                    self.wandb.log({'test_acc': acc_test, "round": epoch})

                if self.mi:
                    mi_auc, mi_F1, mi_acc = get_roc_auc(self.net_t, mi_train_loader, mi_test_loader, mi_label)
                    self.logs['mi_auc'].append(mi_auc)
                    self.logs['mi_F1'].append(mi_F1)
                    self.logs['mi_acc'].append(mi_acc)
                    if self.wandb:
                        self.wandb.log({'mi_auc': mi_auc, "round": epoch})
                        self.wandb.log({'mi_F1': mi_F1, "round": epoch})
                        self.wandb.log({'mi_acc': mi_acc, "round": epoch})

                if self.model_log:
                    self.logs['all_model'].append(copy.deepcopy(self.net_t.state_dict()))

                if self.logs['best_val'] < acc_val:
                    self.logs['best_val'] = acc_val
                    self.logs['best_model'] = copy.deepcopy(self.net_t.state_dict())
                    if self.wandb:
                        self.wandb.config.update({'best_val' : acc_val}, allow_val_change=True)

                print('Epoch {}/{} --- dp {} -- sigma {:.4f} -- Avg norm {:.4f}'.format(
                    epoch, epochs,
                    str(self.dp),
                    sigma,
                    np.mean(local_norms)
                    )
                )
                print("Train Loss {:.4f} --- Val Loss {:.4f} -- Test Loss {:.4f}".format(loss_train, loss_val, loss_test))
                print("Train acc {:.4f} --- Val acc {:.4f} -- Test acc {:.4f} --Best acc {:.4f}".format(acc_train, acc_val,
                                                                                                        acc_test, self.logs['best_val']
                                                                                                        )
                      )


            lr_inner = lr_inner * gamma


        self.net_t.load_state_dict(self.logs['best_model'])
        loss_test, acc_test = test(self.net_t, test_ldr, criterion=self.criterion)
        self.logs['best_test_acc'] = acc_test
        self.logs['best_test_loss'] = loss_test
        if self.wandb:
            self.wandb.config.best_test_acc = acc_test
            self.wandb.config.best_test_loss = loss_test
        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f}'.format(loss_test, acc_test))

        return self.logs['best_val'], self.logs['best_test_acc']

    def _client_weights(self, idxs_users):
        client_sample_num = [len(self.dict_users[u]) for u in idxs_users]
        client_weights = np.array(client_sample_num) / sum(client_sample_num)
        return client_weights

    def _fed_avg(self, local_ws, client_weights, lr_outer, sigma, estimator, order, sigma_ls, epoch):

        w_avg = copy.deepcopy(local_ws[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * client_weights[0]

            for i in range(1, len(local_ws)):
                w_avg[k] += local_ws[i][k] * client_weights[i]

            if self.freq:
                fourier_transform, N = get_frequency(w_avg[k] - self.w_t[k], N=self.sampling_freq)
                if k in self.logs['freq']:
                    self.logs['freq'][k].append((N,fourier_transform.detach().cpu().tolist()))
                else:
                    self.logs['freq'][k] = [(N, fourier_transform.detach().cpu().tolist())]

            if self.hist:
                np_hist = np.histogram((w_avg[k] - self.w_t[k]).cpu().detach().numpy(), bins=self.num_bins)
                if k in self.logs['hist']:
                    self.logs['hist'][k].append(np_hist)
                else:
                    self.logs['hist'][k] = [np_hist]
                if self.wandb:
                    self.wandb.log({"hist_%s"%k: self.wandb.Histogram(np_histogram=np_hist), "round": epoch})

            if self.dp:
                w_avg[k] = w_avg[k] - self.w_t[k] + float(sigma) * torch.randn(w_avg[k].shape).cuda()
            else:
                w_avg[k] = w_avg[k] - self.w_t[k]

            if w_avg[k].numel() != 1:
                if estimator.lower() == 'ls':
                    w_avg[k] = LaplacianSmoothing(w_avg[k], order, k, sigma_ls)
                elif estimator.lower() == 'js':
                    w_avg[k] = JamesStein(w_avg[k], sigma)
                elif estimator.lower() == 'th':
                    w_avg[k] = Thresholding(w_avg[k], sigma)
            else:
                pass

            self.w_t[k] += float(lr_outer) * w_avg[k]

    def _local_update(self, train_ldr, lr_inner, local_ep, optim, wd):

        net = copy.deepcopy(self.net_t).cuda()
        net.train()

        if optim == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr_inner, weight_decay=wd)
        elif optim == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=lr_inner, weight_decay=wd)
        else:
            raise ValueError('Optim should be in [sgd, adam]')

        epoch_loss = []
        norms = []
        for epoch in range(local_ep):
            total_loss = 0

            for batch_idx, (x, y) in enumerate(train_ldr):
                x, y = x.cuda(), y.cuda()
                net.zero_grad()
                outputs = net(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                optimizer.step()
                if self.dp:
                    net, norm = self._project_net(net)
                else:
                    _, norm = clip_diff_norm2(net.parameters(), self.net_t.parameters(), self.G)
                norms.append(norm)
                total_loss += loss.item()

            epoch_loss.append(total_loss/(batch_idx+1))

        return net.state_dict(), np.mean(epoch_loss), np.mean(norms)

    def _project_net(self, net):
        diff, total_norm = clip_diff_norm2(net.parameters(), self.net_t.parameters(), self.G)
        if total_norm > self.G:
            for i, (p, p_t) in enumerate(zip(net.parameters(), self.net_t.parameters())):
                p.data = p_t.data + diff[i]
        return net, total_norm


def clip_diff_norm2(param, param_t, max_norm):
    total_norm = 0
    diff = []
    for p, p_t in zip(param, param_t):
        p_diff = p.data - p_t.data
        diff.append(p_diff)
        p_norm = p_diff.norm(2)
        total_norm += p_norm.item()**2
    total_norm = total_norm**0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for i in range(len(diff)):
            diff[i] *= clip_coef
    return diff, total_norm









