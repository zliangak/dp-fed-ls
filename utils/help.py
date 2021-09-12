import torch
from torch.utils.data import Dataset

from tqdm import tqdm

from utils.models import *
from dp.google_accountant.google_accountant import *
from dp.autodp import rdp_bank
from dp.autodp import rdp_acct


def get_model(model_name, pretrain=False):
    if model_name == 'logistic':
        model = Linear().cuda()
    elif model_name == 'nn':
        model = NN().cuda()
    elif model_name == 'cnn_mnist':
        model = CNNMnist().cuda()
    elif model_name == 'lstm':
        try:
            model = LSTM().cuda()
        except:
            model = LSTM().cuda()
    elif model_name == 'cnn_svhn':
        if pretrain is True:
            model = get_pretrain_svhn()
        else:
            model = CNNSvhn().cuda()
    else:
        raise NameError('Model {} is not implemented yet!'.format(model_name))
    return model


def get_num_epoch(frac, z, eps, delta, sampling_type):
    '''calculate how many epoch can we use without breaking the privacy budget
    '''
    if sampling_type == 'poisson':
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        Ts = list(range(1, 50000))
        for T in Ts:
            rdp = compute_rdp(q=frac,
                              noise_multiplier=z,
                              steps=T,
                              orders=orders
                              )
            eps_t, delta, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
            if eps_t >= eps:
                break
    elif sampling_type == 'uniform':
        raise ValueError('moment accountant for uniform sampling has not been implemented!')
    return T-1


def get_privacy(frac, z, T, delta, sampling_type):
    '''calculate the epsilon given number of training step T
    '''
    if sampling_type == 'poisson':
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(q=frac,
                          noise_multiplier=z,
                          steps=T,
                          orders=orders
                          )
        eps, delta, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    elif sampling_type == 'uniform':
        func = lambda x: rdp_bank.RDP_gaussian({'sigma': z}, x)
        DPobject = rdp_acct.anaRDPacct()
        for t in range(T):
            DPobject.compose_subsampled_mechanism(func, frac)
        eps = DPobject.get_eps(delta)
    return eps


def test(net, loader, criterion):
    max_batch = 100
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx >= max_batch:
                break

    acc = 1.*correct/total
    loss = test_loss/(batch_idx+1)

    return loss, acc


def get_pretrain_svhn():
    model = CNNSvhn().cuda()
    pre_weight = torch.load('./pretrain/mnist_pretrain_weight.p')
    pre_weight['best_model']['fc1.weight'] = torch.randn(pre_weight['best_model']['fc1.weight'].shape)*0.04
    pre_weight['best_model']['fc2.weight'] = torch.randn(pre_weight['best_model']['fc2.weight'].shape)*0.04
    pre_weight['best_model']['fc3.weight'] = torch.randn(pre_weight['best_model']['fc3.weight'].shape)*0.04
    pre_weight['best_model']['fc1.bias'] = torch.ones(pre_weight['best_model']['fc1.bias'].shape) * 0.1
    pre_weight['best_model']['fc2.bias'] = torch.ones(pre_weight['best_model']['fc2.bias'].shape) * 0.1
    pre_weight['best_model']['fc3.bias'] = torch.zeros(pre_weight['best_model']['fc3.bias'].shape)
    model.load_state_dict(pre_weight['best_model'])
    return model


