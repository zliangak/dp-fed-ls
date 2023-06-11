import torch
import numpy as np


def LaplacianSmoothing(grad, order, name, sigma):
    size = torch.numel(grad)
    c = np.zeros(shape=(1, size))
    c[0, 0] = -2.
    c[0, 1] = 1.
    c[0, -1] = 1.
    c = torch.Tensor(c).cuda()
    c_fft = torch.view_as_real(torch.fft.fft(c))
    coeff = 1./(1.-sigma*c_fft[...,0])

    if 'conv' in name and 'weight' in name:
        if order.lower() == 'bcwh':
            tmp = grad.view(-1, size)
        elif order.lower() == 'bchw':
            tmp = grad.permute(0,1,3,2).contiguous().view(-1, size)
        elif order.lower() == 'bwhc':
            tmp = grad.permute(0,2,3,1).contiguous().view(-1, size)
        elif order.lower() == 'bhwc':
            tmp = grad.permute(0,3,2,1).contiguous().view(-1, size)
    else:
        tmp = grad.view(-1, size)

    ft_tmp = torch.fft.fft(tmp)
    ft_tmp = torch.view_as_real(ft_tmp)
    tmp = torch.zeros_like(ft_tmp)
    tmp[..., 0] = ft_tmp[..., 0] * coeff
    tmp[..., 1] = ft_tmp[..., 1] * coeff
    tmp = torch.view_as_complex(tmp)
    tmp = torch.fft.ifft(tmp)

    if 'conv' in name and 'weight' in name:
        if order.lower() == 'bcwh':
            tmp = tmp.view(grad.size())
        elif order.lower() == 'bchw':
            tmp = tmp.view(grad.permute(0,1,3,2).size()).permute(0,1,3,2).contiguous()
        elif order.lower() == 'bwhc':
            tmp = tmp.view(grad.permute(0,2,3,1).size()).permute(0,3,1,2).contiguous()
        elif order.lower() == 'bhwc':
            tmp = tmp.view(grad.permute(0,3,2,1).size()).permute(0,3,2,1).contiguous()
    else:
        tmp = tmp.view(grad.size())

    return tmp.real

import torch
import numpy as np


def LaplacianSmoothing(grad, order, name, sigma):
    size = torch.numel(grad)
    c = np.zeros(shape=(1, size))
    c[0, 0] = -2.
    c[0, 1] = 1.
    c[0, -1] = 1.
    c = torch.Tensor(c).cuda()
    c_fft = torch.view_as_real(torch.fft.fft(c))
    coeff = 1./(1.-sigma*c_fft[...,0])

    if 'conv' in name and 'weight' in name:
        if order.lower() == 'bcwh':
            tmp = grad.view(-1, size)
        elif order.lower() == 'bchw':
            tmp = grad.permute(0,1,3,2).contiguous().view(-1, size)
        elif order.lower() == 'bwhc':
            tmp = grad.permute(0,2,3,1).contiguous().view(-1, size)
        elif order.lower() == 'bhwc':
            tmp = grad.permute(0,3,2,1).contiguous().view(-1, size)
    else:
        tmp = grad.view(-1, size)

    ft_tmp = torch.fft.fft(tmp)
    ft_tmp = torch.view_as_real(ft_tmp)
    tmp = torch.zeros_like(ft_tmp)
    tmp[..., 0] = ft_tmp[..., 0] * coeff
    tmp[..., 1] = ft_tmp[..., 1] * coeff
    tmp = torch.view_as_complex(tmp)
    tmp = torch.fft.ifft(tmp)

    if 'conv' in name and 'weight' in name:
        if order.lower() == 'bcwh':
            tmp = tmp.view(grad.size())
        elif order.lower() == 'bchw':
            tmp = tmp.view(grad.permute(0,1,3,2).size()).permute(0,1,3,2).contiguous()
        elif order.lower() == 'bwhc':
            tmp = tmp.view(grad.permute(0,2,3,1).size()).permute(0,3,1,2).contiguous()
        elif order.lower() == 'bhwc':
            tmp = tmp.view(grad.permute(0,3,2,1).size()).permute(0,3,2,1).contiguous()
    else:
        tmp = tmp.view(grad.size())

    return tmp.real