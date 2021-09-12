import torch

def get_frequency(grad, N=None):
    size = torch.numel(grad)
    tmp = grad.view(-1, size)
    if N is None: N = size
    fourierTransform = torch.fft.fft(tmp) / N
    return abs(fourierTransform), N


