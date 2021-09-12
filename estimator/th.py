import torch

def Thresholding(grad, sigma):
    d = torch.numel(grad)
    d = torch.tensor(d).to(grad)
    lbd = sigma * (2 * torch.log(d)).pow(0.5)
    lbd = lbd.to(grad)
    tmp = torch.sign(grad) * torch.relu(torch.abs(grad) - lbd)
    tmp = tmp.to(grad)
    return tmp