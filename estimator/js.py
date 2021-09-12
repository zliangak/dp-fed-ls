import torch

def JamesStein(grad, sigma):
    d = torch.numel(grad)
    if d >= 3:
        coeff = 1 - (d-2) * sigma**2 / torch.norm(grad, 2)**2
        coeff = coeff.to(grad)
        return coeff * grad
    else:
        return grad