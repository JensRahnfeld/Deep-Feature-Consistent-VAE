import torch


def kl_loss(mu, logvar, beta):
    loss = -0.5 * torch.sum(-torch.exp(logvar/2.0) - mu.pow(2) + 1 + logvar, dim=[1])
    loss = beta * torch.mean(loss)

    return loss