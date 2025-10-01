import torch
from torch import Tensor


def kld_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Compute the Kullback-Leibler divergence loss.
    :param mu: (Tensor) Mean of the latent Gaussian
    :param logvar: (Tensor) Standard deviation of the latent Gaussian
    :return: (Tensor) KLD loss
    """
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.mean()