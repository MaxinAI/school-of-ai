"""
Created on Aug 08, 2020
Implementation of MoCo / MoCo-V2 training and inference
@author: Levan Tsinadze
"""

import torch
from torch import nn


def shuffle_idxs(bsz: int) -> tuple:
    """
    Shuffle indices for ShuffleBN in key encoders
    Args:
        bsz: batch size

    Returns:
        shfl_idxs: indices for forward call
        orig_idxs: indices after call
    """
    shfl_idxs = torch.randperm(bsz).long()
    orig_idxs = torch.zeros(bsz).long()
    value = torch.arange(bsz).long()
    orig_idxs.index_copy_(0, shfl_idxs, value)

    return shfl_idxs, orig_idxs


def shuffle_idxs_orig(bsz: int) -> tuple:
    """
    Shuffle indices for SuffleBN in key encoders
    Args:
        bsz: batch size

    Returns:
        shfl_idxs: shuffled indices
        rest_idxs: restored indices
    """
    shfl_idxs = torch.randperm(bsz).long()
    rest_idxs = torch.argsort(shfl_idxs)

    return shfl_idxs, rest_idxs


def set_bn_train(layer: nn.Module):
    """
    Make batch normalization trainable
    Args:
        layer: layer of the model
    """
    cls_name = layer.__class__.__name__
    if 'BatchNorm' in cls_name:
        layer.train()


def moment_update(model_q, model_k, mt: float = 0.999):
    """
    Update weights of the key encoder with exponentially moving average:
    model_ema = m * model_ema + (1 - m) model
    Args:
        model_q: query encoder
        model_k: key encoder
        mt: momentum for parameters update
    """
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.mul_(mt).add_(1 - mt, p_q.detach().data)
