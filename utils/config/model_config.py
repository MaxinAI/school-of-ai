"""
Created on Apr 23, 2020

Configuration scripts for model

@author: Levan Tsinadze
"""

from argparse import Namespace

import torch
from torch import nn
from torch.jit import ScriptModule, trace

# Config Parameters
_DEF_DEVICE = 'cuda'
_CPU_DEVICE = 'cpu'
_MPS_DEVICE = 'mps'
GPU = _DEF_DEVICE
CPU = _CPU_DEVICE
MPS = _MPS_DEVICE


def init_device(conf: Namespace) -> str:
    """
    Initialize device to bind model abd data
    Args:
        conf (Namespace): configuration parameters

    Returns:
        device name
    """
    return GPU if conf.gpu and torch.cuda.is_available() else MPS if (
        conf.mps and torch.backends.mps.is_available()
    ) else CPU


@torch.no_grad()
def script_model(model: nn.Module, sizes: list) -> ScriptModule:
    """
    Generates converts model to the cript model
    Args:
        model (nn.Module): model to convert
        sizes (list): sizes of input

    Returns:
        graph_model (ScriptModule): converted model
    """
    xs = tuple(torch.randn(1, 3, s, s, requires_grad=False) for s in sizes)
    graph_model = trace(model.eval(), xs)
    graph_model.eval()

    return graph_model
