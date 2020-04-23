"""
Created on Apr 23, 2020

Configuration scripts for model

@author: Levan Tsinadze
"""

from argparse import Namespace

import torch
from torch import nn
from torch.jit import trace, ScriptModule

# Config Parameters
_DEF_DEVICE = 'cuda'
_CPU_DEVICE = 'cpu'
GPU = _DEF_DEVICE
CPU = _CPU_DEVICE


def init_device(conf: Namespace) -> str:
    """
    Initialize device to bind model abd data
    Args:
        conf: configuration parameters

    Returns:
        device name
    """
    return GPU if conf.gpu and torch.cuda.is_available() else CPU


@torch.no_grad()
def script_model(model: nn.Module, sizes: list) -> ScriptModule:
    """
    Generates converts model to the cript model
    Args:
        model: model to convert
        sizes: sizes of input

    Returns:
        graph_model: converted model
    """
    xs = tuple(torch.randn(1, 3, s, s, requires_grad=False) for s in sizes)
    graph_model = trace(model.eval(), xs)
    graph_model.eval()

    return graph_model
