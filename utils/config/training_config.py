"""
Created on Apr 23, 2020

Configuration scripts for training

@author: Levan Tsinadze
"""

import torch


def set_main_device(on_gpu: bool, device_id: int):
    """
    Sets main device
    Args:
        on_gpu: flag to train on GPU device
        device_id: device identifier
    """
    if on_gpu:
        torch.cuda.set_device(device_id)
