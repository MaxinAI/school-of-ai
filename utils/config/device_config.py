"""
Created on Feb 19, 2026

Configuration scripts for device

@author: Levan Tsinadze
"""

import torch


def init_device() -> torch.device:
    """
    Initializes device

    Returns:
        torch.device: initialized device
    """
    return torch.device(
        'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    )
