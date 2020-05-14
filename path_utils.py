"""
Created on Apr 23, 2020

Utility module for data and model files

@author: Levan Tsinadze
"""

from pathlib import Path


def root_path():
    """
    Project root directory path

    Returns:
        root directory path
    """
    return Path(__file__).parent


def data_path() -> Path:
    """
    Initialize data path

    Returns:
        data path
    """
    return root_path() / 'data'


def models_path() -> Path:
    """
    Inirialize models path

    Returns:
        models path
    """
    return root_path() / 'models'


def onnx_path() -> Path:
    """
    Initialize ONNX files directory path

    Returns:
        onnx_dir: ONNX directory path
    """
    data = data_path()
    onnx_dir = data / 'onnx'

    return onnx_dir
