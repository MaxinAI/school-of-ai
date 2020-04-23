"""
Created on Apr 23, 2020

Logger configuration from command line parameters

@author: Levan Tsinadze
"""

import logging


def set_level(verbose: bool):
    """
    Sets logging level
    Args:
        verbose: logging flag
    """
    logging.basicConfig(level='DEBUG' if verbose else 'INFO')
