"""
Created on Apr 23, 2020

Logger for training and prediction

@author: Levan Tsinadze
"""

import logging
import time

# Default parameters
CHAR_NUM = 46
LINE_SEP = '=' * CHAR_NUM

logger = logging.getLogger(__name__)


class TimerWrapper(object):
    """Timer wrapper object"""

    def __init__(self, verbose: bool = False, func: str = None):
        super().__init__()
        self.verbose = verbose if verbose else logging.root.level == logging.DEBUG
        self.func = func if func else 'function'
        if self.verbose:
            self.start = time.clock()

    def timeit(self):
        """
        Logs timed data
        Returns:
            time_taken: time for line / lines executions
        """
        if self.verbose:
            time_taken = time.clock() - self.start
            print_texts(self.verbose, 'Time taken for ', self.func, ' is - ', time_taken)
        else:
            time_taken = None

        return time_taken


def _is_verbose() -> bool:
    """
    Validates if flags are configured for logging

    Returns:
        if logging is set
    """
    return logging.root.level == logging.DEBUG


def _print_texts(*texts):
    """
    Prints passed texts
    Args:
        *texts: array of strings to print
    """
    print(''.join(str(text) for text in texts))


def print_texts(verbose, *texts):
    """
    Prints passed object directly
    Args:
        verbose: logging flag
        *texts: array of strings to print
    """
    if verbose:
        _print_texts(*texts)


def start_timer(verbose: bool = False, func: object = None) -> TimerWrapper:
    """
    Starts timer service
    Args:
        verbose: logging flag
        func: function name

    Returns:
        initialized timer instance
    """
    return TimerWrapper(verbose, func=func)
