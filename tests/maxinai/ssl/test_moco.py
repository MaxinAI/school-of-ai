"""
Created on Aug 08, 2020
Tests for the implementation of MoCo / MoCo-V2 training and inference
@author: Levan Tsinadze
"""

import unittest

import torch

from maxinai.ssl.moco import (shuffle_idxs, shuffle_idxs_orig)


class TestBNShuffle(unittest.TestCase):
    """Test case for BBNShuffle implementation"""

    def setUp(self) -> None:
        self.ch = torch.arange(8, dtype=torch.float32)

    def test_shuffle_idxs(self):
        """Test case for index shuffling"""
        bs = self.ch.size(0)
        fidx, bidx = shuffle_idxs(bs)
        ch1 = self.ch[fidx]
        ch2 = ch1[bidx]
        assert torch.all(self.ch.eq(ch2)), 'Shuffling back does not working properly'
        print(f'{self.ch = }')
        print(f'{ch1 = }')
        print(f'{ch2 = }')

    def test_shuffle_idxs_orig(self):
        """Test case for original index shuffle implementation"""
        bs = self.ch.size(0)
        fidx, bidx = shuffle_idxs_orig(bs)
        ch1 = self.ch[fidx]
        ch2 = ch1[bidx]
        assert torch.all(self.ch.eq(ch2)), 'Shuffling back does not working properly'
        print(f'{self.ch = }')
        print(f'{ch1 = }')
        print(f'{ch2 = }')
