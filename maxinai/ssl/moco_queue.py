"""
Created on Aug 08, 2020
Implementation of MoCo / MoCo-V2 queue for training
@author: Levan Tsinadze
"""

import torch
from torch import nn
from torch.nn.functional import normalize


class MoCoMemoryLoss(nn.Module):
    """Queue for momentum contrast training"""

    def __init__(self, dim: int = 128, queue_sz: int = 65536, tau: float = 0.07, criterion=nn.CrossEntropyLoss):
        super().__init__()
        self.queue_k = queue_sz
        self.tau = tau
        self.criterion = criterion
        self.register_buffer('queue', torch.randn(dim, queue_sz))
        self.queue = normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        De-queue keys and en-queue the new ones
        Args:
            keys: keys to enqueue
        """

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_sz  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.tau

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        loss = self.criterion(logits, labels)

        return logits, labels
