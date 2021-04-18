import io
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def resize_axis(tensor, axis, new_size, fill_value=0, random_sampling=False):
    """Truncates or pads a tensor to new_size on on a given axis.
    Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
    size increases, the padding will be performed at the end, using fill_value.
    Args:
      tensor: The tensor to be resized.
      axis: An integer representing the dimension to be sliced.
      new_size: An integer or 0d tensor representing the new value for
        tensor.shape[axis].
      fill_value: Value to use to fill any new entries in the tensor. Will be
        cast to the type of tensor.
    Returns:
      The resized tensor.
    """
    tensor = torch.Tensor(tensor)
    shape = list(tensor.shape)

    pad_shape = shape[:]
    pad_shape[axis] = max(0, new_size - shape[axis])

    start = 0 if shape[axis] <= new_size else np.random.randint(
        shape[axis] - new_size)  # random clip
    old_length = shape[axis]
    shape[axis] = min(shape[axis], new_size)

    resized = torch.cat([
        torch.index_select(tensor, dim=axis, index=torch.randint(old_length, (new_size,))
                           ) if start > 0 and random_sampling else torch.narrow(tensor, dim=axis, start=start, length=shape[axis]),
        torch.Tensor(*pad_shape).fill_(fill_value)
    ], dim=axis)

    return resized


class CircleLoss(torch.nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        alpha = torch.clamp_min(logits + self.m, min=0).detach()
        alpha[labels] = torch.clamp_min(-logits[labels] +
                                        1 + self.m, min=0).detach()
        delta = torch.ones_like(logits, device=logits.device, dtype=logits.dtype) * self.m
        delta[labels] = 1 - self.m
        return self.loss(alpha * (logits - delta) * self.gamma, labels)