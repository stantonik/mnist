#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright (C) 2025 Stanley Arnaud <stantonik@stantonik-mba.local>
#
# Distributed under terms of the MIT license.

from enum import Enum
from typing import Callable
from tinygrad import Tensor, nn

class Conv:
  def __init__(self, activation_fn: Callable[[Tensor],Tensor] = Tensor.silu ):
    self.layers: list[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), activation_fn,
      nn.Conv2d(32, 32, 5), activation_fn,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), activation_fn,
      nn.Conv2d(64, 64, 3), activation_fn,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10),
    ]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

class MLP():
    def __init__(self, width: int = 512, depth: int = 2, activation_fn: Callable[[Tensor],Tensor] = Tensor.silu):
        self.layers: list[Callable[[Tensor], Tensor]] = [lambda x: x.flatten(1)]
        in_features = 28*28
        for _ in range(depth):
            self.layers.append(nn.Linear(in_features, width))
            self.layers.append(activation_fn)
            in_features = width
        self.layers.append(nn.Linear(in_features, 10))

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)

class Type(Enum):
    MLP = 0
    CONV = 1

