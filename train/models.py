#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright (C) 2025 Stanley Arnaud <stantonik@stantonik-mba.local>
#
# Distributed under terms of the MIT license.

from enum import Enum
from tinygrad import Tensor, nn
from typing import Callable

class Conv:
  def __init__(self):
    self.layers: list[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.silu,
      nn.Conv2d(32, 32, 5), Tensor.silu,
      nn.BatchNorm(32), Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3), Tensor.silu,
      nn.Conv2d(64, 64, 3), Tensor.silu,
      nn.BatchNorm(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), nn.Linear(576, 10),
    ]

  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

class Conv2:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))

class MLP():
    def __init__(self, width: int = 512, depth: int = 2):
        self.layers: list[Callable[[Tensor], Tensor]] = [lambda x: x.flatten(1)]
        in_features = 28*28
        for _ in range(depth):
            self.layers.append(nn.Linear(in_features, width))
            self.layers.append(Tensor.silu)
            in_features = width
        self.layers.append(nn.Linear(in_features, 10))

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)

class Type(Enum):
    MLP = 0
    CONV = 1

