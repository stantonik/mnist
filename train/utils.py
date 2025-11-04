#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright (C) 2025 Stanley Arnaud <stantonik@stantonik-mba.local>
#
# Distributed under terms of the MIT license.

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from tinygrad import Tensor, nn
import math

from tinygrad.nn.optim import Optimizer

@dataclass
class HPConfig():
    batch_size: int = 128
    lr: float = 1e-3
    opt: Callable[[list[Tensor]], Optimizer] = nn.optim.Adam
    width: int = 512
    depth: int = 2
    activation_fn: Callable[[Tensor],Tensor] = Tensor.silu
    epochs: int = 1

    lr_decay: float = 0.9
    patience: int = 50
    angle: int = 15
    scale: float = 0.1
    shift: float = 0.1

@dataclass
class TrainLog:
    step: int
    train_loss: float
    test_loss: Optional[float]
    test_acc: Optional[float]
    best_acc: float
    lr: float
    time: float

    batch_size: Optional[int] = None
    width: Optional[int] = None
    depth: Optional[int] = None
    opt_name: Optional[str] = None


class SamplingMod(Enum):
  BILINEAR = 0
  NEAREST = 1

def geometric_transform(X: Tensor, angle_deg: Tensor, scale: Tensor, shift_x: Tensor, shift_y: Tensor, sampling: SamplingMod) -> Tensor:
    B, C, H, W = X.shape

    angle = angle_deg * math.pi / 180.0
    cos_a, sin_a = Tensor.cos(angle), Tensor.sin(angle)
    R11, R12, T13 = cos_a * scale, -sin_a * scale, shift_x
    R21, R22, T23 = sin_a * scale, cos_a * scale, shift_y
    row1 = Tensor.cat(R11.reshape(B, 1), R12.reshape(B, 1), T13.reshape(B, 1), dim=1).reshape(B, 1, 3)
    row2 = Tensor.cat(R21.reshape(B, 1), R22.reshape(B, 1), T23.reshape(B, 1), dim=1).reshape(B, 1, 3)
    row3 = Tensor([[0.0, 0.0, 1.0]]).expand(B, 1, 3)
    affine_matrix = Tensor.cat(row1, row2, row3, dim=1)

    x_idx, y_idx = Tensor.arange(W).float(), Tensor.arange(H).float()
    grid_y, grid_x = y_idx.reshape(-1, 1).expand(H, W), x_idx.reshape(1, -1).expand(H, W)
    coords = Tensor.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords_homo = Tensor.cat(coords, Tensor.ones(H * W, 1), dim=1).reshape(1, H * W, 3).expand(B, H * W, 3)
    transformed_coords = coords_homo.matmul(affine_matrix.permute(0, 2, 1))

    match sampling:
      case SamplingMod.NEAREST:
        x_idx = transformed_coords[:, :, 0].round().clip(0, W - 1).int()
        y_idx = transformed_coords[:, :, 1].round().clip(0, H - 1).int()
        return X.reshape(B, C * H * W).gather(1, y_idx * W + x_idx).reshape(B, C, H, W)

      case SamplingMod.BILINEAR:
        x_prime, y_prime = transformed_coords[:, :, 0],  transformed_coords[:, :, 1]
        x0, y0 = x_prime.floor().int(), y_prime.floor().int()
        dx, dy = x_prime - x0.float(), y_prime - y0.float()

        x1, y1 = x0 + 1, y0 + 1
        x0, y0 = x0.clip(0, W - 1), y0.clip(0, H - 1)
        x1, y1 = x1.clip(0, W - 1), y1.clip(0, H - 1)

        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        X_flat = X.reshape(B, C * H * W)
        v00 = X_flat.gather(1, y0 * W + x0)
        v10 = X_flat.gather(1, y0 * W + x1)
        v01 = X_flat.gather(1, y1 * W + x0)
        v11 = X_flat.gather(1, y1 * W + x1)

        return ((w00 * v00) + (w10 * v10) + (w01 * v01) + (w11 * v11)).reshape(B, C, H, W)

def normalize(X: Tensor) -> Tensor:
  return X * 2 / 255 - 1

