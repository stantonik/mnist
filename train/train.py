#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright (C) 2025 Stanley Arnaud <stantonik@stantonik-mba.local>
#
# Distributed under terms of the MIT license.

from pathlib import Path
from mpl_toolkits.mplot3d.art3d import math
from tinygrad import Tensor, TinyJit, nn
from tinygrad.device import Device
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from export_model import export_model
import models
from testing import conv_testing, mlp_testing
from utils import HPConfig, SamplingMod, TrainLog, geometric_transform, normalize
from matplotlib import pyplot as plt
import time
plt.style.use("dark_background")

def train_model(type: models.Type, cfg: HPConfig, cvt_webgpu=False) -> list[TrainLog]:
    if type == models.Type.MLP:
        model = models.MLP(width=cfg.width, depth=cfg.depth, activation_fn=cfg.activation_fn)
        model_name = "mnist_mlp"
    elif type == models.Type.CONV:
        model = models.Conv(activation_fn=cfg.activation_fn)
        model_name = "mnist_convnet"

    dir_name = Path("../app/public/models") / model_name
    dir_name.mkdir(exist_ok=True)

    X_train, Y_train, X_test, Y_test = mnist()
    opt = cfg.opt(nn.state.get_parameters(model))
    opt.lr = Tensor(cfg.lr)

    # -----------------
    # Training step
    # -----------------
    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        samples = Tensor.randint(cfg.batch_size, high=int(X_train.shape[0]))
        angle_deg = (Tensor.rand(cfg.batch_size) * 2 * cfg.angle - cfg.angle)
        scale = 1.0 + (Tensor.rand(cfg.batch_size) * 2 * cfg.scale - cfg.scale)
        shift_x = (Tensor.rand(cfg.batch_size) * 2 * cfg.shift - cfg.shift)
        shift_y = (Tensor.rand(cfg.batch_size) * 2 * cfg.shift - cfg.shift)

        opt.zero_grad()
        input = normalize(geometric_transform(X_train[samples], angle_deg, scale, shift_x, shift_y, SamplingMod.NEAREST))
        loss = model(input).sparse_categorical_crossentropy(Y_train[samples]).backward()
        return loss.realize(*opt.schedule_step())

    # -----------------
    # Evaluation step
    # -----------------
    @TinyJit
    def eval_step() -> tuple[Tensor, Tensor]:
        out = model(normalize(X_test))
        test_loss = out.sparse_categorical_crossentropy(Y_test)
        test_acc = (out.argmax(axis=1) == Y_test).mean() * 100
        return test_loss, test_acc

    # -----------------
    # Training loop
    # -----------------
    steps_cnt = math.ceil((len(X_train) / cfg.batch_size) * cfg.epochs)
    logs: list[TrainLog] = []
    best_acc, best_since = 0.0, 0
    start_time = time.time()

    for i in (t := trange(steps_cnt, desc="Training")):
        loss = train_step()
        elapsed = time.time() - start_time

        test_loss, test_acc = None, None
        if i % 10 == 9:  # periodic evaluation
            test_loss, test_acc = eval_step()
            test_loss = test_loss.item()
            test_acc = test_acc.item()

            if test_acc > best_acc:
                best_acc = test_acc
                best_since = 0
                state_dict = get_state_dict(model)
                safe_save(state_dict, str(dir_name / f"{model_name}.safetensors"))
                del state_dict
            else:
                best_since += 1
        else:
            best_since += 1

        # LR decay if plateau
        if best_since % cfg.patience == cfg.patience - 1:
            best_since = 0
            opt.lr *= cfg.lr_decay
            state_dict = safe_load(dir_name / f"{model_name}.safetensors")
            load_state_dict(model, state_dict)
            del state_dict

        # -----------------
        # Logging
        # -----------------
        logs.append(TrainLog(
            step=i,
            train_loss=loss.item(),
            test_loss=test_loss,
            test_acc=test_acc,
            best_acc=best_acc,
            lr=opt.lr.item(),
            time=elapsed,
            batch_size=cfg.batch_size,
            width=cfg.width,
            depth=cfg.depth,
            opt_name=opt.__class__.__name__,
        ))

        t.set_description(f"lr: {opt.lr.item():.2e}  loss: {loss.item():.2f}  best: {best_acc:.2f}%")

    if cvt_webgpu:
        Device.DEFAULT = "WEBGPU"
        model = models.Conv() if TYPE == models.Type.CONV else models.MLP()
        state_dict = safe_load(dir_name / f"{model_name}.safetensors")
        load_state_dict(model, state_dict)
        input = Tensor.randn(1, 1, 28, 28)
        prg, *_, state = export_model(model, Device.DEFAULT.lower(), input, model_name=model_name)
        safe_save(state, str(dir_name / f"{model_name}.webgpu.safetensors"))
        with open(dir_name / f"{model_name}.js", "w") as text_file: text_file.write(prg)

    return logs

if __name__ == "__main__":
    TYPE = models.Type.MLP if getenv("TYPE", "mlp").lower() == "mlp" else models.Type.CONV
    TESTS = getenv("TESTS", 0)

    # Hyperparameters from environment
    B = int(getenv("BATCH", 128))
    LR = float(getenv("LR", 1e-3))
    LR_DECAY = float(getenv("LR_DECAY", 0.9))
    PATIENCE = int(getenv("PATIENCE", 50))

    ANGLE = int(getenv("ANGLE", 15))
    SCALE = float(getenv("SCALE", 0.1))
    SHIFT = float(getenv("SHIFT", 0.1))
    SAMPLING = SamplingMod(getenv("SAMPLING", SamplingMod.NEAREST.value))

    OPT = getenv("OPT", "Adam").lower()
    OPT = nn.optim.Adam if OPT == "adam" else nn.optim.SGD

    ACT_FN = getenv("ACT_FN", "silu").lower()
    ACT_FN = Tensor.silu if ACT_FN == "silu" else Tensor.relu

    DEPTH = int(getenv("DEPTH", 2))
    WIDTH = int(getenv("WIDTH", 512))
    EPOCHS = int(getenv("EPOCHS", 1))

    # Final configuration object
    config = HPConfig()
    config.batch_size = B
    config.lr = LR
    config.opt = OPT
    config.width = WIDTH
    config.depth = DEPTH
    config.activation_fn = ACT_FN
    config.epochs = EPOCHS
    config.lr_decay = LR_DECAY
    config.patience = PATIENCE
    config.angle = ANGLE
    config.scale = SCALE
    config.shift = SHIFT

    print("Loaded training configuration:")
    print(config)

    if TESTS == 0:
        train_model(TYPE, config, cvt_webgpu=True)
    elif TYPE == models.Type.MLP:
        mlp_testing(TYPE, TESTS, train_model)
    elif TYPE == models.Type.CONV:
        conv_testing(TYPE, TESTS, train_model)

