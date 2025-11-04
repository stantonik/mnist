#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright (C) 2025 Stanley Arnaud <stantonik@stantonik-mba.local>
#
# Distributed under terms of the MIT license.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from mpl_toolkits.mplot3d.art3d import math
from tinygrad import Tensor, TinyJit, nn
from tinygrad.device import Device
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import Optimizer
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
from export_model import export_model
import models
from utils import SamplingMod, geometric_transform, normalize
from matplotlib import pyplot as plt
import numpy as np
import time
plt.style.use("dark_background")

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

def train_model(type: models.Type, cfg: HPConfig, cvt_webgpu=False) -> list[TrainLog]:
    if type == models.Type.MLP:
        model = models.MLP(width=cfg.width, depth=cfg.depth)
        model_name = "mnist_mlp"
    elif type == models.Type.CONV:
        model = models.Conv()
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
        state_dict = safe_load(dir_name / f"{model_name}.safetensors")
        load_state_dict(model, state_dict)
        input = Tensor.randn(1, 1, 28, 28)
        prg, *_, state = export_model(model, Device.DEFAULT.lower(), input, model_name=model_name)
        safe_save(state, str(dir_name / f"{model_name}.webgpu.safetensors"))
        with open(dir_name / f"{model_name}.js", "w") as text_file: text_file.write(prg)

    return logs

def mlp_testing(TESTS):
    # To Test
    lrs = [ 3e-4, 1e-3, 3e-3, 1e-2 ]
    depths = [ 2, 3 ]
    widths = [ 512, 1024 ]
    batch_sizes = [ 64, 128, 256 ]
    opts = [ nn.optim.Adam, nn.optim.SGD ]
    activation_fns = [ Tensor.relu, Tensor.silu ]

    config = HPConfig()

    if TESTS == 1:
        for bs in batch_sizes:
            config.batch_size = bs
            logs = train_model(type=TYPE, cfg=config, cvt_webgpu=False)
            plt.plot(list(map(lambda x: x.step, logs)), list(map(lambda x: x.train_loss, logs)), label=f"bs={config.batch_size}")
        plt.xlabel("Training Steps")
        plt.ylabel("Training Loss")
        plt.title("Training Loss vs Steps for Different Batch Sizes")
        plt.grid(True)
        plt.legend(title="Batch Size")
        plt.tight_layout()
        plt.show()

    if TESTS == 2:
        for acti_fn in activation_fns:
            best_acc = []
            for lr in lrs:
                config.lr = lr
                config.activation_fn = acti_fn
                logs = train_model(type=TYPE, cfg=config, cvt_webgpu=False)
                best_acc.append(logs[-1].best_acc)
            plt.plot(lrs, best_acc, label=f"{acti_fn.__name__}", linewidth=0.5)
        plt.xlabel("Learning Step")
        plt.ylabel("Training Accuracy")
        plt.title("Training Accuracy vs LR (SiLU vs ReLU)")
        plt.legend(title="Activation Fn")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if TESTS == 3:
        # LR and Optimizer
        for opt in opts:
            best_acc = []
            for lr in lrs:
                config.lr = lr
                config.opt = opt
                logs = train_model(type=TYPE, cfg=config, cvt_webgpu=False)
                best_acc.append(logs[-1].best_acc)
            plt.plot(lrs, best_acc, label=f"{opt.__name__}", linewidth=0.5)
        plt.xlabel("Learning Step")
        plt.ylabel("Training Accuracy")
        plt.title("Training Accuracy vs LR (Adam vs SGD)")
        plt.legend(title="Optimizer")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    config.activation_fn = Tensor.silu
    config.batch_size = 128
    config.width = 512

    # Still to test
    lrs = [ 1e-3, 3e-3, 1e-2 ]
    depths = [ 2, 3 ]
    opts = [ nn.optim.Adam, nn.optim.SGD ]

    if TESTS == 4:
        for depth in depths:
            for opt in opts:
                best_acc_list = []
                for lr in lrs:
                    # Update config
                    config.depth = depth
                    config.lr = lr
                    config.opt = opt

                    # Train and get logs
                    logs = train_model(type=TYPE, cfg=config, cvt_webgpu=False)
                    best_acc_list.append(logs[-1].best_acc)

                # Plot curve for this optimizer + depth
                plt.plot(
                    lrs,
                    best_acc_list,
                    marker='o',
                    linewidth=1,
                    label=f"{opt.__name__}, depth={depth}"
                )

        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Best Accuracy (%)")
        plt.title("Accuracy vs Learning Rate (Optimizer & Depth)")
        plt.legend(title="Optimizer + Depth")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.tight_layout()
        plt.show()

    final_configs = [
        HPConfig(epochs=10, opt=nn.optim.Adam, depth=2, lr= 1e-2),
        HPConfig(epochs=10, opt=nn.optim.Adam, depth=2, lr= 3e-3),
        HPConfig(epochs=10, opt=nn.optim.Adam, depth=2, lr= 1e-3),
        HPConfig(epochs=10, opt=nn.optim.Adam, depth=3, lr= 1e-2),
        HPConfig(epochs=10, opt=nn.optim.Adam, depth=3, lr= 3e-3),
        HPConfig(epochs=10, opt=nn.optim.Adam, depth=3, lr= 1e-3),
        HPConfig(epochs=10, opt=nn.optim.SGD, depth=2, lr= 1e-2),
        HPConfig(epochs=10, opt=nn.optim.SGD, depth=2, lr= 3e-3),
    ]


    if TESTS == 5:
        config_labels = []
        best_accs = []

        # Run all final configs
        for cfg in final_configs:
            logs = train_model(type=TYPE, cfg=cfg, cvt_webgpu=False)

            # Best test accuracy
            best_test_acc = max(log.test_acc for log in logs if log.test_acc is not None)
            best_accs.append(best_test_acc)

            # Label for x-axis
            label = f"{cfg.opt.__name__}, d={cfg.depth}, lr={cfg.lr:.0e}"
            config_labels.append(label)

        # Sort by best_acc descending
        sorted_idx = np.argsort(best_accs)[::-1]
        best_accs = [best_accs[i] for i in sorted_idx]
        config_labels = [config_labels[i] for i in sorted_idx]

        x = np.arange(len(final_configs))
        width = 0.35  # width of the bars

        fig, ax = plt.subplots(figsize=(12,5))
        bars = ax.bar(x + width/2, best_accs, width, label='Best Test Accuracy', color='salmon')

        # Annotate max bar (first bar after sorting)
        max_bar = bars[0]
        ax.annotate(
            f"{best_accs[0]:.2f}%",
            xy=(max_bar.get_x() + max_bar.get_width()/2, best_accs[0]),
            xytext=(0,5),  # offset above the bar
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='darkred'
        )

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Final Training vs Best Test Accuracy (Sorted)")
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.show()

def conv_testing(TESTS):
    pass


if __name__ == "__main__":
    TYPE = models.Type(getenv("TYPE", models.Type.MLP.value))
    TESTS = getenv("TESTS", 0)

    if TYPE == models.Type.MLP:
        mlp_testing(TESTS)
    elif TYPE == models.Type.CONV:
        conv_testing(TESTS)

