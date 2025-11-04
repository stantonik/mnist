#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright (C) 2025 Stanley Arnaud <stantonik@stantonik-mba.local>
#
# Distributed under terms of the MIT license.

from tinygrad import Tensor, nn
from utils import HPConfig
from matplotlib import pyplot as plt
import numpy as np

def mlp_testing(TYPE, TESTS, train_model):
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

def conv_testing(TYPE, TESTS, train_model):
    # To Test
    lrs = [ 3e-4, 1e-3, 3e-3, 1e-2 ]
    batch_sizes = [ 64, 128, 256 ]
    opts = [ nn.optim.Adam, nn.optim.SGD ]
    activation_fns = [ Tensor.relu, Tensor.silu ]

    config = HPConfig()

    if TESTS == 1:
        best_acc = []
        for bs in batch_sizes:
            config.batch_size = bs
            logs = train_model(type=TYPE, cfg=config, cvt_webgpu=False)
            best_acc.append(logs[-1].best_acc)
        plt.plot(batch_sizes, best_acc, linewidth=1)
        plt.xlabel("Batch Size")
        plt.ylabel("Best Accuracy")
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

    final_configs = [
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.Adam, activation_fn=Tensor.silu, lr= 3e-3),
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.Adam, activation_fn=Tensor.silu, lr= 1e-3),
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.SGD, activation_fn=Tensor.silu, lr= 1e-2),
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.SGD, activation_fn=Tensor.silu, lr= 3e-3),
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.Adam, activation_fn=Tensor.relu, lr= 3e-3),
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.Adam, activation_fn=Tensor.relu, lr= 1e-3),
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.SGD, activation_fn=Tensor.relu, lr= 1e-2),
        HPConfig(epochs=5, batch_size=64, opt=nn.optim.SGD, activation_fn=Tensor.relu, lr= 3e-3),
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
            label = f"{cfg.opt.__name__}, actv_fn={cfg.activation_fn.__name__}, lr={cfg.lr:.0e}"
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


