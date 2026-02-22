#!/usr/bin/env python3
"""
=============================================================================
FirmWare CNN — Systematic Configuration Generator
=============================================================================

Generates model config JSON files for 2-layer and 3-layer distributed CNN
architectures across N = {7, 9, 11, 13, 15} devices for all UCR datasets.

Mathematical Design Principles:
────────────────────────────────
  • Device budget:  N = 1 (Head) + Σ workers + 1 (Tail)
  • Each conv worker handles 16 output channels (channels_per_device = 16)
  • "Expanding pyramid" — deeper layers get ≥ as many channels as earlier ones
  • Dual global pooling (GAP + GMP) → FC input = 2 × last_layer_out_channels
  • Stride pattern: first layer stride=1 (preserve length), deeper layers stride=2
  • Kernel = 5, padding = 2 throughout (maintains or halves temporal length)

2-Layer Worker Split (N − 2 = W0 + W1):
────────────────────────────
  N=7:  W0=2, W1=3   →  L0: 32ch,  L1: 48ch   → FC: 96
  N=9:  W0=3, W1=4   →  L0: 48ch,  L1: 64ch   → FC: 128
  N=11: W0=4, W1=5   →  L0: 64ch,  L1: 80ch   → FC: 160
  N=13: W0=5, W1=6   →  L0: 80ch,  L1: 96ch   → FC: 192
  N=15: W0=6, W1=7   →  L0: 96ch,  L1: 112ch  → FC: 224

3-Layer Worker Split (N − 2 = W0 + W1 + W2):
────────────────────────────
  N=7:  W0=1, W1=2, W2=2   →  L0: 16ch, L1: 32ch, L2: 32ch  → FC: 64
  N=9:  W0=2, W1=2, W2=3   →  L0: 32ch, L1: 32ch, L2: 48ch  → FC: 96
  N=11: W0=2, W1=3, W2=4   →  L0: 32ch, L1: 48ch, L2: 64ch  → FC: 128
  N=13: W0=2, W1=4, W2=5   →  L0: 32ch, L1: 64ch, L2: 80ch  → FC: 160
  N=15: W0=3, W1=4, W2=6   →  L0: 48ch, L1: 64ch, L2: 96ch  → FC: 192

Hyperparameters:
────────────────
  • 2-layer: lr = 0.01,  epochs = 500  (cosine annealing T_max=60, warmup=3)
  • 3-layer: lr = 0.005, epochs = 500  (deeper → smaller lr for stability)
  • Early stopping patience = 50 (in tail_classifier.c)
  • AdamW: β1=0.9, β2=0.999, weight_decay=0.0003
  • Dropout = 0.2 before FC, Group Norm (8 groups) after each Conv1D

Author: Auto-generated
"""

import json
import os
import math

# ═══════════════════════════════════════════════════════════════════
#  Dataset Definitions
# ═══════════════════════════════════════════════════════════════════
DATASETS = {
    "Cricket_X": {"num_classes": 12, "input_length": 300},
    "FaceAll":   {"num_classes": 14, "input_length": 131},
    "ECG5000":   {"num_classes": 5,  "input_length": 140},
    "Coffee":    {"num_classes": 2,  "input_length": 286},
}

CHANNELS_PER_DEVICE = 16

# ═══════════════════════════════════════════════════════════════════
#  Device Splits — mathematically optimal expanding pyramid
# ═══════════════════════════════════════════════════════════════════

# 2-Layer: (Layer0_workers, Layer1_workers)
TWO_LAYER_SPLITS = {
    7:  (2, 3),    # 32ch → 48ch,   Total workers = 5
    9:  (3, 4),    # 48ch → 64ch,   Total workers = 7
    11: (4, 5),    # 64ch → 80ch,   Total workers = 9
    13: (5, 6),    # 80ch → 96ch,   Total workers = 11
    15: (6, 7),    # 96ch → 112ch,  Total workers = 13
}

# 3-Layer: (Layer0_workers, Layer1_workers, Layer2_workers)
THREE_LAYER_SPLITS = {
    7:  (1, 2, 2),   # 16ch → 32ch → 32ch,  Total workers = 5
    9:  (2, 2, 3),   # 32ch → 32ch → 48ch,  Total workers = 7
    11: (2, 3, 4),   # 32ch → 48ch → 64ch,  Total workers = 9
    13: (2, 4, 5),   # 32ch → 64ch → 80ch,  Total workers = 11
    15: (3, 4, 6),   # 48ch → 64ch → 96ch,  Total workers = 13
}

# ═══════════════════════════════════════════════════════════════════
#  Hyperparameters by architecture depth
# ═══════════════════════════════════════════════════════════════════
HYPERPARAMS = {
    2: {
        "learning_rate": 0.01,
        "epochs": 500,
        "kernel_sizes": [5, 5],
        "strides": [1, 2],
        "paddings": [2, 2],
    },
    3: {
        "learning_rate": 0.005,
        "epochs": 500,
        "kernel_sizes": [5, 5, 5],
        "strides": [1, 2, 2],
        "paddings": [2, 2, 2],
    },
}


def compute_output_length(input_length, kernel_size, stride, padding):
    """Conv1D output length: floor((L_in + 2*pad - K) / stride) + 1"""
    return math.floor((input_length + 2 * padding - kernel_size) / stride) + 1


def generate_config(dataset_name, num_devices, num_layers):
    """Generate a complete model config JSON dict."""
    ds = DATASETS[dataset_name]
    hp = HYPERPARAMS[num_layers]
    
    if num_layers == 2:
        workers = TWO_LAYER_SPLITS[num_devices]
    else:
        workers = THREE_LAYER_SPLITS[num_devices]
    
    # Verify device count
    total_workers = sum(workers)
    assert total_workers + 2 == num_devices, \
        f"Worker split {workers} + Head + Tail = {total_workers + 2} ≠ {num_devices}"
    
    layers = []
    in_ch = 1
    in_len = ds["input_length"]
    
    for i in range(num_layers):
        out_ch = workers[i] * CHANNELS_PER_DEVICE
        out_len = compute_output_length(in_len, hp["kernel_sizes"][i],
                                        hp["strides"][i], hp["paddings"][i])
        layer = {
            "id": i,
            "type": "conv1d",
            "in_channels": in_ch,
            "out_channels": out_ch,
            "channels_per_device": CHANNELS_PER_DEVICE,
            "num_devices": workers[i],
            "kernel_size": hp["kernel_sizes"][i],
            "stride": hp["strides"][i],
            "padding": hp["paddings"][i],
            "input_length": in_len,
            "output_length": out_len,
        }
        layers.append(layer)
        in_ch = out_ch
        in_len = out_len
    
    # FC layer with dual pooling (GAP + GMP → 2 × last_out_channels)
    last_out_ch = layers[-1]["out_channels"]
    fc_in = 2 * last_out_ch
    
    fc_layer = {
        "id": num_layers,
        "type": "fc",
        "input_length": in_len,
        "pooling": ["avg", "max"],
        "in_features": fc_in,
        "out_features": ds["num_classes"],
        "num_devices": 1,
    }
    layers.append(fc_layer)
    
    config = {
        "model": {
            "name": f"nRF52840_UniformCNN_{dataset_name}_{num_layers}L_N{num_devices}",
            "version": "2.0",
        },
        "global": {
            "dataset": dataset_name,
            "epochs": hp["epochs"],
            "num_classes": ds["num_classes"],
            "input_length": ds["input_length"],
            "memory_limit_bytes": 220000,
            "learning_rate": hp["learning_rate"],
        },
        "layers": layers,
    }
    
    return config


def main():
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    os.makedirs(config_dir, exist_ok=True)
    
    generated = []
    
    print("=" * 72)
    print("  FirmWare CNN — Configuration Generator")
    print("=" * 72)
    
    for dataset_name in sorted(DATASETS.keys()):
        ds = DATASETS[dataset_name]
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {dataset_name} ({ds['num_classes']} classes, L={ds['input_length']})")
        print(f"{'─' * 60}")
        
        for num_layers in [2, 3]:
            for N in [7, 9, 11, 13, 15]:
                config = generate_config(dataset_name, N, num_layers)
                
                ds_lower = dataset_name.lower()
                filename = f"model_config_{ds_lower}_{num_layers}L_N{N}.json"
                filepath = os.path.join(config_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                
                conv_layers = [l for l in config["layers"] if l["type"] == "conv1d"]
                fc_layer = [l for l in config["layers"] if l["type"] == "fc"][0]
                
                workers_str = " → ".join(
                    f"L{l['id']}:{l['out_channels']}ch({l['num_devices']}w)"
                    for l in conv_layers
                )
                
                print(f"  {num_layers}L N={N:2d} | {workers_str} → FC:{fc_layer['in_features']}→{fc_layer['out_features']} | lr={config['global']['learning_rate']}")
                
                generated.append(filename)
    
    print(f"\n{'=' * 72}")
    print(f"  Generated {len(generated)} configuration files in {config_dir}/")
    print(f"{'=' * 72}")
    
    # Print summary table
    print("\n  Architecture Summary:")
    print("  ┌────┬────────────────────────────────────┬────────────────────────────────────────────────┐")
    print("  │  N │        2-Layer CNN (devices)        │              3-Layer CNN (devices)              │")
    print("  ├────┼────────────────────────────────────┼────────────────────────────────────────────────┤")
    for N in [7, 9, 11, 13, 15]:
        w2 = TWO_LAYER_SPLITS[N]
        w3 = THREE_LAYER_SPLITS[N]
        ch2 = [w * 16 for w in w2]
        ch3 = [w * 16 for w in w3]
        fc2 = 2 * ch2[-1]
        fc3 = 2 * ch3[-1]
        two_str = f"H+{'+'.join(str(w) for w in w2)}+T  ch:{'/'.join(str(c) for c in ch2)} FC:{fc2}"
        three_str = f"H+{'+'.join(str(w) for w in w3)}+T  ch:{'/'.join(str(c) for c in ch3)} FC:{fc3}"
        print(f"  │ {N:2d} │ {two_str:34s} │ {three_str:46s} │")
    print("  └────┴────────────────────────────────────┴────────────────────────────────────────────────┘")
    
    return generated


if __name__ == "__main__":
    main()
