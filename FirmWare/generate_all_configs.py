#!/usr/bin/env python3
"""
=============================================================================
FirmWare CNN — Memory-Aware Configuration Generator  (v2.1)
=============================================================================

Generates model config JSON files for 2-layer and 3-layer distributed CNN
architectures across N = {7, 9, 11, 13, 15} devices for all UCR datasets.

HARD CONSTRAINT: Each device must use ≤ 256 KB heap memory.
    The generator computes exact per-device memory and validates every config.

Design:
  • channels_per_device starts at 16; auto-reduced per-layer if 256 KB is exceeded.
  • "Expanding pyramid" — deeper layers get ≥ channels.
  • Dual global pooling (GAP + GMP) → FC in = 2 × last_out_channels.
  • Stride pattern: first layer stride=1, deeper stride=2.
  • Kernel=5, padding=2 throughout.

Memory formulas (sizeof(float)=4):
  Worker:  4 * [4*W*Cin*K + 4*W + 6*W*Lout + Cin*Lin] + overhead
  Tail:    4 * [5*NC*2C + 8*NC + 10*C + C*L] + overhead
"""

import json, os, math

MEMORY_LIMIT = 256 * 1024        # 256 KB per device
SIZEOF_FLOAT = 4
STRUCT_OVERHEAD = 256

DATASETS = {
    "Cricket_X": {"num_classes": 12, "input_length": 300,
                  "train_samples": 390, "test_samples": 390},
    "FaceAll":   {"num_classes": 14, "input_length": 131,
                  "train_samples": 560, "test_samples": 1690},
    "ECG5000":   {"num_classes":  5, "input_length": 140,
                  "train_samples": 500, "test_samples": 4500},
    "Coffee":    {"num_classes":  2, "input_length": 286,
                  "train_samples":  28, "test_samples":   28},
}

TWO_LAYER_SPLITS = {
    7:  (2, 3),  9:  (3, 4),  11: (4, 5),  13: (5, 6),  15: (6, 7),
}
THREE_LAYER_SPLITS = {
    7:  (1, 2, 2),  9:  (2, 2, 3),  11: (2, 3, 4),  13: (2, 4, 5),  15: (3, 4, 6),
}

HYPERPARAMS = {
    2: {"learning_rate": 0.01,  "epochs": 500,
        "kernel_sizes": [5, 5],    "strides": [1, 2], "paddings": [2, 2]},
    3: {"learning_rate": 0.005, "epochs": 500,
        "kernel_sizes": [5, 5, 5], "strides": [1, 2, 2], "paddings": [2, 2, 2]},
}

# ──────────────── helpers ────────────────

def out_len(lin, k, s, p):
    return (lin + 2*p - k) // s + 1

def worker_mem(cin, cpd, k, lin, lout):
    W, K = cpd, k
    floats = (4*W*cin*K + 4*W + 6*W*lout + cin*lin)
    return floats * SIZEOF_FLOAT + STRUCT_OVERHEAD

def tail_mem(last_ch, nc, lout):
    C = last_ch
    floats = 5*nc*2*C + 8*nc + 10*C + C*lout
    return floats * SIZEOF_FLOAT + STRUCT_OVERHEAD

def head_mem(ds_name, lin):
    ds = DATASETS[ds_name]
    n = ds["train_samples"] + ds["test_samples"]
    return n * lin * SIZEOF_FLOAT + n * SIZEOF_FLOAT + lin * SIZEOF_FLOAT + 128

def best_cpd(cin, k, lin, lout, limit=MEMORY_LIMIT):
    """Largest channels_per_device fitting in limit, multiple of 4 (GroupNorm)."""
    lo, hi, best = 4, 128, 4
    while lo <= hi:
        mid = (lo + hi) // 2
        if worker_mem(cin, mid, k, lin, lout) <= limit:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return (best // 4) * 4 or 4  # round down to multiple of 4

# ──────────────── config generation ────────────────

def generate_config(ds_name, N, nlayers):
    ds = DATASETS[ds_name]
    hp = HYPERPARAMS[nlayers]
    workers = (TWO_LAYER_SPLITS if nlayers == 2 else THREE_LAYER_SPLITS)[N]
    assert sum(workers) + 2 == N

    layers, mem_report = [], []
    cin, lin = 1, ds["input_length"]

    for i in range(nlayers):
        k, s, p = hp["kernel_sizes"][i], hp["strides"][i], hp["paddings"][i]
        lout = out_len(lin, k, s, p)
        cpd = min(16, best_cpd(cin, k, lin, lout))
        och = workers[i] * cpd
        wm = worker_mem(cin, cpd, k, lin, lout)
        layers.append({
            "id": i, "type": "conv1d",
            "in_channels": cin, "out_channels": och,
            "channels_per_device": cpd, "num_devices": workers[i],
            "kernel_size": k, "stride": s, "padding": p,
            "input_length": lin, "output_length": lout,
        })
        mem_report.append({"device": f"Worker_L{i}", "bytes": wm,
                           "kb": round(wm/1024, 1), "fits": wm <= MEMORY_LIMIT})
        cin, lin = och, lout

    last_ch = layers[-1]["out_channels"]
    fc_in = 2 * last_ch
    layers.append({
        "id": nlayers, "type": "fc", "input_length": lin,
        "pooling": ["avg","max"], "in_features": fc_in,
        "out_features": ds["num_classes"], "num_devices": 1,
    })
    tm = tail_mem(last_ch, ds["num_classes"], lin)
    mem_report.append({"device": "Tail", "bytes": tm,
                       "kb": round(tm/1024, 1), "fits": tm <= MEMORY_LIMIT})

    hm = head_mem(ds_name, ds["input_length"])
    mem_report.append({"device": "Head", "bytes": hm,
                       "kb": round(hm/1024, 1), "fits": hm <= MEMORY_LIMIT})

    max_dev = max(r["bytes"] for r in mem_report if r["device"] != "Head")

    cfg = {
        "model": {
            "name": f"nRF52840_UniformCNN_{ds_name}_{nlayers}L_N{N}",
            "version": "2.1",
        },
        "global": {
            "dataset": ds_name, "epochs": hp["epochs"],
            "num_classes": ds["num_classes"],
            "input_length": ds["input_length"],
            "memory_limit_bytes": MEMORY_LIMIT,
            "learning_rate": hp["learning_rate"],
        },
        "layers": layers,
        "memory_analysis": {
            "per_device": mem_report,
            "max_worker_or_tail_kb": round(max_dev/1024, 1),
            "head_kb": round(hm/1024, 1),
            "within_256kb": max_dev <= MEMORY_LIMIT,
        },
    }
    return cfg, mem_report


def main():
    cfgdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    os.makedirs(cfgdir, exist_ok=True)
    generated, violations = [], []

    print("="*76)
    print(f"  FirmWare CNN — Memory-Aware Config Generator (limit={MEMORY_LIMIT//1024}KB)")
    print("="*76)

    for ds_name in sorted(DATASETS):
        ds = DATASETS[ds_name]
        print(f"\n{'─'*64}")
        print(f"  {ds_name} ({ds['num_classes']} classes, L={ds['input_length']})")
        print(f"{'─'*64}")
        for nl in [2, 3]:
            for N in [7, 9, 11, 13, 15]:
                cfg, mr = generate_config(ds_name, N, nl)
                fn = f"model_config_{ds_name.lower()}_{nl}L_N{N}.json"
                with open(os.path.join(cfgdir, fn), 'w') as f:
                    json.dump(cfg, f, indent=2)
                conv = [l for l in cfg["layers"] if l["type"]=="conv1d"]
                fc = [l for l in cfg["layers"] if l["type"]=="fc"][0]
                ws = " → ".join(
                    f"L{l['id']}:{l['out_channels']}ch({l['num_devices']}w,cpd={l['channels_per_device']})"
                    for l in conv)
                mx = cfg["memory_analysis"]["max_worker_or_tail_kb"]
                ok = "✓" if cfg["memory_analysis"]["within_256kb"] else "✗"
                print(f"  {nl}L N={N:2d} | {ws} → FC:{fc['in_features']}→{fc['out_features']}"
                      f" | max={mx:.0f}KB {ok}")
                if not cfg["memory_analysis"]["within_256kb"]:
                    violations.append(fn)
                generated.append(fn)

    print(f"\n{'='*76}")
    print(f"  Generated {len(generated)} configs in {cfgdir}/")
    if violations:
        print(f"  ⚠ {len(violations)} exceed 256KB:")
        for v in violations: print(f"    - {v}")
    else:
        print("  ✓ ALL configs within 256KB per compute device")
    print(f"{'='*76}\n")
    return generated

if __name__ == "__main__":
    main()
