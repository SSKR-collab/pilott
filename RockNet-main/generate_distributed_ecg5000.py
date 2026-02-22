#!/usr/bin/env python3
"""
Generate Distributed RockNet configuration for ECG5000.
Produces rocket_config.h and rocket_config.c in distributed_sim/ directory.

This adapts GenerateCodeDistributedRocket.py for:
  - ECG5000 dataset (500 train, 4500 test, 140 time points, 5 classes)
  - 7 nodes (matching FirmWare CNN)
  - 8-bit quantized input (int8_t)
  - QADAM optimizer
"""

import numpy as np
import math
import copy
import pandas as pd
from pathlib import Path
from sympy.utilities.iterables import multiset_permutations


def generate_kernels():
    k = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
    kernel_bin = []
    for e in multiset_permutations(k):
        kernel_bin.append(0)
        for i in range(9):
            kernel_bin[-1] += 2 ** i * e[i]
    return np.array(kernel_bin)


def generate_dilations(len_timeseries):
    max_val = int(min(math.log2((len_timeseries - 1) / 8), 32))
    return np.array([2 ** i for i in range(max_val + 1)])


def quantiles(n):
    return np.array([((_ * ((np.sqrt(5) + 1) / 2)) % 1) for _ in range(1, n + 1)], dtype=np.float32)


def split_kernels(num_nodes, num_kernels):
    split = [num_kernels // num_nodes for _ in range(num_nodes)]
    i = 0
    while i < num_kernels % num_nodes:
        split[i] += 1
        i += 1
    kernel_idx = [0]
    for l in split:
        kernel_idx.append(kernel_idx[-1] + l)
    return split, kernel_idx


def generate_matrix_code(matrix, use_float):
    data = "{"
    if len(matrix.shape) == 1:
        for i in range(len(matrix)):
            data += f"{float(matrix[i]) if use_float else int(matrix[i])}, "
    else:
        for i in range(len(matrix)):
            data += f"{generate_matrix_code(matrix[i], use_float)}, "
    return data[0:-1] + "}"


def load_ucr_dataset(name, test=False):
    data = copy.deepcopy(
        pd.read_csv(f"{Path.home()}/datasets/{name}/{name}_{'TRAIN' if not test else 'TEST'}.tsv",
                     sep="\t", header=None))
    data = data.interpolate(axis=1)
    X = np.array(data[data.columns[1:]])
    y = np.array(data[data.columns[0]])

    shuffle_vec = np.array([i for i in range(len(y))])
    np.random.shuffle(shuffle_vec)

    X = X[shuffle_vec, :]
    X -= np.mean(X)
    X /= np.std(X)
    y = y[shuffle_vec]

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def quantize_8_bit(data, offset, scaling):
    return np.clip((data - offset) / scaling * 127, a_min=-127, a_max=127)


def main():
    num_nodes = 7
    quantize = True
    np.random.seed(1)

    print(f"Loading ECG5000 dataset...")
    X_train, y_train = load_ucr_dataset("ECG5000", test=False)
    X_test, y_test = load_ucr_dataset("ECG5000", test=True)

    num_classes = len(np.unique(np.concatenate([y_train, y_test])))

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Time series length: {len(X_train[0])}")
    print(f"  Classes: {num_classes}")
    print(f"  Nodes: {num_nodes}")

    len_timeseries = len(X_train[0])
    dilations = generate_dilations(len_timeseries)
    kernels = generate_kernels()

    num_biases_per_kernel = int(10_000 / (len(dilations) * len(kernels)))
    num_features = len(dilations) * len(kernels) * num_biases_per_kernel

    print(f"  Kernels: {len(kernels)}")
    print(f"  Dilations: {len(dilations)} -> {list(dilations)}")
    print(f"  Biases per kernel: {num_biases_per_kernel}")
    print(f"  Total features: {num_features}")

    num_kernels_per_device, kernel_idx = split_kernels(num_nodes, len(kernels))
    devices_num_features = np.array([len(dilations) * num_biases_per_kernel * n for n in num_kernels_per_device])

    print(f"  Kernels per device: {num_kernels_per_device}")
    print(f"  Kernel indices: {kernel_idx}")
    print(f"  Features per device: {list(devices_num_features)}")
    print(f"  Max features per device: {max(devices_num_features)}")

    # Quantize
    data_training = [X_train, y_train]
    data_evaluation = [X_test, y_test]

    if quantize:
        quantization_offset = np.mean(data_training[0])
        quantization_scaling = np.percentile(np.abs(data_training[0] - quantization_offset), q=99.9)
        data_training[0] = quantize_8_bit(data_training[0], quantization_offset, quantization_scaling)
        data_evaluation[0] = quantize_8_bit(data_evaluation[0], quantization_offset, quantization_scaling)

    batch_size = 128

    # Generate rocket_config.h
    h_content = f"""#ifndef ROCKET_CONFIG_H
#define ROCKET_CONFIG_H

#include <stdint.h>

#define MINIMUM(a,b) \\
   ({{ __typeof__ (a) _a = (a); \\
       __typeof__ (b) _b = (b); \\
     _a < _b ? _a : _b; }})

#define LENGTH_TIME_SERIES ({len_timeseries})
#define NUM_KERNELS ({len(kernels)})
#define NUM_DILATIONS ({len(dilations)})
#define NUM_BIASES_PER_KERNEL ({num_biases_per_kernel})
#define NUM_FEATURES (NUM_KERNELS * NUM_DILATIONS * NUM_BIASES_PER_KERNEL)

#define NUM_TRAINING_TIMESERIES ({len(data_training[0])})
#define NUM_EVALUATION_TIMESERIES ({len(data_evaluation[0])})

#define MAX_FEATURES_PER_DEVICE ({max(devices_num_features)})

typedef {"int8_t" if quantize else "float"} time_series_type_t;

static uint16_t devices_kernels_idx[] = {generate_matrix_code(np.array(kernel_idx), use_float=False)};

static uint16_t devices_num_features[] = {generate_matrix_code(devices_num_features, use_float=False)};

#define DEVICE_NUM_FEATURES (devices_num_features[TOS_NODE_ID-1])

#define NUM_CLASSES ({num_classes})

#define BATCH_SIZE MINIMUM({batch_size}, NUM_TRAINING_TIMESERIES)

void init_rocket();

const time_series_type_t const **get_training_timeseries();

const uint8_t *get_training_labels();

const time_series_type_t const **get_evaluation_timeseries();

const uint8_t *get_evaluation_labels();

const uint16_t *get_kernels();

const uint32_t *get_dilations();

const float *get_quantiles();

float *get_biases();

#endif
"""

    # Generate rocket_config.c
    c_lines = []
    c_lines.append('#include "rocket_config.h"')
    c_lines.append('#include "conv.h"')
    c_lines.append('#include <stdint.h>')
    c_lines.append('')
    c_lines.append(f'extern uint16_t TOS_NODE_ID;')
    c_lines.append('')
    c_lines.append(f'static time_series_type_t const *training_timeseries[NUM_TRAINING_TIMESERIES];')
    c_lines.append(f'static time_series_type_t const *evaluation_timeseries[NUM_EVALUATION_TIMESERIES];')
    c_lines.append('')
    c_lines.append('//-----------------Training Data------------------')
    c_lines.append('')

    for i, ts in enumerate(data_training[0]):
        c_lines.append(f'static const time_series_type_t training_timeseries_data{i+1}[] = {generate_matrix_code(ts, use_float=not quantize)};')

    c_lines.append('')
    c_lines.append('//-----------------Evaluation Data------------------')
    c_lines.append('')

    for i, ts in enumerate(data_evaluation[0]):
        c_lines.append(f'static const time_series_type_t evaluation_timeseries_data{i+1}[] = {generate_matrix_code(ts, use_float=not quantize)};')

    c_lines.append('')
    c_lines.append(f'static const uint8_t training_labels[] = {generate_matrix_code(data_training[1] - 1, use_float=False)};')
    c_lines.append(f'static const uint8_t evaluation_labels[] = {generate_matrix_code(data_evaluation[1] - 1, use_float=False)};')
    c_lines.append('')
    c_lines.append(f'static const uint16_t kernels[] = {generate_matrix_code(kernels, use_float=False)};')
    c_lines.append(f'static const uint32_t dilations[] = {generate_matrix_code(dilations, use_float=False)};')

    q = quantiles(len(dilations) * len(kernels) * num_biases_per_kernel)
    c_lines.append(f'static const float quantiles_arr[] = {generate_matrix_code(q, use_float=True)};')
    c_lines.append('')
    c_lines.append(f'static float biases[MAX_FEATURES_PER_DEVICE];')
    c_lines.append('')

    # init_rocket
    c_lines.append('void init_rocket()')
    c_lines.append('{')
    for i in range(len(data_training[0])):
        c_lines.append(f'    training_timeseries[{i}] = training_timeseries_data{i+1};')
    c_lines.append('')
    for i in range(len(data_evaluation[0])):
        c_lines.append(f'    evaluation_timeseries[{i}] = evaluation_timeseries_data{i+1};')
    c_lines.append('')
    c_lines.append('    calc_bias(training_timeseries[0], biases, (uint16_t*)kernels, NUM_KERNELS,')
    c_lines.append('             (uint32_t*)dilations, NUM_DILATIONS, (float*)quantiles_arr, NUM_BIASES_PER_KERNEL);')
    c_lines.append('}')
    c_lines.append('')

    # Accessor functions
    c_lines.append('const time_series_type_t const **get_training_timeseries() { return training_timeseries; }')
    c_lines.append('const uint8_t *get_training_labels() { return training_labels; }')
    c_lines.append('const time_series_type_t const **get_evaluation_timeseries() { return evaluation_timeseries; }')
    c_lines.append('const uint8_t *get_evaluation_labels() { return evaluation_labels; }')
    c_lines.append('const uint16_t *get_kernels() { return kernels; }')
    c_lines.append('const uint32_t *get_dilations() { return dilations; }')
    c_lines.append('const float *get_quantiles() { return quantiles_arr; }')
    c_lines.append('float *get_biases() { return biases; }')
    c_lines.append('')

    # Write files
    out_dir = Path(__file__).parent / "c_src" / "distributed_sim"
    out_dir.mkdir(parents=True, exist_ok=True)

    h_path = out_dir / "rocket_config.h"
    c_path = out_dir / "rocket_config.c"

    with open(h_path, 'w') as f:
        f.write(h_content)
    print(f"\nWrote {h_path}")

    with open(c_path, 'w') as f:
        f.write('\n'.join(c_lines))
    print(f"Wrote {c_path}")

    print(f"\nConfiguration generated for ECG5000 with {num_nodes} nodes.")
    print(f"  Kernel split: {num_kernels_per_device}")
    print(f"  Max features/device: {max(devices_num_features)}")
    print(f"  Total features: {num_features}")


if __name__ == "__main__":
    main()
