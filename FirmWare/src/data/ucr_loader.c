#include "nn_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    float* data;         // All samples flattened
    int* labels;         // Sample labels
    int num_samples;
    int sample_length;
    int num_classes;
} dataset_t;

// Parse a single line from UCR dataset file
static int parse_ucr_line(const char* line, float* sample, int* label, int sample_length) {
    if (!line || !sample || !label) return -1;
    
    char line_copy[8192];
    strncpy(line_copy, line, sizeof(line_copy) - 1);
    line_copy[sizeof(line_copy) - 1] = '\0';
    
    // Parse comma-separated values
    char* token = strtok(line_copy, ",");
    if (!token) {
        log_error("Failed to parse label from line: %.50s", line);
        return -1;
    }
    
    // First value is the class label
    *label = (int)atoi(token);
    
    // Parse the rest as sample values
    for (int i = 0; i < sample_length; i++) {
        token = strtok(NULL, ",");
        if (!token) {
            log_error("Failed to parse sample value %d", i);
            return -1;
        }
        sample[i] = atof(token);
    }
    
    return 0;
}


#define MAX_LINE_LEN 65536   // Safe for long UCR rows

dataset_t* load_ucr_dataset(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        log_error("Failed to open dataset file: %s", filename);
        return NULL;
    }

    // ---------- First pass: count samples & length ----------
    char line[MAX_LINE_LEN];
    int num_lines = 0;
    int sample_length = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) {
            num_lines++;

            if (sample_length == 0) {
                char line_copy[MAX_LINE_LEN];
                strncpy(line_copy, line, sizeof(line_copy) - 1);
                line_copy[sizeof(line_copy) - 1] = '\0';

                char* token = strtok(line_copy, ","); // label
                token = strtok(NULL, ",");

                while (token) {
                    sample_length++;
                    token = strtok(NULL, ",");
                }
            }
        }
    }

    if (num_lines == 0 || sample_length == 0) {
        log_error("Invalid dataset file: %s", filename);
        fclose(file);
        return NULL;
    }

    // ---------- Memory safety limits ----------
    // First, load ALL lines into memory, then shuffle, then trim
    int total_lines = num_lines;  // Save original count

    log_info("Dataset file: %d samples, %d time points", num_lines, sample_length);

    // ---------- Allocate dataset (for ALL samples initially using malloc) ----------
    dataset_t* dataset = sim_malloc(sizeof(dataset_t));
    if (!dataset) {
        fclose(file);
        return NULL;
    }

    dataset->num_samples = num_lines;
    dataset->sample_length = sample_length;
    dataset->num_classes = 0;

    // Use regular malloc for initial large buffers (will be trimmed after shuffle)
    size_t data_size  = num_lines * sample_length * sizeof(float);
    size_t label_size = num_lines * sizeof(int);

    dataset->data   = malloc(data_size);
    dataset->labels = malloc(label_size);

    if (!dataset->data || !dataset->labels) {
        log_error("Dataset memory allocation failed");
        fclose(file);
        return NULL;
    }

    // ---------- Second pass: load data ----------
    rewind(file);
    int sample_idx = 0;
    int max_label = -1;

    while (fgets(line, sizeof(line), file) && sample_idx < num_lines) {
        if (strlen(line) <= 1) continue;

        float* sample_data = &dataset->data[sample_idx * sample_length];
        int label;

        if (parse_ucr_line(line, sample_data, &label, sample_length) == 0) {
            label -= 1;  // 🔴 UCR labels are 1-based → convert to 0-based

            if (label < 0) {
                log_error("Invalid label in dataset: %d", label);
                continue;
            }

            dataset->labels[sample_idx] = label;
            if (label > max_label) max_label = label;
            sample_idx++;
        }
    }

    fclose(file);

    // ---------- Shuffle dataset (Fisher-Yates) to ensure class balance ----------
    {
        static int seeded = 0;
        if (!seeded) { srand((unsigned)time(NULL)); seeded = 1; }
        float* tmp_sample = malloc(sample_length * sizeof(float));
        for (int i = sample_idx - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            // Swap samples
            memcpy(tmp_sample, &dataset->data[i * sample_length], sample_length * sizeof(float));
            memcpy(&dataset->data[i * sample_length], &dataset->data[j * sample_length], sample_length * sizeof(float));
            memcpy(&dataset->data[j * sample_length], tmp_sample, sample_length * sizeof(float));
            // Swap labels
            int tmp_label = dataset->labels[i];
            dataset->labels[i] = dataset->labels[j];
            dataset->labels[j] = tmp_label;
        }
        free(tmp_sample);
        log_info("Shuffled %d samples", sample_idx);
    }

    // ---------- Apply limits AFTER shuffle ----------
    int max_samples = MEMORY_LIMIT_BYTES / (sample_length * sizeof(float) * 2);
    if (sample_idx > max_samples) {
        log_info("Limiting dataset: %d → %d samples (memory limit)", sample_idx, max_samples);
        sample_idx = max_samples;
    }
    int sample_limit = 500;
    if (sample_idx > sample_limit) {
        log_info("Limiting dataset to %d samples", sample_limit);
        sample_idx = sample_limit;
    }
    dataset->num_samples = sample_idx;

    // Copy trimmed data to sim_malloc buffers and free large malloc buffers
    {
        size_t final_data_size  = sample_idx * sample_length * sizeof(float);
        size_t final_label_size = sample_idx * sizeof(int);
        float* final_data   = sim_malloc(final_data_size);
        int*   final_labels = sim_malloc(final_label_size);
        if (final_data && final_labels) {
            memcpy(final_data,   dataset->data,   final_data_size);
            memcpy(final_labels, dataset->labels, final_label_size);
            free(dataset->data);
            free(dataset->labels);
            dataset->data   = final_data;
            dataset->labels = final_labels;
        }
    }

    // ---------- Count unique classes ----------
    int* seen = calloc(max_label + 1, sizeof(int));
    int unique = 0;

    for (int i = 0; i < sample_idx; i++) {
        int lbl = dataset->labels[i];
        if (!seen[lbl]) {
            seen[lbl] = 1;
            unique++;
        }
    }

    free(seen);
    dataset->num_classes = unique;

    log_info("Loaded dataset: %d samples, %d classes, length=%d",
             dataset->num_samples,
             dataset->num_classes,
             dataset->sample_length);

    return dataset;
}


void free_dataset(dataset_t* dataset) {
    if (dataset) {
        if (dataset->data) {
            size_t data_size = dataset->num_samples * dataset->sample_length * sizeof(float);
            sim_free_tracked(dataset->data, data_size);
        }
        if (dataset->labels) {
            size_t label_size = dataset->num_samples * sizeof(int);
            sim_free_tracked(dataset->labels, label_size);
        }
        sim_free_tracked(dataset, sizeof(dataset_t));
    }
}

// Get a single sample from the dataset
tensor_t* get_dataset_sample(const dataset_t* dataset, int sample_idx) {
    if (!dataset || sample_idx < 0 || sample_idx >= dataset->num_samples) {
        return NULL;
    }
    
    // Create tensor for single sample: batch=1, channels=1, length=sample_length
    tensor_t* tensor = tensor_create(1, 1, dataset->sample_length);
    if (!tensor) {
        return NULL;
    }
    
    // Copy sample data
    const float* sample_data = &dataset->data[sample_idx * dataset->sample_length];
    memcpy(tensor->data, sample_data, dataset->sample_length * sizeof(float));
    
    return tensor;
}

int get_dataset_label(const dataset_t* dataset, int sample_idx) {
    if (!dataset || sample_idx < 0 || sample_idx >= dataset->num_samples) {
        return -1;
    }
    
    return dataset->labels[sample_idx];
}

void normalize_dataset(dataset_t* dataset) {
    if (!dataset || !dataset->data) return;

    for (int s = 0; s < dataset->num_samples; s++) {
        float* sample = dataset->data + s * dataset->sample_length;

        // Mean
        double sum = 0.0;
        for (int i = 0; i < dataset->sample_length; i++) {
            sum += sample[i];
        }
        float mean = (float)(sum / dataset->sample_length);

        // Std
        double var_sum = 0.0;
        for (int i = 0; i < dataset->sample_length; i++) {
            double d = sample[i] - mean;
            var_sum += d * d;
        }
        float std = (float)sqrt(var_sum / dataset->sample_length);

        // Normalize
        if (std > 1e-8f) {
            for (int i = 0; i < dataset->sample_length; i++) {
                sample[i] = (sample[i] - mean) / std;
            }
        }
    }
}

