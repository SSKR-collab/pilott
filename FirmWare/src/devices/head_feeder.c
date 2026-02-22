#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "comm_types.h"
#include "config_types.h"
#include "ipc_tensor.h"
#include <time.h>

// Forward declarations from ucr_loader.c
typedef struct {
    float* data;
    int* labels;
    int num_samples;
    int sample_length;
    int num_classes;
} dataset_t;

dataset_t* load_ucr_dataset(const char* filename);
void free_dataset(dataset_t* dataset);
tensor_t* get_dataset_sample(const dataset_t* dataset, int sample_idx);
int get_dataset_label(const dataset_t* dataset, int sample_idx);
void normalize_dataset(dataset_t* dataset);

// External configuration
extern model_config_t* g_model_config;

// ============================================================
// Data Augmentation Functions (in-place on float buffer)
// ============================================================

// Generate a random float in [-1, 1] from rand()
static float randf(void) {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// Gaussian-ish noise via Box-Muller approximation (cheap version)
static float gauss_noise(void) {
    // Sum of 6 uniform by CLT approximates N(0,1)
    float sum = 0.0f;
    for (int i = 0; i < 6; i++) sum += (float)rand() / (float)RAND_MAX;
    return sum - 3.0f; // mean 0, std ~0.5
}

// Jitter: add Gaussian noise σ=jitter_std to each time step
static void augment_jitter(float* data, int length, float jitter_std) {
    for (int i = 0; i < length; i++) {
        data[i] += gauss_noise() * jitter_std;
    }
}

// Scaling: multiply all values by a random factor near 1.0
static void augment_scaling(float* data, int length, float scale_range) {
    float scale = 1.0f + randf() * scale_range; // e.g., [1-0.2, 1+0.2]
    for (int i = 0; i < length; i++) {
        data[i] *= scale;
    }
}

// Magnitude warping: smoothly vary magnitude along time axis  
static void augment_magnitude_warp(float* data, int length, float warp_std) {
    // Create 4 knot points with random warping factors
    float knots[4];
    for (int k = 0; k < 4; k++) {
        knots[k] = 1.0f + gauss_noise() * warp_std;
    }
    // Linearly interpolate between knots
    for (int i = 0; i < length; i++) {
        float t = (float)i / (float)(length - 1) * 3.0f; // maps to [0, 3]
        int k0 = (int)t;
        if (k0 >= 3) k0 = 2;
        float frac = t - (float)k0;
        float warp = knots[k0] * (1.0f - frac) + knots[k0 + 1] * frac;
        data[i] *= warp;
    }
}

// Window slicing: randomly crop and resize (simple version: random shift)
static void augment_time_shift(float* data, int length, int max_shift) {
    int shift = (rand() % (2 * max_shift + 1)) - max_shift;
    if (shift == 0) return;
    
    float* tmp = (float*)malloc(sizeof(float) * length);
    for (int i = 0; i < length; i++) {
        int src = i - shift;
        if (src < 0) src = 0;
        if (src >= length) src = length - 1;
        tmp[i] = data[src];
    }
    memcpy(data, tmp, sizeof(float) * length);
    free(tmp);
}

// Apply all augmentations with randomized selection
static void apply_augmentation(float* data, int length) {
    // Always apply jitter (small noise helps a lot)
    augment_jitter(data, length, 0.03f);
    
    // 70% chance: scaling
    if (rand() % 100 < 70)
        augment_scaling(data, length, 0.15f);
    
    // 50% chance: magnitude warping
    if (rand() % 100 < 50)
        augment_magnitude_warp(data, length, 0.1f);
    
    // 50% chance: time shift (±5 steps)
    if (rand() % 100 < 50)
        augment_time_shift(data, length, 5);
}

 int run_head_device(int device_id, const char* dataset_name) {
    log_info("Starting Head Device %d with dataset: %s", device_id, dataset_name);
    int num_layer0_workers = 1;
    int output_channels = 16;  // Default output channels for layer 0
    int output_length = 300;   // Default output length (same as input for layer 0)

    if (g_model_config && g_model_config->num_layers > 0) {
        num_layer0_workers = g_model_config->layers[0].num_devices;
        output_channels = g_model_config->layers[0].in_channels; // Use in_channels for first layer
        output_length = g_model_config->input_length;
        log_info("Config-based setup: %d Layer workers, %d channels", num_layer0_workers, output_channels);
    } else {
        log_info("Using default configuration: 1 worker, 16 channels");
    }


    //-------->>>>
    // Create shared memory and semaphores for Layer 1 (output to workers)
    size_t shm_size = sizeof(ipc_layer_shm_t) + sizeof(float) * output_channels * output_length*2;
    char shm_name[64], fwd_sem_name[64], bwd_sem_name[64];
    snprintf(shm_name, sizeof(shm_name), "/ipc_tensor_L0");
    snprintf(fwd_sem_name, sizeof(fwd_sem_name), "/ipc_sem_L0_fwd");
    snprintf(bwd_sem_name, sizeof(bwd_sem_name), "/ipc_sem_L0_bwd");

    void* shm_ptr = NULL;
    if (ipc_tensor_open(shm_name, shm_size, 1, &shm_ptr) < 0) {
        log_error("Head: Failed to create shared memory for Layer 1");
        return -1;
    }
    sem_t* fwd_sem = NULL;
    if (ipc_sem_open(fwd_sem_name, 1, &fwd_sem) < 0) {
        log_error("Head: Failed to create forward semaphore for Layer 1");
        ipc_tensor_close(shm_ptr, shm_size);
        return -1;
    }
    sem_t* bwd_sem = NULL;
    if (ipc_sem_open(bwd_sem_name, 1, &bwd_sem) < 0) {
        log_error("Head: Failed to create backward semaphore for Layer 1");
        ipc_sem_close(fwd_sem);
        ipc_tensor_close(shm_ptr, shm_size);
        return -1;
    }
    ipc_layer_shm_t* shm_tensor = (ipc_layer_shm_t*)shm_ptr;

    log_info("Shared memory initialized successfully for head device");


   //-----<<<<<<<<<<<

    // Load dataset
    char train_file[512], test_file[512];
    const char *data_root = getenv("UCR_DATA_ROOT");
    if (!data_root) data_root = "/mnt/d/New folder/UCR_DATASETS";
    snprintf(train_file, sizeof(train_file),"%s/%s/%s_TRAIN", data_root, dataset_name, dataset_name);
    snprintf(test_file, sizeof(test_file),"%s/%s/%s_TEST", data_root, dataset_name, dataset_name);

    dataset_t* train_dataset = load_ucr_dataset(train_file);
    dataset_t* test_dataset  = load_ucr_dataset(test_file);
    if (!train_dataset || !test_dataset) {
        log_error("Failed to load datasets");
        // cleanup resources if needed
        free_dataset(train_dataset);
        free_dataset(test_dataset);
        ipc_tensor_close(shm_ptr, shm_size);
        ipc_sem_close(fwd_sem);
        ipc_sem_close(bwd_sem);
        return -1;
    }
    normalize_dataset(train_dataset);
    normalize_dataset(test_dataset);

    log_info("Dataset loaded: %d samples, %d time points, %d classes",
             train_dataset->num_samples, train_dataset->sample_length, train_dataset->num_classes);
    
    // Check memory constraint
    size_t dataset_memory = train_dataset->num_samples * train_dataset->sample_length * sizeof(float) + train_dataset->num_samples * sizeof(int);

    if (dataset_memory > MEMORY_LIMIT_BYTES * 0.8) {  // Use 80% of limit for safety
        log_error("Dataset too large for memory constraint (%.1f KB > %.1f KB)",
                  dataset_memory / 1024.0f, (MEMORY_LIMIT_BYTES * 0.8) / 1024.0f);
        free_dataset(train_dataset);
        free_dataset(test_dataset);
        ipc_tensor_close(shm_ptr, shm_size);
        ipc_sem_close(fwd_sem);
        ipc_sem_close(bwd_sem);
        return -1;
    }
    

    sleep(6);  // Give workers and tail 6 seconds to initialize (tail starts last at ~5s)
    log_info("Head: All devices should be ready, starting sequential training");
    log_info("Head: Data augmentation ENABLED for training samples (jitter+scale+warp+shift)");

    // Allocate augmentation buffer (reused each sample)
    float* aug_buffer = (float*)malloc(sizeof(float) * output_channels * output_length);
    if (!aug_buffer) {
        log_error("Failed to allocate augmentation buffer");
        return -1;
    }

    int train_rounds = train_dataset->num_samples;
    int test_rounds = test_dataset->num_samples;
    int epochs = 10000;
    int epoch_num = 0;
    dataset_t* active_dataset;
    int dataset_idx;
    int do_testing = 1;
    while (epochs > 0) {
        epoch_num++;
        log_info("Starting epoch %d", epoch_num);
        for(int round = 0;round<train_rounds; round++){
            log_info("Round %d: Training sample %d/%d", round + 1, round + 1, train_rounds);
            tensor_t* sample = get_dataset_sample(train_dataset, round);
            int label = get_dataset_label(train_dataset, round);
            if (round > 0) {
                sem_wait(bwd_sem);
            }

            // Copy sample data to augmentation buffer and apply augmentation
            int sample_size = output_channels * output_length;
            memcpy(aug_buffer, sample->data, sizeof(float) * sample_size);
            apply_augmentation(aug_buffer, sample_size);

            float* shm_data = shm_tensor->buffer;
            memcpy(shm_data, aug_buffer, sizeof(float) * sample_size);
        
            shm_tensor->label = label;
            shm_tensor->sample_id = round + 1;
            shm_tensor->counter = 0;
            shm_tensor->bwd_counter = 0;
            shm_tensor->is_testing = 0;
            for(int w =0; w < num_layer0_workers; w++) 
                sem_post(fwd_sem);
        

            log_info("Head: Sample %d (dataset[%d]), Label=%d, Input[0-4]: %.4f %.4f %.4f %.4f %.4f",
                 round + 1, dataset_idx, label,
                 aug_buffer[0], aug_buffer[1], aug_buffer[2],
                 aug_buffer[3], aug_buffer[4]);

            tensor_free(sample);   
        }

        if(do_testing){
            for(int round = 0;round<test_rounds; round++){
                log_info("Round %d: Testing sample %d/%d", round + 1, round + 1, test_rounds);
                tensor_t* sample = get_dataset_sample(test_dataset, round);
                int label = get_dataset_label(test_dataset, round);
                if (round > 0) {
                    // log_info("Head: Waiting for complete pipeline processing of previous sample...");
                    sem_wait(bwd_sem);
                    // log_info("Head: Pipeline completed for previous sample");
                }

            
                float* shm_data = shm_tensor->buffer; // Use buffer for tensor data
                memcpy(shm_data, sample->data, sizeof(float) * output_channels * output_length);
        
                shm_tensor->label = label;
                shm_tensor->sample_id = round + 1;
                shm_tensor->counter = 0; // Reset forward counter for this new sample
                shm_tensor->bwd_counter = 0; // Reset backward counter for this new sample
                shm_tensor->is_testing = 1; // Indicate this is a testing sample
                for(int w =0; w < num_layer0_workers; w++) 
                    sem_post(fwd_sem);
        

                log_info("Head: Sample %d (dataset[%d]), Label=%d, Input[0-4]: %.4f %.4f %.4f %.4f %.4f",
                    round + 1, dataset_idx, label,
                    sample->data[0], sample->data[1], sample->data[2],
                    sample->data[3], sample->data[4]);

                tensor_free(sample);   
            }
        }
        epochs--;
        
    }
    
    ipc_sem_close(fwd_sem);
    ipc_sem_close(bwd_sem);
    ipc_tensor_close(shm_ptr, shm_size);
    shm_unlink(shm_name);
    sem_unlink(fwd_sem_name);
    sem_unlink(bwd_sem_name);
    free_dataset(train_dataset);
    free_dataset(test_dataset);

    log_info("Head device finished");
    return 0;
}
