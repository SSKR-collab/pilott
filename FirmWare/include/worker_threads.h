#ifndef WORKER_THREADS_H
#define WORKER_THREADS_H

#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "comm_types.h"

typedef struct {
    int device_id;
    
    // Shared memory configuration
    int layer_id;       // Which layer this worker processes (0-4)
    int worker_id;      // Worker ID within layer (0-based)
    int num_workers;    // Total workers in this layer
    
    // Neural network configuration
    conv1d_config_t* conv_config;
    
    // Forward pass buffers
    tensor_t* forward_input;
    tensor_t* forward_output;
    tensor_t* forward_temp;  // After conv, before activation
    
    // Backward pass buffers  
    tensor_t* backward_grad_output;
    tensor_t* backward_grad_input;

    // Gradient buffers for Conv1D weights/bias
    float* grad_weights;
    float* grad_bias;
    size_t grad_weights_size;
    size_t grad_bias_size;
    
    // Statistics
    int samples_processed;
    int gradients_processed;
} worker_context_t;

// Function prototypes
worker_context_t* create_worker_context(int device_id, int layer_id, int in_channels, int out_channels,
                                       int kernel_size, int stride, int padding);
void free_worker_context(worker_context_t* ctx);

#endif // WORKER_THREADS_H
