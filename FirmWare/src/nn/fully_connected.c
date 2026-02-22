#include "nn_types.h"
#include <string.h>

void fully_connected_forward(const tensor_t* input, const fc_config_t* config, tensor_t* output) {
    if (!input || !config || !output || !input->data || !config->weights || !output->data) {
        log_error("Invalid arguments for fully connected forward");
        return;
    }
    
    int batch_size = input->batch_size;
    int in_features = input->channels * input->length;  // Flatten input
    int out_features = config->out_features;
    
    if (in_features != config->in_features) {
        log_error("Input feature mismatch: %d vs %d", in_features, config->in_features);
        return;
    }
    
    if (output->batch_size != batch_size || 
        output->channels != out_features || 
        output->length != 1) {
        log_error("Output tensor shape mismatch for fully connected");
        return;
    }
    
    // Perform matrix multiplication: output = input * weights^T + bias
    for (int b = 0; b < batch_size; b++) {
        float* input_batch = input->data + b * in_features;
        float* output_batch = output->data + b * out_features;
        
        for (int out_idx = 0; out_idx < out_features; out_idx++) {
            float sum = 0.0f;
            
            // Dot product with weight row
            for (int in_idx = 0; in_idx < in_features; in_idx++) {
                float weight = config->weights[out_idx * in_features + in_idx];
                sum += input_batch[in_idx] * weight;
            }
            
            // Add bias
            if (config->bias) {
                sum += config->bias[out_idx];
            }
            
            output_batch[out_idx] = sum;
        }
    }
    
    log_debug("FC forward: %dx%d → %dx%d", batch_size, in_features, batch_size, out_features);
}

void fully_connected_backward(const tensor_t* grad_output, const tensor_t* input, const fc_config_t* config, tensor_t* grad_input, float* grad_weights, float* grad_bias) {
    if (!grad_output || !input || !config || !grad_input || 
        !grad_output->data || !input->data || !config->weights || !grad_input->data) {
        log_error("Invalid arguments for fully connected backward");
        return;
    }
    
    int batch_size = input->batch_size;
    int in_features = input->channels * input->length;
    int out_features = config->out_features;
    
    // Initialize gradient arrays
    if (grad_weights) {
        memset(grad_weights, 0, config->weights_size);
    }
    if (grad_bias) {
        memset(grad_bias, 0, config->bias_size);
    }
    
    // Compute gradients for each sample in batch
    for (int b = 0; b < batch_size; b++) {
        float* input_batch = input->data + b * in_features;
        float* grad_output_batch = grad_output->data + b * out_features;
        float* grad_input_batch = grad_input->data + b * in_features;
        
        // Initialize grad_input for this batch to zero
        memset(grad_input_batch, 0, in_features * sizeof(float));
        
        for (int out_idx = 0; out_idx < out_features; out_idx++) {
            float grad_out = grad_output_batch[out_idx];
            
            // Gradient w.r.t. bias
            if (grad_bias) {
                grad_bias[out_idx] += grad_out / batch_size;
            }
            
            for (int in_idx = 0; in_idx < in_features; in_idx++) {
                float weight = config->weights[out_idx * in_features + in_idx];
                float input_val = input_batch[in_idx];
                
                // Gradient w.r.t. input
                grad_input_batch[in_idx] += grad_out * weight;
                
                // Gradient w.r.t. weights
                if (grad_weights) {
                    grad_weights[out_idx * in_features + in_idx] += grad_out * input_val / batch_size;
                }
            }
        }
    }
    
    log_debug("FC backward: gradients computed for %dx%d weights", out_features, in_features);
}

fc_config_t* create_fc_config(int in_features, int out_features) {
    fc_config_t* config = (fc_config_t*)sim_malloc(sizeof(fc_config_t));
    if (!config) {
        return NULL;
    }
    
    config->in_features = in_features;
    config->out_features = out_features;
    config->weights_size = out_features * in_features * sizeof(float);
    config->bias_size = out_features * sizeof(float);
    
    // Allocate weights and bias
    config->weights = (float*)sim_malloc(config->weights_size);
    config->bias = (float*)sim_malloc(config->bias_size);
    
    if (!config->weights || !config->bias) {
        if (config->weights) sim_free(config->weights);
        if (config->bias) sim_free(config->bias);
        sim_free(config);
        return NULL;
    }
    
    // Initialize weights randomly (Xavier initialization)
    float scale = sqrtf(2.0f / (in_features + out_features));
    for (int i = 0; i < out_features * in_features; i++) {
        config->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    // Initialize bias to zero
    memset(config->bias, 0, config->bias_size);
    
    log_info("Created FC layer: %d → %d (weights: %.1fKB)", 
             in_features, out_features, config->weights_size / 1024.0f);
    
    return config;
}

void free_fc_config(fc_config_t* config) {
    if (config) {
        if (config->weights) sim_free_tracked(config->weights, config->weights_size);
        if (config->bias) sim_free_tracked(config->bias, config->bias_size);
        sim_free_tracked(config, sizeof(fc_config_t));
    }
}