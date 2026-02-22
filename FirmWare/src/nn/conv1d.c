#include "nn_types.h"
#include <string.h>
#include <math.h>

// Helper function to calculate output length
static int calc_output_length(int input_length, int kernel_size, int stride, int padding) {
    return (input_length + 2 * padding - kernel_size) / stride + 1;
}

void conv1d_forward(const tensor_t* input, const conv1d_config_t* config, tensor_t* output) {
    if (!input || !config || !output || !input->data || !config->weights || !output->data) {
        log_error("Invalid arguments for conv1d forward");
        return;
    }
    
    int batch_size = input->batch_size;
    int in_channels = input->channels;
    int in_length = input->length;
    int out_channels = config->out_channels;
    int kernel_size = config->kernel_size;
    int stride = config->stride;
    int padding = config->padding;
    
    int out_length = calc_output_length(in_length, kernel_size, stride, padding);
    
    // Verify output tensor dimensions
    if (output->batch_size != batch_size || 
        output->channels != out_channels || 
        output->length != out_length) {
        log_error("Output tensor shape mismatch for conv1d");
        return;
    }
    
    // Initialize output to zero
    tensor_fill_zeros(output);
    
    // Convolution computation
    for (int b = 0; b < batch_size; b++) {
        for (int out_c = 0; out_c < out_channels; out_c++) {
            for (int out_pos = 0; out_pos < out_length; out_pos++) {
                float sum = 0.0f;
                
                // Convolve across input channels
                for (int in_c = 0; in_c < in_channels; in_c++) {
                    for (int k = 0; k < kernel_size; k++) {
                        int in_pos = out_pos * stride - padding + k;
                        
                        // Check bounds (zero padding)
                        if (in_pos >= 0 && in_pos < in_length) {
                            int input_idx = b * in_channels * in_length + in_c * in_length + in_pos;
                            int weight_idx = out_c * in_channels * kernel_size + in_c * kernel_size + k;
                            
                            sum += input->data[input_idx] * config->weights[weight_idx];
                        }
                    }
                }
                
                // Add bias
                if (config->bias) {
                    sum += config->bias[out_c];
                }
                
                // Store result
                int output_idx = b * out_channels * out_length + out_c * out_length + out_pos;
                output->data[output_idx] = sum;
            }
        }
    }
    
    log_debug("Conv1D forward: %dx%dx%d → %dx%dx%d (kernel=%d, stride=%d, padding=%d)",
              batch_size, in_channels, in_length, batch_size, out_channels, out_length,
              kernel_size, stride, padding);
}

void conv1d_backward(const tensor_t* grad_output, const tensor_t* input, const conv1d_config_t* config, tensor_t* grad_input,float* grad_weights, float* grad_bias) {
    if (!grad_output || !input || !config || !grad_input || 
        !grad_output->data || !input->data || !config->weights || !grad_input->data) {
        log_error("Invalid arguments for conv1d backward");
        return;
    }
    
    int batch_size = input->batch_size;
    int in_channels = input->channels;
    int in_length = input->length;
    int out_channels = config->out_channels;
    int kernel_size = config->kernel_size;
    int stride = config->stride;
    int padding = config->padding;
    int out_length = calc_output_length(in_length, kernel_size, stride, padding);
    
    // Initialize gradients
    tensor_fill_zeros(grad_input);
    if (grad_weights) {
        memset(grad_weights, 0, config->weights_size);
    }
    if (grad_bias) {
        memset(grad_bias, 0, config->bias_size);
    }
    
    // Compute gradients
    for (int b = 0; b < batch_size; b++) {
        for (int out_c = 0; out_c < out_channels; out_c++) {
            for (int out_pos = 0; out_pos < out_length; out_pos++) {
                int grad_output_idx = b * out_channels * out_length + out_c * out_length + out_pos;
                float grad_out = grad_output->data[grad_output_idx];
                
                // Gradient w.r.t. bias
                if (grad_bias) {
                    grad_bias[out_c] += grad_out;
                }
                
                // Gradients w.r.t. weights and input
                for (int in_c = 0; in_c < in_channels; in_c++) {
                    for (int k = 0; k < kernel_size; k++) {
                        int in_pos = out_pos * stride - padding + k;
                        
                        if (in_pos >= 0 && in_pos < in_length) {
                            int input_idx = b * in_channels * in_length + in_c * in_length + in_pos;
                            int weight_idx = out_c * in_channels * kernel_size + in_c * kernel_size + k;
                            
                            float input_val = input->data[input_idx];
                            float weight_val = config->weights[weight_idx];
                            
                            // Gradient w.r.t. input
                            grad_input->data[input_idx] += grad_out * weight_val;
                            
                            // Gradient w.r.t. weights
                            if (grad_weights) {
                                grad_weights[weight_idx] += grad_out * input_val;
                            }
                        }
                    }
                }
            }
        }
    }
    
    log_debug("Conv1D backward: gradients computed");
}

conv1d_config_t* create_conv1d_config(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    conv1d_config_t* config = (conv1d_config_t*)sim_malloc(sizeof(conv1d_config_t));
    if (!config) {
        return NULL;
    }
    
    config->in_channels = in_channels;
    config->out_channels = out_channels;
    config->kernel_size = kernel_size;
    config->stride = stride;
    config->padding = padding;
    
    int num_weights = out_channels * in_channels * kernel_size;
    config->weights_size = num_weights * sizeof(float);
    config->bias_size = out_channels * sizeof(float);
    
    // Allocate weights and bias
    config->weights = (float*)sim_malloc(config->weights_size);
    config->bias = (float*)sim_malloc(config->bias_size);
    
    if (!config->weights || !config->bias) {
        if (config->weights) sim_free(config->weights);
        if (config->bias) sim_free(config->bias);
        sim_free(config);
        return NULL;
    }
    
    // Initialize weights with Xavier initialization
    float fan_in = in_channels * kernel_size;
    float fan_out = out_channels * kernel_size;
    float scale = sqrtf(2.0f / (fan_in + fan_out));
    
    for (int i = 0; i < num_weights; i++) {
        config->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    
    // Initialize bias to zero  
    memset(config->bias, 0, config->bias_size);
    
    log_info("Created Conv1D layer: %d→%d, kernel=%d, stride=%d, padding=%d (weights: %.1fKB)",
             in_channels, out_channels, kernel_size, stride, padding,
             config->weights_size / 1024.0f);
    
    return config;
}

void free_conv1d_config(conv1d_config_t* config) {
    if (config) {
        if (config->weights) sim_free_tracked(config->weights, config->weights_size);
        if (config->bias) sim_free_tracked(config->bias, config->bias_size);
        sim_free_tracked(config, sizeof(conv1d_config_t));
    }
}

// Simple Group Normalization (simplified version for this demo)
void group_norm_forward(tensor_t* input, tensor_t* output, int num_groups) {
    if (!input || !output || !input->data || !output->data) {
        log_error("Invalid tensors for group norm");
        return;
    }
    
    int batch_size = input->batch_size;
    int channels = input->channels;
    int length = input->length;
    int channels_per_group = channels / num_groups;
    
    for (int b = 0; b < batch_size; b++) {
        for (int g = 0; g < num_groups; g++) {
            int start_channel = g * channels_per_group;
            int end_channel = (g + 1) * channels_per_group;
            
            // Calculate mean for this group
            float sum = 0.0f;
            int count = 0;
            for (int c = start_channel; c < end_channel; c++) {
                for (int l = 0; l < length; l++) {
                    int idx = b * channels * length + c * length + l;
                    sum += input->data[idx];
                    count++;
                }
            }
            float mean = sum / count;
            
            // Calculate variance for this group
            float var_sum = 0.0f;
            for (int c = start_channel; c < end_channel; c++) {
                for (int l = 0; l < length; l++) {
                    int idx = b * channels * length + c * length + l;
                    float diff = input->data[idx] - mean;
                    var_sum += diff * diff;
                }
            }
            float variance = var_sum / count;
            float std_dev = sqrtf(variance + 1e-5f);  // Add epsilon for numerical stability
            
            // Normalize this group
            for (int c = start_channel; c < end_channel; c++) {
                for (int l = 0; l < length; l++) {
                    int idx = b * channels * length + c * length + l;
                    output->data[idx] = (input->data[idx] - mean) / std_dev;
                }
            }
        }
    }
    
    log_debug("Group norm: %d groups, mean normalization applied", num_groups);
}


void group_norm_backward(const tensor_t* input,const tensor_t* grad_output,tensor_t* grad_input,int num_groups) {
    log_info("Group norm backward: num_groups=%d", num_groups);
    if (!input || !grad_output || !grad_input ||
        !input->data || !grad_output->data || !grad_input->data) {
        log_error("Invalid tensors for group_norm_backward");
        return;
    }
    log_info("Group norm backward: num_groups=%d", num_groups);
    int B = input->batch_size;
    int C = input->channels;
    int L = input->length;

    if (C % num_groups != 0) {
        log_error("Channels (%d) not divisible by num_groups (%d)", C, num_groups);
        return;
    }

    int G = num_groups;
    int CpG = C / G;              // channels per group
    int N = CpG * L;              // elements per group
    const float eps = 1e-5f;
    log_info("Group norm backward: B=%d, C=%d, L=%d, G=%d, CpG=%d", B, C, L, G, CpG);
    tensor_fill_zeros(grad_input);
    log_info("Group norm backward: B=%d, C=%d, L=%d, G=%d, CpG=%d", B, C, L, G, CpG);
    for (int b = 0; b < B; b++) {
        const float* x     = input->data + b * C * L;
        const float* dy    = grad_output->data + b * C * L;
        float* dx          = grad_input->data + b * C * L;

        for (int g = 0; g < G; g++) {
            int c_start = g * CpG;
            int c_end   = c_start + CpG;

            /* ---- compute mean ---- */
            float mean = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                for (int l = 0; l < L; l++) {
                    mean += x[c * L + l];
                }
            }
            mean /= (float)N;

            /* ---- compute variance ---- */
            float var = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                for (int l = 0; l < L; l++) {
                    float v = x[c * L + l] - mean;
                    var += v * v;
                }
            }
            var /= (float)N;

            float inv_std = 1.0f / sqrtf(var + eps);

            /* ---- compute sums needed for backward ---- */
            float sum_dy = 0.0f;
            float sum_dy_xmu = 0.0f;

            for (int c = c_start; c < c_end; c++) {
                for (int l = 0; l < L; l++) {
                    int idx = c * L + l;
                    float xmu = x[idx] - mean;
                    sum_dy += dy[idx];
                    sum_dy_xmu += dy[idx] * xmu;
                }
            }
            
            /* ---- backward formula ---- */
            for (int c = c_start; c < c_end; c++) {
                for (int l = 0; l < L; l++) {
                    int idx = c * L + l;
                    float xmu = x[idx] - mean;

                    dx[idx] =
                        inv_std * (
                            dy[idx]
                            - sum_dy / N
                            - xmu * sum_dy_xmu / (N * (var + eps))
                        );
                }
            }
        }
    }

    log_info("Group norm backward done..................");
}



// // Backward pass for group normalization
// void group_norm_backward(const tensor_t* input, const tensor_t* grad_output, tensor_t* grad_input, int num_groups) {
//     if (!input || !grad_output || !grad_input || !input->data || !grad_output->data || !grad_input->data) {
//         log_error("Invalid tensors for group_norm_backward");
//         return;
//     }

//     int batch_size = input->batch_size;
//     int channels = input->channels;
//     int length = input->length;
//     int group_size = channels / num_groups;

//     // For each batch
//     for (int b = 0; b < batch_size; b++) {
//         float* input_data = input->data + b * channels * length;
//         float* grad_out_data = grad_output->data + b * channels * length;
//         float* grad_in_data = grad_input->data + b * channels * length;

//         // For each group
//         for (int g = 0; g < num_groups; g++) {
//             int c_start = g * group_size;
//             int c_end = (g + 1) * group_size;

//             // Compute mean and variance for this group
//             float mean = 0.0f, var = 0.0f;
//             int count = 0;
//             for (int c = c_start; c < c_end; c++) {
//                 for (int l = 0; l < length; l++) {
//                     int idx = c * length + l;
//                     mean += input_data[idx];
//                     count++;
//                 }
//             }
//             mean /= count;
//             for (int c = c_start; c < c_end; c++) {
//                 for (int l = 0; l < length; l++) {
//                     int idx = c * length + l;
//                     float diff = input_data[idx] - mean;
//                     var += diff * diff;
//                 }
//             }
//             var /= count;
//             float std = sqrtf(var + 1e-5f);

//             // Backpropagate
//             for (int c = c_start; c < c_end; c++) {
//                 for (int l = 0; l < length; l++) {
//                     int idx = c * length + l;
//                     // Simple gradient: pass through normalized grad_output
//                     grad_in_data[idx] = grad_out_data[idx] / std;
//                 }
//             }
//         }
//     }
// }
