#include "nn_types.h"
#include <math.h>
#include <float.h>

void global_average_pooling1d(const tensor_t* input, tensor_t* output) {
    if (!input || !output || !input->data || !output->data) {
        log_error("Invalid tensors for global average pooling");
        return;
    }
    
    int batch_size = input->batch_size;
    int channels = input->channels;
    int length = input->length;
    
    // Output should have shape [batch_size, channels, 1]
    if (output->batch_size != batch_size || output->channels != channels || output->length != 1) {
        log_error("Output tensor shape mismatch for global average pooling");
        return;
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            // Sum all values in this channel
            for (int l = 0; l < length; l++) {
                int input_idx = b * channels * length + c * length + l;
                sum += input->data[input_idx];
            }
            
            // Average and store
            int output_idx = b * channels + c;
            output->data[output_idx] = sum / length;
        }
    }
}

void global_max_pooling1d(const tensor_t* input, tensor_t* output) {
    if (!input || !output || !input->data || !output->data) {
        log_error("Invalid tensors for global max pooling");
        return;
    }
    
    int batch_size = input->batch_size;
    int channels = input->channels;
    int length = input->length;
    
    // Output should have shape [batch_size, channels, 1]
    if (output->batch_size != batch_size || output->channels != channels || output->length != 1) {
        log_error("Output tensor shape mismatch for global max pooling");
        return;
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            float max_val = -FLT_MAX;
            
            // Find maximum value in this channel
            for (int l = 0; l < length; l++) {
                int input_idx = b * channels * length + c * length + l;
                if (input->data[input_idx] > max_val) {
                    max_val = input->data[input_idx];
                }
            }
            
            // Store maximum
            int output_idx = b * channels + c;
            output->data[output_idx] = max_val;
        }
    }
}

void dual_pooling1d(const tensor_t* input, tensor_t* output) {
    if (!input || !output || !input->data || !output->data) {
        log_error("Invalid tensors for dual pooling");
        return;
    }
    
    int batch_size = input->batch_size;
    int channels = input->channels;
    int length = input->length;
    
    // Output should have shape [batch_size, channels*2, 1] (GAP + GMP concatenated)
    if (output->batch_size != batch_size || output->channels != channels * 2 || output->length != 1) {
        log_error("Output tensor shape mismatch for dual pooling");
        return;
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            float max_val = -FLT_MAX;
            
            // Calculate both sum and max in single pass
            for (int l = 0; l < length; l++) {
                int input_idx = b * channels * length + c * length + l;
                float val = input->data[input_idx];
                
                sum += val;
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // Store GAP in first half of channels
            int gap_idx = b * channels * 2 + c;
            output->data[gap_idx] = sum / length;
            
            // Store GMP in second half of channels
            int gmp_idx = b * channels * 2 + channels + c;
            output->data[gmp_idx] = max_val;
        }
    }
    
    log_debug("Dual pooling: %dx%dx%d → %dx%dx%d", 
              input->batch_size, input->channels, input->length,
              output->batch_size, output->channels, output->length);
}

void global_average_pooling1d_backward(const tensor_t* grad_output, const tensor_t* input, 
                                       tensor_t* grad_input) {
    if (!grad_output || !input || !grad_input || 
        !grad_output->data || !input->data || !grad_input->data) {
        log_error("Invalid tensors for global average pooling backward");
        return;
    }
    
    int batch_size = input->batch_size;
    int channels = input->channels;
    int length = input->length;
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            int output_idx = b * channels + c;
            float grad_val = grad_output->data[output_idx] / length;
            
            // Distribute gradient equally across all positions
            for (int l = 0; l < length; l++) {
                int input_idx = b * channels * length + c * length + l;
                grad_input->data[input_idx] = grad_val;
            }
        }
    }
}

void global_max_pooling1d_backward(const tensor_t* grad_output, const tensor_t* input, 
                                   tensor_t* grad_input) {
    if (!grad_output || !input || !grad_input || 
        !grad_output->data || !input->data || !grad_input->data) {
        log_error("Invalid tensors for global max pooling backward");
        return;
    }
    
    int batch_size = input->batch_size;
    int channels = input->channels;
    int length = input->length;
    
    // Initialize gradient input to zeros
    tensor_fill_zeros(grad_input);
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            // Find the position of maximum value
            int max_pos = 0;
            float max_val = -FLT_MAX;
            
            for (int l = 0; l < length; l++) {
                int input_idx = b * channels * length + c * length + l;
                if (input->data[input_idx] > max_val) {
                    max_val = input->data[input_idx];
                    max_pos = l;
                }
            }
            
            // Pass gradient only to maximum position
            int output_idx = b * channels + c;
            int max_input_idx = b * channels * length + c * length + max_pos;
            grad_input->data[max_input_idx] = grad_output->data[output_idx];
        }
    }
}

void dual_pooling1d_backward(const tensor_t* grad_output,const tensor_t* input,tensor_t* grad_input){
    if (!grad_output || !input || !grad_input ||
        !grad_output->data || !input->data || !grad_input->data) {
        log_error("Invalid tensors for dual pooling backward");
        return;
    }

    int B = input->batch_size;
    int C = input->channels;
    int L = input->length;

    // grad_output: [B, 2C, 1]
    // grad_input : [B, C, L]

    // Initialize gradients to zero
    tensor_fill_zeros(grad_input);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {

            /* -------- Global Average Pooling Backward -------- */
            int gap_out_idx = b * (2 * C) + c;
            float gap_grad = grad_output->data[gap_out_idx] / (float)L;

            for (int l = 0; l < L; l++) {
                int in_idx = b * C * L + c * L + l;
                grad_input->data[in_idx] += gap_grad;
            }

            /* -------- Global Max Pooling Backward -------- */
            int max_pos = 0;
            float max_val = -FLT_MAX;

            for (int l = 0; l < L; l++) {
                int in_idx = b * C * L + c * L + l;
                float v = input->data[in_idx];
                if (v > max_val) {
                    max_val = v;
                    max_pos = l;
                }
            }

            int gmp_out_idx = b * (2 * C) + C + c;
            int max_in_idx  = b * C * L + c * L + max_pos;

            grad_input->data[max_in_idx] += grad_output->data[gmp_out_idx];
        }
    }
}
