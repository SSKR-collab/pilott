#include "worker_threads.h"
#include "shared_memory.h"

worker_context_t* create_worker_context(int device_id, int layer_id, int in_channels, int out_channels, 
                                       int kernel_size, int stride, int padding) {
    worker_context_t* ctx = (worker_context_t*)sim_malloc(sizeof(worker_context_t));
    if (!ctx) {
        return NULL;
    }
    
    memset(ctx, 0, sizeof(worker_context_t));
    ctx->device_id = device_id;
    
    // Create Conv1D configuration using provided parameters
    ctx->conv_config = create_conv1d_config(in_channels, out_channels, kernel_size, stride, padding);
    if (!ctx->conv_config) {
        sim_free_tracked(ctx, sizeof(worker_context_t));
        return NULL;
    }
    
    // Allocate tensor buffers (assuming max input length of 300 for now)
    int max_input_length = 300;
    int output_length = (max_input_length + 2 * padding - kernel_size) / stride + 1;
    
    // Gradient w.r.t. this worker's input has the same channel count as forward input.
    int prev_layer_input_channels = in_channels;
    
    ctx->forward_input = tensor_create(1, in_channels, max_input_length);
    ctx->forward_temp = tensor_create(1, out_channels, output_length);
    ctx->forward_output = tensor_create(1, out_channels, output_length);
    ctx->backward_grad_output = tensor_create(1, out_channels, output_length);
    // **CORRECTED: Use previous layer input channels for gradient tensor**
    ctx->backward_grad_input = tensor_create(1, prev_layer_input_channels, max_input_length);

    // Allocate gradient buffers once (reusable across samples)
    ctx->grad_weights_size = ctx->conv_config->weights_size;
    ctx->grad_bias_size = ctx->conv_config->bias_size;
    ctx->grad_weights = (float*)sim_malloc(ctx->grad_weights_size);
    ctx->grad_bias = (float*)sim_malloc(ctx->grad_bias_size);
    
    log_info("Worker context tensors: forward_input=%dx%d, backward_grad_input=%dx%d (prev_layer_channels=%d)",
             in_channels, max_input_length, prev_layer_input_channels, max_input_length, prev_layer_input_channels);
    
    if (!ctx->forward_input || !ctx->forward_temp || !ctx->forward_output ||
        !ctx->backward_grad_output || !ctx->backward_grad_input ||
        !ctx->grad_weights || !ctx->grad_bias) {
        if (ctx->forward_input) tensor_free(ctx->forward_input);
        if (ctx->forward_temp) tensor_free(ctx->forward_temp);
        if (ctx->forward_output) tensor_free(ctx->forward_output);
        if (ctx->backward_grad_output) tensor_free(ctx->backward_grad_output);
        if (ctx->backward_grad_input) tensor_free(ctx->backward_grad_input);
        if (ctx->grad_weights) sim_free_tracked(ctx->grad_weights, ctx->grad_weights_size);
        if (ctx->grad_bias) sim_free_tracked(ctx->grad_bias, ctx->grad_bias_size);
        free_conv1d_config(ctx->conv_config);
        sim_free_tracked(ctx, sizeof(worker_context_t));
        return NULL;
    }
    
    log_info("Created worker context for device %d: %d→%d channels",
             device_id, in_channels, out_channels);
    
    return ctx;
}

void free_worker_context(worker_context_t* ctx) {
    if (ctx) {
        // Cleanup resources
        if (ctx->forward_input) tensor_free(ctx->forward_input);
        if (ctx->forward_temp) tensor_free(ctx->forward_temp);
        if (ctx->forward_output) tensor_free(ctx->forward_output);
    if (ctx->backward_grad_output) tensor_free(ctx->backward_grad_output);
    if (ctx->backward_grad_input) tensor_free(ctx->backward_grad_input);
    if (ctx->grad_weights) sim_free_tracked(ctx->grad_weights, ctx->grad_weights_size);
    if (ctx->grad_bias) sim_free_tracked(ctx->grad_bias, ctx->grad_bias_size);
    if (ctx->conv_config) free_conv1d_config(ctx->conv_config);
        
        sim_free_tracked(ctx, sizeof(worker_context_t));
    }
}
