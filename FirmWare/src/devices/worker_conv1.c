#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "comm_types.h"
#include "config_types.h"
#include "worker_threads.h"
#include "ipc_tensor.h"


// External configuration
// extern device_json_config_t* g_device_config;
extern model_config_t* g_model_config;
// extern volatile pipeline_phase_t* g_pipeline_ctrl;  

// External runtime parameters (from command line)
extern int g_in_channels;
extern int g_out_channels;
extern int g_kernel_size;
extern int g_stride;
extern int g_padding;
extern int g_layer_id;
extern int g_worker_id;
extern int g_num_workers;


int run_worker_device(int device_id) {
    log_info("Starting Worker Device %d (Conv1D processor)", device_id);
    
    // Shared memory parameters (required)
    int layer_id = g_layer_id;  // Which layer (0-4) 
    int worker_id = g_worker_id;  // Which worker within layer (0-N)
    int num_workers = g_num_workers;  // Total workers in this layer
    int prev_layer_num_workers = (layer_id == 0) ? 1 : g_model_config->layers[layer_id - 1].num_devices;
    int next_layer_num_workers = g_model_config->layers[layer_id + 1].num_devices;
    // Get layer parameters from model config
    int in_channels = 16;  // Default input channels (from previous layer)  
    int worker_out_channels = 16;  // Each worker outputs 16 channels (uniform)
    int out_channels = worker_out_channels * num_workers; // Total output channels for this layer
    int kernel_size = 5;
    int stride = 1; 
    int padding = 2;
    
    if (layer_id < g_model_config->num_layers && strcmp(g_model_config->layers[layer_id].type, "conv1d") == 0) {
        // Use model config parameters
        kernel_size = g_model_config->layers[layer_id].kernel_size;
        stride = g_model_config->layers[layer_id].stride;
        padding = g_model_config->layers[layer_id].padding;
        
        // Total output channels for layer divided by number of workers
        out_channels = g_model_config->layers[layer_id].out_channels;
        in_channels = g_model_config->layers[layer_id].in_channels;
    }
    
    log_info("Worker configuration: Layer %d, Worker %d/%d, Conv1D %d→%d channels (kernel=%d, stride=%d, padding=%d)",
             layer_id, worker_id, num_workers, in_channels, worker_out_channels, kernel_size, stride, padding);
    

    // Calculate dimensions for input (from previous layer) and output
    int input_length = g_model_config->layers[layer_id].input_length;  // Initial input length from HEAD
    int output_length = (input_length + 2 * padding - kernel_size) / stride + 1;
    
    conv1d_config_t* conv_config = create_conv1d_config(in_channels, worker_out_channels, kernel_size, stride, padding);
    
    float* grad_weights = malloc(conv_config->weights_size);
    float* grad_bias    = malloc(conv_config->bias_size);

    // --- Adam optimizer buffers ---
    int num_w = conv_config->weights_size / sizeof(float);
    int num_b = conv_config->bias_size / sizeof(float);
    float* m_weights = (float*)calloc(num_w, sizeof(float));
    float* v_weights = (float*)calloc(num_w, sizeof(float));
    float* m_bias    = (float*)calloc(num_b, sizeof(float));
    float* v_bias    = (float*)calloc(num_b, sizeof(float));
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps   = 1e-8f;
    int adam_timestep = 0;

    // --- AdamW Weight Decay (regularization) ---
    float weight_decay = 0.0003f;

    // --- Epoch tracking for LR schedule (Option C) ---
    int current_epoch = 0;
    int prev_is_testing = 0;
    

    // --- IPC SHM/SEM SETUP B/W PREV AND CURRENT ---
    char prev_shm_name[64], prev_fwd_sem_name[64], prev_bwd_sem_name[64];
    snprintf(prev_shm_name, sizeof(prev_shm_name), "/ipc_tensor_L%d", layer_id);
    snprintf(prev_fwd_sem_name, sizeof(prev_fwd_sem_name), "/ipc_sem_L%d_fwd", layer_id);
    snprintf(prev_bwd_sem_name, sizeof(prev_bwd_sem_name), "/ipc_sem_L%d_bwd", layer_id);

    // Calculate correct shared memory size based on previous layer's total output

    size_t prev_shm_size = sizeof(ipc_layer_shm_t) + sizeof(float) * in_channels * input_length*2; // Double size for input data and backward gradients
    int attach = 0;
    void* prev_shm_ptr = NULL;
    if (ipc_tensor_open(prev_shm_name, prev_shm_size, attach, &prev_shm_ptr) < 0) {
        log_error("Failed to open shared memory %s", prev_shm_name);
        return -1;
    }
    sem_t* prev_fwd_sem = NULL;
    if (ipc_sem_open(prev_fwd_sem_name, attach, &prev_fwd_sem) < 0) {
        log_error("Failed to open forward semaphore %s", prev_fwd_sem_name);
        ipc_tensor_close(prev_shm_ptr, prev_shm_size);
        return -1;
    }
    sem_t* prev_bwd_sem = NULL;
    if (ipc_sem_open(prev_bwd_sem_name, attach, &prev_bwd_sem) < 0) {
        log_error("Failed to open backward semaphore %s", prev_bwd_sem_name);
        ipc_sem_close(prev_fwd_sem);
        ipc_tensor_close(prev_shm_ptr, prev_shm_size);
        return -1;
    }
    ipc_layer_shm_t* prev_shm = (ipc_layer_shm_t*)prev_shm_ptr;

    // --- IPC SHM/SEM SETUP B/W CURRENT AND NEXT ---
    int create = (worker_id == 0) ? 1 : 0;
    char next_shm_name[64], next_fwd_sem_name[64], next_bwd_sem_name[64];
    snprintf(next_shm_name, sizeof(next_shm_name), "/ipc_tensor_L%d", layer_id + 1);
    snprintf(next_fwd_sem_name, sizeof(next_fwd_sem_name), "/ipc_sem_L%d_fwd", layer_id + 1);
    snprintf(next_bwd_sem_name, sizeof(next_bwd_sem_name), "/ipc_sem_L%d_bwd", layer_id + 1);

    size_t next_shm_size = sizeof(ipc_layer_shm_t) + sizeof(float) * out_channels * output_length*2; // Double size for forward output and backward gradients
    void* next_shm_ptr = NULL;
    if (ipc_tensor_open(next_shm_name, next_shm_size, create, &next_shm_ptr) < 0) {
        log_error("Failed to create output shared memory %s", next_shm_name);
        ipc_sem_close(prev_fwd_sem);
        ipc_sem_close(prev_bwd_sem);
        ipc_tensor_close(prev_shm_ptr, prev_shm_size);
        return -1;
    }
    sem_t* next_fwd_sem = NULL;
    if (ipc_sem_open(next_fwd_sem_name, create, &next_fwd_sem) < 0) {
        log_error("Failed to create output forward semaphore %s", next_fwd_sem_name);
        ipc_tensor_close(next_shm_ptr, next_shm_size);
        ipc_sem_close(prev_fwd_sem);
        ipc_sem_close(prev_bwd_sem);
        ipc_tensor_close(prev_shm_ptr, prev_shm_size);
        return -1;
    }
    sem_t* next_bwd_sem = NULL;
    if (ipc_sem_open(next_bwd_sem_name, create, &next_bwd_sem) < 0) {
        log_error("Failed to create output backward semaphore %s", next_bwd_sem_name);
        ipc_sem_close(next_fwd_sem);
        ipc_tensor_close(next_shm_ptr, next_shm_size);
        ipc_sem_close(prev_fwd_sem);
        ipc_sem_close(prev_bwd_sem);
        ipc_tensor_close(prev_shm_ptr, prev_shm_size);
        return -1;
    }
    ipc_layer_shm_t* next_shm = (ipc_layer_shm_t*)next_shm_ptr;


    float* data_input_buffer = prev_shm->buffer;  //[in_channels * input_length,in_channels * input_length] = [input data, output gradient]
    float* data_output_buffer = next_shm->buffer; //[out_channels * output_length, out_channels * output_length] = [output data, input gradient]
    float* grad_input_buffer = next_shm->buffer + (out_channels * output_length); 
    float* grad_output_buffer = prev_shm->buffer + (in_channels * input_length); 


    // ---------- INPUT ----------
    tensor_t input_tensor;
    input_tensor.batch_size = 1;
    input_tensor.channels   = in_channels;
    input_tensor.length     = input_length;
    input_tensor.data       = data_input_buffer; // Use buffer for input data from previous layer's shared memory

    // ---------- FORWARD ----------
    tensor_t conv_out;
    conv_out.batch_size = 1;
    conv_out.channels   = worker_out_channels;
    conv_out.length     = output_length;
    conv_out.data       = malloc(sizeof(float) * worker_out_channels * output_length);

    tensor_t gn_out = conv_out;
    gn_out.data = malloc(sizeof(float) * worker_out_channels * output_length);
    tensor_t gn_pre_relu = gn_out;
    gn_pre_relu.data = malloc(sizeof(float) * worker_out_channels * output_length);

    // ---------- BACKWARD ----------
    tensor_t grad_out = conv_out;
    grad_out.data = malloc(sizeof(float) * worker_out_channels * output_length);

    tensor_t grad_gn = conv_out;
    grad_gn.data = malloc(sizeof(float) * worker_out_channels * output_length);

    tensor_t grad_input;
    grad_input.batch_size = 1;
    grad_input.channels   = in_channels;
    grad_input.length     = input_length;
    grad_input.data       = malloc(sizeof(float) * in_channels * input_length);

    tensor_t grad_conv;
    grad_conv.batch_size = 1;
    grad_conv.channels   = worker_out_channels;
    grad_conv.length     = output_length;
    grad_conv.data       = malloc(sizeof(float) * worker_out_channels * output_length);

    
    log_info("starting round loop...............");
    int round = 1;
    while (1) {
        log_info("Worker %d (Layer %d): Starting round %d", device_id, layer_id, round);
        if (worker_id == 0) {
            memset(grad_output_buffer, 0,sizeof(float) * in_channels * input_length); // Clear backward gradient buffer for this new sample
            __sync_synchronize();
        }
        
        // log_info("Worker %d (Layer %d): Waiting for sample %d...", device_id, layer_id, round);
        sem_wait(prev_fwd_sem);
        // log_info("Worker %d (Layer %d): Detected sample %d ready", device_id, layer_id, round);
        log_info("Worker %d (Layer %d): Input[0-4]: %.4f %.4f %.4f %.4f %.4f",
                 device_id, layer_id, round,
                 input_tensor.data[0], input_tensor.data[1],
                 input_tensor.data[2], input_tensor.data[3],
                 input_tensor.data[4]);
        memset(grad_weights, 0, conv_config->weights_size);
        memset(grad_bias, 0, conv_config->bias_size);

        conv1d_forward(&input_tensor, conv_config, &conv_out);
        // relu_forward(&conv_out, &gn_out);
        int num_groups = 8; 
        group_norm_forward(&conv_out, &gn_pre_relu, num_groups);
        relu_forward(&gn_pre_relu, &gn_out);
        

        // Log output data produced
        log_info("Worker %d (Layer %d): Sample %d, Output[0-4]: %.4f %.4f %.4f %.4f %.4f",
                 device_id, layer_id, round,
                 gn_out.data[0], gn_out.data[1],
                 gn_out.data[2], gn_out.data[3],
                 gn_out.data[4]);

        // Write output to next layer's shared memory at the correct offset
        next_shm->is_testing = prev_shm->is_testing; // Pass along training/testing flag
        next_shm->label = prev_shm->label; // Pass along label for loss calculation in tail
        memcpy(data_output_buffer + worker_id * worker_out_channels * output_length, gn_out.data,sizeof(float) * worker_out_channels * output_length);

        int new_val = ipc_counter_increment(&next_shm->counter);
        if (new_val == num_workers) {
            for (int w = 0; w < next_layer_num_workers; w++) {
                sem_post(next_fwd_sem);
            }
            __sync_lock_test_and_set(&next_shm->counter, 0);
        }

        // log_info("Worker %d (Layer %d): Completed forward pass for sample %d", device_id, layer_id, round);
        sem_wait(next_bwd_sem);
        // log_info("Worker %d (Layer %d): Detected backward pass start for sample %d", device_id, layer_id, round);
        int is_testing = prev_shm->is_testing;

        // --- Detect epoch boundary: testing -> training = new epoch ---
        if (prev_is_testing == 1 && is_testing == 0) {
            current_epoch++;
            log_info("Worker %d (Layer %d): Epoch %d started, LR=%.6f",
                     device_id, layer_id, current_epoch,
                     lr_schedule(g_model_config->learning_rate, current_epoch));
        }
        prev_is_testing = is_testing;

        if(!is_testing){
            float* grad_ptr = grad_input_buffer + worker_id * worker_out_channels * output_length;
            memcpy(grad_out.data,grad_ptr,sizeof(float) * worker_out_channels * output_length);
            log_info("Worker %d (Layer %d): Sample %d, Grad Input[0-4]: %.4f %.4f %.4f %.4f %.4f",
                     device_id, layer_id, round,
                     grad_out.data[0], grad_out.data[1],
                     grad_out.data[2], grad_out.data[3],
                     grad_out.data[4]);
            // relu_backward(&grad_out, &conv_out, &grad_gn);
            // conv1d_backward(&grad_gn,&input_tensor,conv_config,&grad_input,grad_weights,grad_bias);
            relu_backward(&grad_out, &gn_pre_relu, &grad_gn);
            log_info("relu output[0-4]: %.4f %.4f %.4f %.4f %.4f", 
                     gn_pre_relu.data[0], gn_pre_relu.data[1], gn_pre_relu.data[2],
                     gn_pre_relu.data[3], gn_pre_relu.data[4]);
            group_norm_backward(&conv_out, &grad_gn, &grad_conv, num_groups);
            log_info("gn backward output[0-4]: %.4f %.4f %.4f %.4f %.4f", 
                     grad_conv.data[0], grad_conv.data[1], grad_conv.data[2],
                     grad_conv.data[3], grad_conv.data[4]);
            conv1d_backward(&grad_conv,&input_tensor,conv_config,&grad_input,grad_weights,grad_bias);
            log_info("conv backward output[0-4]: %.4f %.4f %.4f %.4f %.4f", 
                     grad_input.data[0], grad_input.data[1], grad_input.data[2],
                     grad_input.data[3], grad_input.data[4]);
            // Adam optimizer with cosine annealing LR
            float base_lr = g_model_config->learning_rate;
            float lr = lr_cosine_annealing(base_lr, current_epoch, 60, 1e-5f);
            adam_timestep++;
            adam_update(conv_config->weights, grad_weights, m_weights, v_weights, num_w, lr, adam_beta1, adam_beta2, adam_eps, adam_timestep, weight_decay);
            adam_update_bias(conv_config->bias, grad_bias, m_bias, v_bias, num_b, lr, adam_beta1, adam_beta2, adam_eps, adam_timestep);
            log_info("weights[0-4]: %.4f %.4f %.4f %.4f %.4f", 
                     conv_config->weights[0], conv_config->weights[1], conv_config->weights[2],
                     conv_config->weights[3], conv_config->weights[4]);
            log_info("bias[0-4]: %.4f %.4f %.4f %.4f %.4f", 
                     conv_config->bias[0], conv_config->bias[1], conv_config->bias[2],
                     conv_config->bias[3], conv_config->bias[4]);
            log_info("Worker %d (Layer %d): Sample %d, Grad Output[0-4]: %.4f %.4f %.4f %.4f %.4f",
                     device_id, layer_id, round,
                     grad_input.data[0], grad_input.data[1],
                     grad_input.data[2], grad_input.data[3],
                     grad_input.data[4]);
            // Write gradients back to previous layer's shared memory
            float* prev_grad_ptr = grad_output_buffer; // Buffer for this worker's output gradients to previous layer
            for (int i = 0; i < in_channels * input_length; i++) {
                prev_grad_ptr[i] += grad_input.data[i];
            }
        }
        
        int bwd_val = ipc_counter_increment(&prev_shm->bwd_counter);
        if (bwd_val == num_workers) {
            for (int w = 0; w < prev_layer_num_workers; w++) {
                sem_post(prev_bwd_sem);
            }
            __sync_lock_test_and_set(&prev_shm->bwd_counter, 0);
        }

        round++;
    }

    log_info("Worker device processing completed");

    ipc_sem_close(prev_fwd_sem);
    ipc_sem_close(prev_bwd_sem);
    ipc_tensor_close(prev_shm_ptr, prev_shm_size);
    ipc_sem_close(next_fwd_sem);
    ipc_sem_close(next_bwd_sem);
    ipc_tensor_close(next_shm_ptr, next_shm_size);
free(conv_out.data);
free(gn_out.data);
free(gn_pre_relu.data);
free(grad_out.data);
free(grad_gn.data);
free(grad_input.data);
free(grad_conv.data);
free(grad_weights);
free(grad_bias);
free(m_weights);
free(v_weights);
free(m_bias);
free(v_bias);

    free_conv1d_config(conv_config);
    log_info("Worker device %d completed successfully", device_id);
    return 0;
}