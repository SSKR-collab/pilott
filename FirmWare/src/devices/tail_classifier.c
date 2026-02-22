#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "config_types.h"
#include "ipc_tensor.h"
#include <time.h>
#include <sys/resource.h>

static long get_rss_kb(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss;
}

// External configuration
extern model_config_t* g_model_config;

static int argmax(const float* x, int n) {
    int idx = 0;
    float max = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
            idx = i;
        }
    }
    return idx;
}

int run_tail_device(int device_id, int num_classes) {
    log_info("Starting Tail Device %d (Classifier) with %d classes", device_id, num_classes);
    
    if (!g_model_config) {
        log_error("Model configuration is required for tail device");
        return -1;
    }

    int last_layer = g_model_config->num_layers - 2;
    log_info("Tail device will read from layer %d", last_layer);

    // --- IPC SHM/SEM SETUP (Read from last worker layer) ---
    int num_workers_prev_layer = g_model_config->layers[last_layer].num_devices;
    int in_channels = g_model_config->layers[last_layer].out_channels;
    int input_length = g_model_config->layers[last_layer].output_length;
    log_info("Tail expects input from layer %d with %d channels and length %d", 
             last_layer, in_channels, input_length);

     // Open shared memory and semaphores for last layer
    char shm_name[64], fwd_sem_name[64], bwd_sem_name[64];
    snprintf(shm_name, sizeof(shm_name), "/ipc_tensor_L%d", last_layer +1);
    snprintf(fwd_sem_name, sizeof(fwd_sem_name), "/ipc_sem_L%d_fwd", last_layer+1 );
    snprintf(bwd_sem_name, sizeof(bwd_sem_name), "/ipc_sem_L%d_bwd", last_layer +1);

    size_t shm_size = sizeof(ipc_layer_shm_t) + sizeof(float) * in_channels * input_length*2; // Double size for forward input and backward gradients
    void* shm_ptr = NULL;
    if (ipc_tensor_open(shm_name, shm_size, 0, &shm_ptr) < 0) {
        log_error("Tail: Failed to open shared memory for last layer");
        return -1;
    }
    log_info("Tail: Opened shared memory %s (size %zu)", shm_name, shm_size);
    sem_t* fwd_sem = NULL;
    if (ipc_sem_open(fwd_sem_name, 0, &fwd_sem) < 0) {
        log_error("Tail: Failed to open forward semaphore for last layer");
        ipc_tensor_close(shm_ptr, shm_size);
        return -1;
    }
    log_info("Tail: Opened forward semaphore %s", fwd_sem_name);
    sem_t* bwd_sem = NULL;
    if (ipc_sem_open(bwd_sem_name, 0, &bwd_sem) < 0) {
        log_error("Tail: Failed to open backward semaphore for last layer");
        ipc_sem_close(fwd_sem);
        ipc_tensor_close(shm_ptr, shm_size);
        return -1;
    }
    log_info("Tail: Opened backward semaphore %s", bwd_sem_name);
    ipc_layer_shm_t* shm_tensor = (ipc_layer_shm_t*)shm_ptr;

    log_info("Tail reading from shared memory layer %d (%d channels, length %d)", 
             last_layer, in_channels, input_length);
    

    float* input_buffer = shm_tensor->buffer; // Use buffer for input data from last layer's shared memory
    float* grad_buffer = shm_tensor->buffer + (in_channels * input_length); // Use second half of buffer for gradients back to last layer


    // FC config
    int in_features = g_model_config->layers[last_layer+1].in_features;
    int out_features = g_model_config->layers[last_layer+1].out_features;
    fc_config_t* fc_config = create_fc_config(in_features, out_features);
    log_info("Tail FC layer config: in_features=%d, out_features=%d", in_features, out_features);
    // Setup pooling and fully connected layer
    tensor_t input_tensor, pooled_tensor, output_tensor;
    // Assume batch_size = 1 for simplicity
    input_tensor.batch_size = 1;
    input_tensor.channels = in_channels;
    input_tensor.length = input_length;
    input_tensor.data = input_buffer;

    // Pooling output: [batch_size, 2*channels, 1]
    pooled_tensor.batch_size = 1;
    pooled_tensor.channels = in_channels*2;
    pooled_tensor.length = 1;
    pooled_tensor.data = (float*)sim_malloc(sizeof(float) * in_channels * 2);

    // FC output: [batch_size, num_classes, 1]
    output_tensor.batch_size = 1;
    output_tensor.channels = out_features;
    output_tensor.length = 1;
    output_tensor.data = (float*)sim_malloc(sizeof(float) * out_features);

    // softmax probabilities
    tensor_t prob_tensor;
    prob_tensor.batch_size = 1;
    prob_tensor.channels = num_classes;
    prob_tensor.length = 1;
    prob_tensor.data = (float*)sim_malloc(sizeof(float) * num_classes);

    // Gradients for backward pass
    tensor_t grad_output, grad_pooled, grad_input;
    grad_output.batch_size = 1;
    grad_output.channels = out_features;
    grad_output.length = 1;
    grad_output.data = (float*)sim_malloc(sizeof(float) * out_features);
    grad_pooled.batch_size = 1;
    grad_pooled.channels = in_channels*2;
    grad_pooled.length = 1;
    grad_pooled.data = (float*)sim_malloc(sizeof(float) * in_channels * 2);
    grad_input.batch_size = 1;
    grad_input.channels = in_channels;
    grad_input.length = input_length;
    grad_input.data = (float*)sim_malloc(sizeof(float) * in_channels * input_length);

    float* grad_weights = (float*)sim_malloc(fc_config->weights_size);
    float* grad_bias = (float*)sim_malloc(fc_config->bias_size);

    // --- Adam optimizer buffers ---
    int num_weights = fc_config->weights_size / sizeof(float);
    int num_biases = fc_config->bias_size / sizeof(float);
    float* m_weights = (float*)sim_malloc(sizeof(float) * num_weights);
    float* v_weights = (float*)sim_malloc(sizeof(float) * num_weights);
    float* m_bias = (float*)sim_malloc(sizeof(float) * num_biases);
    float* v_bias = (float*)sim_malloc(sizeof(float) * num_biases);
    memset(m_weights, 0, sizeof(float) * num_weights);
    memset(v_weights, 0, sizeof(float) * num_weights);
    memset(m_bias, 0, sizeof(float) * num_biases);
    memset(v_bias, 0, sizeof(float) * num_biases);
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_eps   = 1e-8f;
    int adam_timestep = 0;

    // --- AdamW Weight Decay (regularization) ---
    float weight_decay = 0.0003f;
    log_info("AdamW weight decay enabled: lambda=%.4f", weight_decay);

    // --- Dropout before FC layer ---
    float dropout_rate = 0.2f;  // Drop 20% of pooled features during training
    float* dropout_mask = (float*)sim_malloc(sizeof(float) * in_channels * 2);
    tensor_t dropout_out;
    dropout_out.batch_size = 1;
    dropout_out.channels = in_channels * 2;
    dropout_out.length = 1;
    dropout_out.data = (float*)sim_malloc(sizeof(float) * in_channels * 2);
    // Gradient through dropout
    tensor_t grad_dropout;
    grad_dropout.batch_size = 1;
    grad_dropout.channels = in_channels * 2;
    grad_dropout.length = 1;
    grad_dropout.data = (float*)sim_malloc(sizeof(float) * in_channels * 2);
    log_info("Dropout enabled: rate=%.2f (before FC layer)", dropout_rate);

    // --- Early Stopping ---
    float best_test_acc = 0.0f;
    int best_epoch = 0;
    int patience = 50;           // Stop after 50 epochs with no improvement
    int epochs_without_improve = 0;
    int early_stopped = 0;
    // Save best weights for restoring
    float* best_fc_weights = (float*)sim_malloc(fc_config->weights_size);
    float* best_fc_bias = (float*)sim_malloc(fc_config->bias_size);
    memcpy(best_fc_weights, fc_config->weights, fc_config->weights_size);
    memcpy(best_fc_bias, fc_config->bias, fc_config->bias_size);
    log_info("Early stopping enabled: patience=%d epochs", patience);

    // --- Epoch tracking for LR schedule (Option C) + accuracy reset ---
    int current_epoch = 0;
    int prev_is_testing = 0;  // track phase transitions
    int train_sample_count = 0;

    // --- Timing & train accuracy tracking ---
    struct timespec run_start, epoch_ts, test_sample_start, test_sample_end;
    clock_gettime(CLOCK_MONOTONIC, &run_start);
    int train_correct = 0;
    int train_total = 0;
    double test_infer_time_sum = 0.0;  // cumulative test inference time per epoch
    int test_infer_count = 0;

    log_info("Tail device setup complete. Entering main processing loop...");
    log_info("Regularization: L2=%.4f, Dropout=%.2f, EarlyStop patience=%d",
             weight_decay, dropout_rate, patience);
    // Main loop: forward and backward pass
    int correct = 0;
    int total = 0;
    float test_loss_sum = 0.0f;
    int round = 1;
    while (1) {
        log_info("Tail: Processing sample %d", round);
        sem_wait(fwd_sem);
        
        int is_testing_now = shm_tensor->is_testing;
        
        // Start timing BEFORE forward pass for test samples
        if (is_testing_now) {
            clock_gettime(CLOCK_MONOTONIC, &test_sample_start);
        }
        
        log_info("input[0-4]: %.4f %.4f %.4f %.4f %.4f", 
                 input_tensor.data[0], input_tensor.data[1], input_tensor.data[2],
                 input_tensor.data[3], input_tensor.data[4]);
        // Forward pass: Pool → Dropout → FC → Softmax
        dual_pooling1d(&input_tensor, &pooled_tensor);
        
        // Apply dropout BEFORE FC layer (only during training)
        int is_testing_phase = shm_tensor->is_testing;
        dropout_forward(&pooled_tensor, &dropout_out, dropout_mask, dropout_rate, !is_testing_phase);
        
        fully_connected_forward(&dropout_out, fc_config, &output_tensor);
        softmax_forward(&output_tensor, &prob_tensor);

        int label = shm_tensor->label;  // from shared memory
        float loss = cross_entropy_loss(&prob_tensor, &label, 1);
        log_info("Loss = %f", loss);
        log_info("prob[0-9]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", 
                 prob_tensor.data[0], prob_tensor.data[1], prob_tensor.data[2],
                 prob_tensor.data[3], prob_tensor.data[4], prob_tensor.data[5],
                 prob_tensor.data[6], prob_tensor.data[7], prob_tensor.data[8],
                 prob_tensor.data[9]);
        
        // Backward pass
        int is_testing = shm_tensor->is_testing;

        // --- Detect epoch boundary: transition from testing -> training = new epoch ---
        if (prev_is_testing == 1 && is_testing == 0) {
            current_epoch++;
            float test_acc = (total > 0) ? (float)correct / total * 100.0f : 0.0f;
            float avg_loss = (total > 0) ? test_loss_sum / total : 0.0f;
            float train_acc = (train_total > 0) ? (float)train_correct / train_total * 100.0f : 0.0f;
            double avg_infer_ms = (test_infer_count > 0) ? test_infer_time_sum / test_infer_count * 1000.0 : 0.0;

            clock_gettime(CLOCK_MONOTONIC, &epoch_ts);
            double timespan = (epoch_ts.tv_sec - run_start.tv_sec) + (epoch_ts.tv_nsec - run_start.tv_nsec) / 1e9;

            time_t now = time(NULL);
            struct tm *tm_now = localtime(&now);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_now);

            long rss = get_rss_kb();

            log_info("=== EPOCH %d COMPLETE === Test Accuracy: %.2f%% (%d/%d), Avg Loss: %.4f",
                     current_epoch, test_acc, correct, total, avg_loss);
            log_info("[METRICS] Timestamp=%s | Timespan=%.1fs | Epoch=%d | Train_Acc=%.2f%% | Test_Acc=%.2f%% | Infer_Latency=%.3fms | Memory=%ldKB",
                     timestamp, timespan, current_epoch, train_acc, test_acc, avg_infer_ms, rss);

            // --- Early stopping check ---
            if (test_acc > best_test_acc) {
                best_test_acc = test_acc;
                best_epoch = current_epoch;
                epochs_without_improve = 0;
                // Save best weights
                memcpy(best_fc_weights, fc_config->weights, fc_config->weights_size);
                memcpy(best_fc_bias, fc_config->bias, fc_config->bias_size);
                log_info("[EARLY_STOP] New best test accuracy: %.2f%% at epoch %d", best_test_acc, best_epoch);
            } else {
                epochs_without_improve++;
                log_info("[EARLY_STOP] No improvement for %d/%d epochs (best=%.2f%% at epoch %d)",
                         epochs_without_improve, patience, best_test_acc, best_epoch);
            }
            if (epochs_without_improve >= patience && !early_stopped) {
                early_stopped = 1;
                // Restore best weights
                memcpy(fc_config->weights, best_fc_weights, fc_config->weights_size);
                memcpy(fc_config->bias, best_fc_bias, fc_config->bias_size);
                log_info("[EARLY_STOP] *** EARLY STOPPING TRIGGERED at epoch %d ***", current_epoch);
                log_info("[EARLY_STOP] Restored best weights from epoch %d (Test Acc=%.2f%%)",
                         best_epoch, best_test_acc);
                log_info("[EARLY_STOP] Training will continue in eval mode (no weight updates)");
            }

            // Reset accuracy metrics for next epoch
            correct = 0;
            total = 0;
            test_loss_sum = 0.0f;
            train_sample_count = 0;
            train_correct = 0;
            train_total = 0;
            test_infer_time_sum = 0.0;
            test_infer_count = 0;
        }
        prev_is_testing = is_testing;

        if(!is_testing) {
            train_sample_count++;
            /* Track train accuracy */
            int train_pred = argmax(prob_tensor.data, num_classes);
            if (train_pred == label) train_correct++;
            train_total++;

            // Skip weight updates if early stopping triggered
            if (!early_stopped) {
                cross_entropy_backward(&prob_tensor, &label, 1, &grad_output);
                // FC backward uses dropout_out (the input to FC) instead of pooled_tensor
                fully_connected_backward(&grad_output, &dropout_out, fc_config, &grad_pooled, grad_weights, grad_bias);
                // Dropout backward: propagate grad through dropout mask
                dropout_backward(&grad_pooled, dropout_mask, &grad_dropout);
                dual_pooling1d_backward(&grad_dropout, &input_tensor, &grad_input);
            
                // Adam optimizer with cosine annealing LR
                float base_lr = g_model_config->learning_rate;
                float lr = lr_cosine_annealing(base_lr, current_epoch, 60, 1e-5f);
                adam_timestep++;
                adam_update(fc_config->weights, grad_weights, m_weights, v_weights, num_weights, lr, adam_beta1, adam_beta2, adam_eps, adam_timestep, weight_decay);
                adam_update_bias(fc_config->bias, grad_bias, m_bias, v_bias, num_biases, lr, adam_beta1, adam_beta2, adam_eps, adam_timestep);
            } else {
                // After early stop: still compute gradients for workers but don't update FC
                cross_entropy_backward(&prob_tensor, &label, 1, &grad_output);
                fully_connected_backward(&grad_output, &dropout_out, fc_config, &grad_pooled, grad_weights, grad_bias);
                dropout_backward(&grad_pooled, dropout_mask, &grad_dropout);
                dual_pooling1d_backward(&grad_dropout, &input_tensor, &grad_input);
            }
            log_info("weights[0-9]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", 
                     fc_config->weights[0], fc_config->weights[1], fc_config->weights[2],
                     fc_config->weights[3], fc_config->weights[4], fc_config->weights[5],
                     fc_config->weights[6], fc_config->weights[7], fc_config->weights[8],
                     fc_config->weights[9]);
            log_info("bias[0-9]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f", 
                     fc_config->bias[0], fc_config->bias[1], fc_config->bias[2],
                     fc_config->bias[3], fc_config->bias[4], fc_config->bias[5],
                     fc_config->bias[6], fc_config->bias[7], fc_config->bias[8],
                     fc_config->bias[9]);
           
            memcpy(grad_buffer, grad_input.data, sizeof(float) * in_channels * input_length);

        }else{
            // For testing samples, time inference and compute accuracy
            int predicted_label = argmax(prob_tensor.data, num_classes);
            clock_gettime(CLOCK_MONOTONIC, &test_sample_end);
            double sample_infer_s = (test_sample_end.tv_sec - test_sample_start.tv_sec)
                                   + (test_sample_end.tv_nsec - test_sample_start.tv_nsec) / 1e9;
            if (predicted_label == label) {
                correct++;
            }
            total++;
            test_loss_sum += loss;
            test_infer_time_sum += sample_infer_s;
            test_infer_count++;
            log_info("Testing sample %d: True label=%d, Predicted=%d, Loss=%.4f, Accuracy=%.2f%%",
                     round, label, predicted_label, loss, (float)correct / total * 100.0f);
        }
        
        for (int w = 0; w < num_workers_prev_layer; w++) {
            sem_post(bwd_sem);
        }
        round++;
    }
    
    log_info("Tail device processing completed with %d samples, accuracy %.2f%%", total, (float)correct / total * 100.0f);

    
    // Cleanup
    free(pooled_tensor.data);
    free(output_tensor.data);
    free(prob_tensor.data);
    free(grad_output.data);
    free(grad_pooled.data);
    free(grad_input.data);
    free(grad_weights);
    free(grad_bias);
    free(m_weights);
    free(v_weights);
    free(m_bias);
    free(v_bias);
    free(dropout_mask);
    free(dropout_out.data);
    free(grad_dropout.data);
    free(best_fc_weights);
    free(best_fc_bias);
    ipc_sem_close(fwd_sem);
    ipc_sem_close(bwd_sem);
    ipc_tensor_close(shm_ptr, shm_size);
    sim_free(fc_config);
    shm_unlink(shm_name);
    sem_unlink(bwd_sem_name);
    sem_unlink(fwd_sem_name);
    log_info("Tail device %d completed successfully", device_id);
    return 0;
}
