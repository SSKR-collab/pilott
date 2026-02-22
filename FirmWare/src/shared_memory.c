#include "shared_memory.h"
#include "lw_pilot_sim.h"
#include "config_types.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdbool.h>

// Global shared memory manager
static shm_manager_t g_shm_manager = {0};

// Pipeline control variables (global for shutdown detection across devices)
static volatile pipeline_phase_t g_current_phase = PHASE_FORWARD_LAYER0;
static int g_pipeline_shm_id = -1;
volatile pipeline_phase_t* g_pipeline_ctrl = NULL;  // Non-static for extern access

int shm_init(void) {
    if (g_shm_manager.initialized) {
        return 0;  // Already initialized
    }
    
    memset(&g_shm_manager, 0, sizeof(shm_manager_t));
    g_shm_manager.initialized = 1;
    
    log_info("Shared memory system initialized");
    return 0;
}

// Helper function to get number of upstream workers that write forward data to this layer
static int get_forward_contributors(shm_layer_id_t layer_id) {
    switch (layer_id) {
        case SHM_LAYER0_INPUT: return 1;  // Head device
        case SHM_LAYER1_INPUT: return 1;  // Layer 0 has 1 worker
        case SHM_LAYER2_INPUT: return 2;  // Layer 1 has 2 workers
        case SHM_LAYER3_INPUT: return 3;  // Layer 2 has 3 workers
        case SHM_LAYER4_INPUT: return 4;  // Layer 3 has 4 workers
        case SHM_GRAD_LAYER0: return 2;   // Layer 1 has 2 workers (for gradients)
        case SHM_GRAD_LAYER1: return 3;   // Layer 2 has 3 workers
        case SHM_GRAD_LAYER2: return 4;   // Layer 3 has 4 workers
        case SHM_GRAD_LAYER3: return 1;   // Tail device
        case SHM_PIPELINE_CTRL: return 1;  // Pipeline control (dummy value)
        default:
            log_error("Invalid layer_id %d for forward contributor count", layer_id);
            return 1;
    }
}

// Helper function to get number of downstream workers that write backward gradients to this layer
static int get_backward_contributors(shm_layer_id_t layer_id) {
    switch (layer_id) {
        case SHM_LAYER0_INPUT: return 1;  // Layer 0 has 1 worker (writes gradients back)
        case SHM_LAYER1_INPUT: return 2;  // Layer 1 has 2 workers
        case SHM_LAYER2_INPUT: return 3;  // Layer 2 has 3 workers
        case SHM_LAYER3_INPUT: return 4;  // Layer 3 has 4 workers
        case SHM_LAYER4_INPUT: return 1;  // Tail device
        case SHM_GRAD_LAYER0: return 1;   // Not applicable (Layer 0 is endpoint)
        case SHM_GRAD_LAYER1: return 1;   // Layer 0 writes back
        case SHM_GRAD_LAYER2: return 2;   // Layer 1 writes back
        case SHM_GRAD_LAYER3: return 3;   // Layer 2 writes back
        case SHM_PIPELINE_CTRL: return 1;  // Pipeline control (dummy value)
        default:
            log_error("Invalid layer_id %d for backward contributor count", layer_id);
            return 1;
    }
}

int shm_create_segment(shm_layer_id_t layer_id, int channels, int length, int num_workers) {
    if (!g_shm_manager.initialized) {
        shm_init();
    }
    
    if (layer_id >= MAX_SHM_LAYERS) {
        log_error("Invalid layer ID: %d", layer_id);
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    
    // Calculate buffer size: header + channels * length * sizeof(float)
    size_t header_size = sizeof(shm_header_t);
    size_t data_size = channels * length * sizeof(float);
    seg->size = header_size + data_size;
    seg->channels = channels;
    seg->length = length;
    seg->num_workers = num_workers;
    
    log_debug("SHM Create: layer=%d, channels=%d, length=%d, header=%zu, data=%zu, total=%zu",
             layer_id, channels, length, header_size, data_size, seg->size);
    
    // Create unique key for this layer
    key_t key = 0x1234 + layer_id;
    
    // Try to get existing shared memory first
    seg->shm_id = shmget(key, seg->size, 0666);
    
    if (seg->shm_id < 0) {
        // Doesn't exist, create new
        seg->shm_id = shmget(key, seg->size, IPC_CREAT | 0666);
        if (seg->shm_id < 0) {
            log_error("Failed to create shared memory for layer %d: %s", layer_id, strerror(errno));
            return -1;
        }
        log_info("Created shared memory segment for layer %d (key=0x%x, size=%zu bytes)", 
                 layer_id, key, seg->size);
    } else {
        log_info("Attached to existing shared memory for layer %d", layer_id);
    }
    
    // Attach to shared memory
    seg->shm_ptr = shmat(seg->shm_id, NULL, 0);
    if (seg->shm_ptr == (void*)-1) {
        log_error("Failed to attach shared memory for layer %d: %s", layer_id, strerror(errno));
        return -1;
    }
    
    // Initialize header pointer and P2P flags
    seg->header = (shm_header_t*)seg->shm_ptr;
    seg->header->forward_ready = 0;
    seg->header->backward_ready = 0;
    seg->header->current_sample_id = 0;
    seg->header->completed_samples = 0;
    seg->header->samples_sent = 0;
    seg->header->forward_contributors_completed = 0;
    seg->header->backward_contributors_completed = 0;
    seg->header->forward_expected_contributors = get_forward_contributors(layer_id);
    seg->header->backward_expected_contributors = get_backward_contributors(layer_id);
    
    // Initialize barriers if multiple workers
    if (num_workers > 1 && !seg->forward_barrier.initialized) {
        pthread_barrierattr_init(&seg->forward_barrier.attr);
        pthread_barrierattr_setpshared(&seg->forward_barrier.attr, PTHREAD_PROCESS_SHARED);
        
        if (pthread_barrier_init(&seg->forward_barrier.barrier, &seg->forward_barrier.attr, num_workers) != 0) {
            log_error("Failed to initialize forward barrier for layer %d", layer_id);
            shmdt(seg->shm_ptr);
            return -1;
        }
        seg->forward_barrier.num_workers = num_workers;
        seg->forward_barrier.initialized = 1;
        
        pthread_barrierattr_init(&seg->backward_barrier.attr);
        pthread_barrierattr_setpshared(&seg->backward_barrier.attr, PTHREAD_PROCESS_SHARED);
        
        if (pthread_barrier_init(&seg->backward_barrier.barrier, &seg->backward_barrier.attr, num_workers) != 0) {
            log_error("Failed to initialize backward barrier for layer %d", layer_id);
            pthread_barrier_destroy(&seg->forward_barrier.barrier);
            shmdt(seg->shm_ptr);
            return -1;
        }
        seg->backward_barrier.num_workers = num_workers;
        seg->backward_barrier.initialized = 1;
        
        log_info("Initialized barriers for layer %d (%d workers)", layer_id, num_workers);
    }
    
    log_info("Shared memory segment ready: layer=%d, channels=%d, length=%d, size=%zu",
             layer_id, channels, length, seg->size);
    
    return 0;
}

void* shm_get_layer_output(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        return NULL;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->shm_ptr) return NULL;
    
    // Return pointer to data section (after header)
    return (char*)seg->shm_ptr + sizeof(shm_header_t);
}

// Helper function to get input channel count for gradient calculations (renamed to avoid conflict)
static int get_gradient_input_channels(shm_layer_id_t grad_layer_id) {
    switch (grad_layer_id) {
        case SHM_GRAD_LAYER0: return 1;   // Layer 0 processes 1 input channel
        case SHM_GRAD_LAYER1: return 16;  // Layer 1 processes 16 input channels 
        case SHM_GRAD_LAYER2: return 32;  // Layer 2 processes 32 input channels
        case SHM_GRAD_LAYER3: return 48;  // Layer 3 processes 48 input channels
        default:
            log_error("Invalid gradient layer_id %d for input channel lookup", grad_layer_id);
            return 16;  // Safe fallback
    }
}

// Helper function to get number of gradient contributors for a layer
static int get_gradient_contributors(shm_layer_id_t grad_layer_id) {
    switch (grad_layer_id) {
        case SHM_GRAD_LAYER0: return 2;  // Layer 1 has 2 workers
        case SHM_GRAD_LAYER1: return 3;  // Layer 2 has 3 workers
        case SHM_GRAD_LAYER2: return 4;  // Layer 3 has 4 workers
        case SHM_GRAD_LAYER3: return 1;  // Tail device
        default:
            log_error("Invalid gradient layer_id %d for contributor count", grad_layer_id);
            return 1;
    }
}

size_t shm_get_worker_offset(shm_layer_id_t layer_id, int worker_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        return 0;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    
    // Check if this is a gradient segment (backward pass)
    bool is_gradient_segment = (layer_id >= SHM_GRAD_LAYER0 && layer_id <= SHM_GRAD_LAYER3);
    
    if (is_gradient_segment) {
        // For gradients: Each contributor writes complete gradients to separate buffer
        // Each buffer holds complete input channels from the layer receiving gradients
        int input_channels = get_gradient_input_channels(layer_id);
        size_t offset = worker_id * input_channels * seg->length * sizeof(float);
        return offset;
    } else {
        // For forward data: Workers write to separate channel ranges (original logic)
        int channels_per_worker = seg->channels / seg->num_workers;
        size_t offset = worker_id * channels_per_worker * seg->length * sizeof(float);
        return offset;
    }
}

int shm_write_tensor(shm_layer_id_t layer_id, int worker_id, const tensor_t* tensor) {
    if (layer_id >= MAX_SHM_LAYERS || !tensor) {
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->shm_ptr) {
        log_error("Shared memory not initialized for layer %d", layer_id);
        return -1;
    }
    
    size_t header_size = sizeof(shm_header_t);
    size_t worker_offset = shm_get_worker_offset(layer_id, worker_id);
    size_t tensor_size = tensor->channels * tensor->length * sizeof(float);
    
    if (header_size + worker_offset + tensor_size > seg->size) {
        log_error("Tensor write would exceed shared memory bounds (offset=%zu, size=%zu, max=%zu)",
                  worker_offset, tensor_size, seg->size);
        return -1;
    }
    
    // Copy tensor data to shared memory at worker's offset (after header)
    float* dest = (float*)((char*)seg->shm_ptr + header_size + worker_offset);
    memcpy(dest, tensor->data, tensor_size);
    
    log_debug("Worker %d wrote tensor to layer %d (offset=%zu, size=%zu bytes)",
              worker_id, layer_id, worker_offset, tensor_size);
    
    return 0;
}

int shm_read_tensor(shm_layer_id_t layer_id, tensor_t* tensor) {
    if (layer_id >= MAX_SHM_LAYERS || !tensor) {
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->shm_ptr) {
        log_error("Shared memory not initialized for layer %d", layer_id);
        return -1;
    }
    
    size_t header_size = sizeof(shm_header_t);
    size_t tensor_size = tensor->channels * tensor->length * sizeof(float);
    
    if (header_size + tensor_size > seg->size) {
        log_error("Tensor size exceeds shared memory size");
        return -1;
    }
    
    // Copy entire shared memory buffer to tensor (skip header)
    memcpy(tensor->data, (char*)seg->shm_ptr + header_size, tensor_size);
    
    log_debug("Read tensor from layer %d (size=%zu bytes)", layer_id, tensor_size);
    
    return 0;
}

void shm_barrier_forward(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        return;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (seg->forward_barrier.initialized && seg->num_workers > 1) {
        log_debug("Worker waiting at forward barrier for layer %d", layer_id);
        pthread_barrier_wait(&seg->forward_barrier.barrier);
        log_debug("Worker passed forward barrier for layer %d", layer_id);
    }
}

void shm_barrier_backward(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        return;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (seg->backward_barrier.initialized && seg->num_workers > 1) {
        log_debug("Worker waiting at backward barrier for layer %d", layer_id);
        pthread_barrier_wait(&seg->backward_barrier.barrier);
        log_debug("Worker passed backward barrier for layer %d", layer_id);
    }
}

// Completion counter functions for multi-worker coordination

// Wait for all upstream workers to complete writing forward data
int shm_wait_for_forward_complete(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) {
        log_error("Layer %d header not initialized", layer_id);
        return -1;
    }
    
    log_info("Layer %d: Waiting for forward_ready flag", layer_id);
    
    // Wait for forward_ready flag (set by last upstream writer)
    while (seg->header->forward_ready == 0) {
        // Check for shutdown
        if (g_pipeline_ctrl && *g_pipeline_ctrl == PHASE_DONE) {
            log_info("Shutdown detected while waiting for forward completion on layer %d", layer_id);
            return -1;
        }
        usleep(10);  // 10us polling interval
    }
    
    log_info("Layer %d: Forward data ready (sample_id=%d)", 
             layer_id, seg->header->current_sample_id);
    return 0;
}

// Signal that this worker completed writing forward data
int shm_signal_forward_complete(shm_layer_id_t layer_id, int sample_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        log_error("shm_signal_forward_complete: Invalid layer_id %d", layer_id);
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) {
        log_error("Layer %d header not initialized in shm_signal_forward_complete", layer_id);
        return -1;
    }
    
    // Atomically increment completion counter
    int completed = __sync_add_and_fetch(&seg->header->forward_contributors_completed, 1);
    int expected = seg->header->forward_expected_contributors;
    
    log_info("Layer %d: Worker signaled forward complete (%d/%d contributors, sample %d)", 
             layer_id, completed, expected, sample_id);
    
    // If I'm the last writer, reset counter, set ready flag, and update sample_id
    if (completed >= expected) {
        seg->header->current_sample_id = sample_id;
        seg->header->forward_contributors_completed = 0;
        __sync_synchronize();  // Memory barrier
        seg->header->forward_ready = 1;
        log_info("Layer %d: Last forward contributor, forward_ready=1, sample_id=%d", 
                 layer_id, sample_id);
    }
    
    return 0;
}

// Wait for all upstream workers to complete writing backward gradients
int shm_wait_for_backward_complete(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) {
        log_error("Layer %d header not initialized", layer_id);
        return -1;
    }
    
    log_info("Layer %d: Waiting for backward_ready flag", layer_id);
    
    // Wait for backward_ready flag (set by last upstream writer)
    while (seg->header->backward_ready == 0) {
        // Check for shutdown
        if (g_pipeline_ctrl && *g_pipeline_ctrl == PHASE_DONE) {
            log_info("Shutdown detected while waiting for backward completion on layer %d", layer_id);
            return -1;
        }
        usleep(10);  // 10us polling interval
    }
    
    log_info("Layer %d: Backward gradients ready", layer_id);
    return 0;
}

// Signal that this worker completed writing backward gradients
int shm_signal_backward_complete(shm_layer_id_t layer_id, int sample_id) {
    if (layer_id >= MAX_SHM_LAYERS) {
        log_error("shm_signal_backward_complete: Invalid layer_id %d", layer_id);
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) {
        log_error("Layer %d header not initialized in shm_signal_backward_complete", layer_id);
        return -1;
    }
    
    // Atomically increment completion counter
    int completed = __sync_add_and_fetch(&seg->header->backward_contributors_completed, 1);
    int expected = seg->header->backward_expected_contributors;
    
    log_info("Layer %d: Worker signaled backward complete (%d/%d contributors, sample %d)", 
             layer_id, completed, expected, sample_id);
    
    // If I'm the last writer, reset counter and set ready flag
    if (completed >= expected) {
        seg->header->backward_contributors_completed = 0;
        __sync_synchronize();  // Memory barrier
        seg->header->backward_ready = 1;
        log_info("Layer %d: Last backward contributor, backward_ready=1", layer_id);
    }
    
    return 0;
}

// Head waits for layer to complete forward pass by polling phase change
void shm_wait_layer_forward_complete(shm_layer_id_t layer_id) {
    // Simple busy-wait with small sleep
    // Workers signal completion by passing their barrier
    // Give enough time for all workers to complete
    usleep(5000);  // 5ms base wait
    log_debug("Head: Layer %d forward assumed complete", layer_id);
}

// Head waits for layer to complete backward pass  
void shm_wait_layer_backward_complete(shm_layer_id_t layer_id) {
    // Simple busy-wait with small sleep
    usleep(5000);  // 5ms base wait
    log_debug("Head: Layer %d backward assumed complete", layer_id);
}

// ============================================================================
// P2P COORDINATION VIA READY FLAGS
// ============================================================================

void shm_set_forward_ready(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return;
    
    __sync_lock_test_and_set(&seg->header->forward_ready, 1);
    log_debug("Layer %d: Set forward_ready=1", layer_id);
}

void shm_wait_for_forward_ready(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return;
    
    int wait_count = 0;
    
    while (__sync_fetch_and_add(&seg->header->forward_ready, 0) == 0) {
        usleep(100);  // 100us polling interval
        wait_count++;
        
        if (wait_count % 10000 == 0) {  // Log every 1 second
            log_debug("Still waiting for layer %d forward_ready (%.1fs)", 
                     layer_id, wait_count / 10000.0);
        }
        
        // Check for shutdown signal
        if (g_pipeline_ctrl && *g_pipeline_ctrl == PHASE_DONE) {
            log_info("Shutdown detected while waiting for layer %d", layer_id);
            return;
        }
    }
    
    log_debug("Layer %d: forward_ready detected", layer_id);
}

void shm_clear_forward_ready(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return;
    
    __sync_lock_test_and_set(&seg->header->forward_ready, 0);
    log_debug("Layer %d: Cleared forward_ready", layer_id);
}

void shm_set_backward_ready(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return;
    
    __sync_lock_test_and_set(&seg->header->backward_ready, 1);
    log_debug("Layer %d: Set backward_ready=1", layer_id);
}

void shm_wait_for_backward_ready(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return;
    
    int wait_count = 0;
    
    while (__sync_fetch_and_add(&seg->header->backward_ready, 0) == 0) {
        usleep(100);  // 100us polling interval
        wait_count++;
        
        if (wait_count % 10000 == 0) {  // Log every 1 second
            log_debug("Still waiting for layer %d backward_ready (%.1fs)", 
                     layer_id, wait_count / 10000.0);
        }
        
        // Check for shutdown signal
        if (g_pipeline_ctrl && *g_pipeline_ctrl == PHASE_DONE) {
            log_info("Shutdown detected while waiting for layer %d backward", layer_id);
            return;
        }
    }
    
    log_debug("Layer %d: backward_ready detected", layer_id);
}

void shm_clear_backward_ready(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return;
    
    __sync_lock_test_and_set(&seg->header->backward_ready, 0);
    log_debug("Layer %d: Cleared backward_ready", layer_id);
}

// Check if pipeline has completed processing current sample
int shm_check_layer_complete(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return 0;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return 0;
    
    // Layer is complete when:
    // 1. forward_ready is 0 (input consumed)
    // 2. backward_ready is 0 (gradients consumed)
    int forward_done = (__sync_fetch_and_add(&seg->header->forward_ready, 0) == 0);
    int backward_done = (__sync_fetch_and_add(&seg->header->backward_ready, 0) == 0);
    
    log_debug("Layer %d completion check: forward_ready=%d, backward_ready=%d", 
              layer_id, !forward_done, !backward_done);
    
    return (forward_done && backward_done);
}

// Wait for specific sample completion from Layer 0
int shm_wait_for_sample_completion(int sample_id) {
    shm_segment_t* layer0_seg = &g_shm_manager.layers[SHM_LAYER0_INPUT];
    if (!layer0_seg->header) {
        log_error("Layer 0 shared memory not initialized for completion check");
        return -1;
    }
    
    log_debug("Head: Waiting for sample %d completion...", sample_id);
    int wait_count = 0;
    int max_wait = 10000000; // 10 seconds timeout
    
    while (1) {
        int completed = __sync_fetch_and_add(&layer0_seg->header->completed_samples, 0);
        if (completed >= sample_id) {
            log_debug("Head: Sample %d completed (total completed: %d)", sample_id, completed);
            return 0;
        }
        
        wait_count++;
        if (wait_count % 10000 == 0) {
            log_debug("Head: Still waiting for sample %d (completed: %d, checks: %d)", 
                     sample_id, completed, wait_count);
        }
        
        if (wait_count >= max_wait) {
            log_error("Head: Timeout waiting for sample %d completion (completed: %d)", sample_id, completed);
            return -1;
        }
        
        usleep(100); // 100us polling
    }
}

// Signal that a sample has been completed by Layer 0 backward pass
void shm_signal_sample_complete(int sample_id) {
    shm_segment_t* layer0_seg = &g_shm_manager.layers[SHM_LAYER0_INPUT];
    if (!layer0_seg->header) {
        log_error("Layer 0 shared memory not initialized for completion signal");
        return;
    }
    
    // Atomically increment completed samples counter
    int new_count = __sync_add_and_fetch(&layer0_seg->header->completed_samples, 1);
    log_debug("Layer 0: Signaled sample %d complete (total: %d)", sample_id, new_count);
}

// Get current sample ID being processed in a layer
int shm_get_current_sample_id(shm_layer_id_t layer_id) {
    if (layer_id >= MAX_SHM_LAYERS) return -1;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return -1;
    
    return __sync_fetch_and_add(&seg->header->current_sample_id, 0);
}

// Set current sample ID being processed in a layer
void shm_set_current_sample_id(shm_layer_id_t layer_id, int sample_id) {
    if (layer_id >= MAX_SHM_LAYERS) return;
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->header) return;
    
    __sync_lock_test_and_set(&seg->header->current_sample_id, sample_id);
    log_debug("Layer %d: Set current sample ID to %d", layer_id, sample_id);
}

// Increment samples sent counter (called by Head)
int shm_increment_samples_sent(void) {
    if (!g_shm_manager.initialized) return -1;
    
    // Use Layer 0 header for global counters
    shm_segment_t* seg = &g_shm_manager.layers[SHM_LAYER0_INPUT];
    if (!seg->header) return -1;
    
    int new_count = __sync_fetch_and_add(&seg->header->samples_sent, 1) + 1;
    log_debug("Head: Incremented samples_sent to %d", new_count);
    return new_count;
}

// Increment samples completed counter (called by Tail)
int shm_increment_samples_completed(void) {
    if (!g_shm_manager.initialized) {
        log_error("Tail: shm_manager not initialized for completion counter");
        return -1;
    }
    
    // Use Layer 0 header for global counters  
    shm_segment_t* seg = &g_shm_manager.layers[SHM_LAYER0_INPUT];
    if (!seg->header) {
        log_error("Tail: Layer 0 header not available for completion counter");
        return -1;
    }
    
    int new_count = __sync_fetch_and_add(&seg->header->samples_completed, 1) + 1;
    log_info("Tail: Incremented samples_completed to %d", new_count);
    return new_count;
}

// **CORRECTED INPUT READING: Read complete layer output (not worker subset)**
int shm_read_complete_layer_output(shm_layer_id_t layer_id, tensor_t* tensor) {
    if (layer_id >= MAX_SHM_LAYERS || !tensor) {
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[layer_id];
    if (!seg->shm_ptr) {
        log_error("Shared memory not initialized for layer %d", layer_id);
        return -1;
    }
    
    // Read complete layer output (all channels from all workers)
    size_t header_size = sizeof(shm_header_t);
    size_t total_size = seg->channels * seg->length * sizeof(float);
    
    if (header_size + total_size > seg->size) {
        log_error("Complete layer read would exceed bounds (size=%zu, max=%zu)", 
                  total_size, seg->size);
        return -1;
    }
    
    // Ensure tensor has correct dimensions for complete layer output
    if (tensor->channels != seg->channels || tensor->length != seg->length) {
        log_error("Tensor dimensions mismatch: expected %dx%d, got %dx%d", 
                  seg->channels, seg->length, tensor->channels, tensor->length);
        return -1;
    }
    
    // Copy complete layer output data
    float* src = (float*)((char*)seg->shm_ptr + header_size);
    memcpy(tensor->data, src, total_size);
    
    log_debug("Read complete layer output: %d channels x %d length", 
              seg->channels, seg->length);
    
    return 0;
}

// **GRADIENT AGGREGATION FUNCTIONS FOR CORRECTED ARCHITECTURE**

// Write gradient contribution from downstream worker to gradient segment  
int shm_write_gradient_contribution(shm_layer_id_t grad_layer_id, int contributor_id, const tensor_t* gradients) {
    if (grad_layer_id >= MAX_SHM_LAYERS || !gradients) {
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[grad_layer_id];
    if (!seg->shm_ptr) {
        log_error("Gradient shared memory not initialized for layer %d", grad_layer_id);
        return -1;
    }
    
    // Calculate offset for this contributor's gradient buffer
    size_t header_size = sizeof(shm_header_t);
    int input_channels = get_gradient_input_channels(grad_layer_id);
    int max_contributors = get_gradient_contributors(grad_layer_id);
    size_t contributor_offset = contributor_id * input_channels * seg->length * sizeof(float);
    size_t tensor_size = gradients->channels * gradients->length * sizeof(float);
    
    log_debug("Gradient write: layer=%d, contributor=%d, input_channels=%d, seg->length=%d, tensor_size=%zu",
              grad_layer_id, contributor_id, input_channels, seg->length, tensor_size);
    
    // Validate contributor ID is within bounds
    if (contributor_id < 0 || contributor_id >= max_contributors) {
        log_error("Invalid contributor_id %d (max allowed: %d) for gradient layer %d", 
                  contributor_id, max_contributors, grad_layer_id);
        return -1;
    }
    
    // Validate tensor size matches expected size
    size_t expected_tensor_size = input_channels * seg->length * sizeof(float);
    if (tensor_size != expected_tensor_size) {
        log_error("Tensor size mismatch: expected %zu, got %zu (channels=%d vs %d, length=%d vs %d)",
                  expected_tensor_size, tensor_size, input_channels, gradients->channels, seg->length, gradients->length);
        return -1;
    }
    
    if (header_size + contributor_offset + tensor_size > seg->size) {
        log_error("Gradient contribution would exceed bounds (header=%zu + offset=%zu + tensor=%zu = %zu > max=%zu)",
                  header_size, contributor_offset, tensor_size, header_size + contributor_offset + tensor_size, seg->size);
        log_error("Buffer sizing: layer=%d, contributors=%d, input_channels=%d, length=%d",
                  grad_layer_id, max_contributors, input_channels, seg->length);
        return -1;
    }
    
    // Write gradient contribution to contributor's buffer
    float* dest = (float*)((char*)seg->shm_ptr + header_size + contributor_offset);
    memcpy(dest, gradients->data, tensor_size);
    
    log_debug("Contributor %d wrote gradients to layer %d (offset=%zu, size=%zu bytes)",
              contributor_id, grad_layer_id, contributor_offset, tensor_size);
    
    return 0;
}

// Read and aggregate all gradient contributions for this worker
int shm_read_aggregated_gradients(shm_layer_id_t grad_layer_id, tensor_t* aggregated_gradients) {
    if (grad_layer_id >= MAX_SHM_LAYERS || !aggregated_gradients) {
        return -1;
    }
    
    shm_segment_t* seg = &g_shm_manager.layers[grad_layer_id];
    if (!seg->shm_ptr) {
        log_error("Gradient shared memory not initialized for layer %d", grad_layer_id);
        return -1;
    }
    
    // Get number of contributors (downstream workers)
    int num_contributors = get_gradient_contributors(grad_layer_id);
    int input_channels = get_gradient_input_channels(grad_layer_id);
    size_t header_size = sizeof(shm_header_t);
    
    // Initialize aggregated gradients to zero
    memset(aggregated_gradients->data, 0, aggregated_gradients->channels * aggregated_gradients->length * sizeof(float));
    
    // Sum contributions from all downstream workers
    for (int contrib_id = 0; contrib_id < num_contributors; contrib_id++) {
        size_t contrib_offset = contrib_id * input_channels * seg->length * sizeof(float);
        float* contrib_data = (float*)((char*)seg->shm_ptr + header_size + contrib_offset);
        
        // Add this contribution to aggregated gradients
        for (int i = 0; i < aggregated_gradients->channels * aggregated_gradients->length; i++) {
            aggregated_gradients->data[i] += contrib_data[i];
        }
    }
    
    log_debug("Aggregated gradients from %d contributors for layer %d", num_contributors, grad_layer_id);
    return 0;
}

void shm_cleanup(void) {
    if (!g_shm_manager.initialized) {
        return;
    }
    
    for (int i = 0; i < MAX_SHM_LAYERS; i++) {
        shm_segment_t* seg = &g_shm_manager.layers[i];
        
        if (seg->shm_ptr && seg->shm_ptr != (void*)-1) {
            shmdt(seg->shm_ptr);
            seg->shm_ptr = NULL;
        }
        
        if (seg->shm_id > 0) {
            // Mark for deletion (will be deleted when all processes detach)
            shmctl(seg->shm_id, IPC_RMID, NULL);
            seg->shm_id = 0;
        }
        
        if (seg->forward_barrier.initialized) {
            pthread_barrier_destroy(&seg->forward_barrier.barrier);
            pthread_barrierattr_destroy(&seg->forward_barrier.attr);
            seg->forward_barrier.initialized = 0;
        }
        
        if (seg->backward_barrier.initialized) {
            pthread_barrier_destroy(&seg->backward_barrier.barrier);
            pthread_barrierattr_destroy(&seg->backward_barrier.attr);
            seg->backward_barrier.initialized = 0;
        }
    }
    
    g_shm_manager.initialized = 0;
    log_info("Shared memory cleaned up");
}

int shm_init_pipeline_control(void) {
    key_t key = 0x2000;  // Unique key for pipeline control
    
    // Try to get existing pipeline control segment
    g_pipeline_shm_id = shmget(key, sizeof(pipeline_phase_t), 0666);
    
    if (g_pipeline_shm_id < 0) {
        // Create new pipeline control segment
        g_pipeline_shm_id = shmget(key, sizeof(pipeline_phase_t), IPC_CREAT | 0666);
        if (g_pipeline_shm_id < 0) {
            log_error("Failed to create pipeline control shared memory: %s", strerror(errno));
            return -1;
        }
        log_info("Created pipeline control shared memory");
    }
    
    // Attach to shared memory
    g_pipeline_ctrl = (volatile pipeline_phase_t*)shmat(g_pipeline_shm_id, NULL, 0);
    if (g_pipeline_ctrl == (void*)-1) {
        log_error("Failed to attach to pipeline control shared memory: %s", strerror(errno));
        return -1;
    }
    
    // Initialize phase if we created the segment
    if (g_pipeline_shm_id >= 0) {
        *g_pipeline_ctrl = PHASE_IDLE;  // Start in IDLE state, head will signal when ready
    }
    
    log_info("Pipeline control initialized, current phase: %d", *g_pipeline_ctrl);
    return 0;
}

void shm_set_phase(pipeline_phase_t phase) {
    if (g_pipeline_ctrl == NULL) {
        if (shm_init_pipeline_control() < 0) {
            log_error("Failed to initialize pipeline control for set_phase");
            return;
        }
    }
    
    *g_pipeline_ctrl = phase;
    log_debug("Set pipeline phase to %d", phase);
}

void shm_wait_for_phase(pipeline_phase_t phase) {
    if (g_pipeline_ctrl == NULL) {
        if (shm_init_pipeline_control() < 0) {
            log_error("Failed to initialize pipeline control for wait_phase");
            return;
        }
    }
    
    log_debug("Waiting for pipeline phase %d (current: %d)", phase, *g_pipeline_ctrl);
    while (*g_pipeline_ctrl != phase && *g_pipeline_ctrl != PHASE_DONE) {
        usleep(1000);  // Wait 1ms
    }
    log_debug("Pipeline phase %d reached", phase);
}

pipeline_phase_t shm_get_phase(void) {
    if (g_pipeline_ctrl == NULL) {
        if (shm_init_pipeline_control() < 0) {
            log_error("Failed to initialize pipeline control for get_phase");
            return PHASE_FORWARD_LAYER0;
        }
    }
    return *g_pipeline_ctrl;
}

int shm_create_gradient_segments(const model_config_t* model_config) {
    if (!model_config) {
        log_error("Model config is required for gradient segment creation");
        return -1;
    }
    
    log_info("Creating gradient shared memory segments from model config");
    
    // Create gradient segments based on model layers
    int current_length = model_config->input_length;
    
    for (int i = 0; i < model_config->num_layers; i++) {
        if (strcmp(model_config->layers[i].type, "conv1d") == 0) {
            // Calculate output length after conv
            int kernel_size = model_config->layers[i].kernel_size;
            int stride = model_config->layers[i].stride;
            int padding = model_config->layers[i].padding;
            current_length = (current_length + 2 * padding - kernel_size) / stride + 1;
            
            // Create gradient segment for this layer
            shm_layer_id_t grad_layer = (shm_layer_id_t)(SHM_GRAD_LAYER0 - i);
            int channels = model_config->layers[i].out_channels;
            int num_workers = model_config->layers[i].num_devices;
            
            if (shm_create_segment(grad_layer, channels, current_length, num_workers) < 0) {
                log_error("Failed to create gradient segment for layer %d", i);
                return -1;
            }
            
            log_info("Created gradient segment %d: %d channels × %d length for %d workers", 
                     grad_layer, channels, current_length, num_workers);
        }
    }
    
    log_info("All gradient segments created successfully");
    return 0;
}
