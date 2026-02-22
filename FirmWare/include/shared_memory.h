#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <stddef.h>
#include <pthread.h>
#include "nn_types.h"
#include "config_types.h"

// Shared memory configuration
#define MAX_SHM_LAYERS 25
#define MAX_SHARED_BUFFER_SIZE (1024 * 1024)  // 1MB per layer buffer

// Shared memory IDs for each layer
typedef enum {
    // Forward data flow
    SHM_LAYER0_INPUT = 0,   // Head → Layer0 (1×300)
    SHM_LAYER1_INPUT = 1,   // Layer0 → Layer1 (16×300) 
    SHM_LAYER2_INPUT = 2,   // Layer1 → Layer2 (32×300)
    SHM_LAYER3_INPUT = 3,   // Layer2 → Layer3 (48×300)
    SHM_LAYER4_INPUT = 4,   // Layer3 → Tail (64×300)
    
    // Backward gradient flow
    SHM_GRAD_LAYER3 = 10,   // Tail → Layer3 (64×300)
    SHM_GRAD_LAYER2 = 11,   // Layer3 → Layer2 (48×300)
    SHM_GRAD_LAYER1 = 12,   // Layer2 → Layer1 (32×300)
    SHM_GRAD_LAYER0 = 13,   // Layer1 → Layer0 (16×300)
    
    // Pipeline control
    SHM_PIPELINE_CTRL = 20
} shm_layer_id_t;

// Synchronization barrier for workers in a layer
typedef struct {
    pthread_barrier_t barrier;
    pthread_barrierattr_t attr;
    int num_workers;
    int initialized;
} shm_barrier_t;

// Completion counter for layer synchronization
typedef struct {
    volatile int forward_counter;
    volatile int backward_counter;
    int num_workers;
} shm_completion_t;

// Header stored IN shared memory for P2P coordination
typedef struct {
    volatile int forward_ready;   // P2P: Data ready for next layer (1=ready, 0=not ready)
    volatile int backward_ready;  // P2P: Gradients ready for prev layer (1=ready, 0=not ready)
    volatile int samples_sent;    // Pipeline: Samples sent by Head
    volatile int samples_completed; // Pipeline: Samples completed by Tail
    volatile int current_sample_id; // Sample coordination: Current sample being processed
    volatile int current_label;     // Sample coordination: Label for current sample
    volatile int completed_samples; // Sample coordination: Total completed samples
    volatile int forward_contributors_completed;  // How many upstream workers finished writing forward data
    volatile int backward_contributors_completed; // How many upstream workers finished writing gradients
    volatile int forward_consumers_completed;     // How many downstream workers consumed forward data
    volatile int backward_consumers_completed;    // How many downstream workers consumed backward data
    int forward_expected_consumers;              // Total downstream consumers for forward data
    volatile int forward_count_sample_id;   // Sample ID currently being counted for forward contributors
    volatile int backward_count_sample_id;  // Sample ID currently being counted for backward contributors
    int forward_expected_contributors;  // Total upstream workers for forward pass
    int backward_expected_contributors; // Total upstream workers for backward pass
    int padding[1];               // Align to cache line
} shm_header_t;

// Shared memory segment for a layer
typedef struct {
    int shm_id;
    void* shm_ptr;                   // Points to shared memory (starts with shm_header_t)
    shm_header_t* header;            // Pointer to header in shared memory
    size_t size;
    int channels;
    int length;
    int num_workers;
    shm_barrier_t forward_barrier;   // All workers complete forward
    shm_barrier_t backward_barrier;  // All workers complete backward
    shm_completion_t* completion;    // Pointer to shared completion counters
} shm_segment_t;

// Global shared memory manager
typedef struct {
    shm_segment_t layers[MAX_SHM_LAYERS];
    int initialized;
} shm_manager_t;

// Pipeline phases for sequential processing
typedef enum {
    PHASE_IDLE = -1,              // Waiting for head to start
    PHASE_FORWARD_LAYER0 = 0,
    PHASE_FORWARD_LAYER1 = 1,
    PHASE_FORWARD_LAYER2 = 2, 
    PHASE_FORWARD_LAYER3 = 3,
    PHASE_FORWARD_TAIL = 4,
    PHASE_BACKWARD_TAIL = 10,
    PHASE_BACKWARD_LAYER3 = 11,
    PHASE_BACKWARD_LAYER2 = 12,
    PHASE_BACKWARD_LAYER1 = 13,
    PHASE_BACKWARD_LAYER0 = 14,
    PHASE_DONE = 20
} pipeline_phase_t;

// Initialize shared memory system
int shm_init(void);

// Create or attach to a shared memory segment
int shm_create_segment(shm_layer_id_t layer_id, int channels, int length, int num_workers);

// Get pointer to layer output buffer
void* shm_get_layer_output(shm_layer_id_t layer_id);

// Write tensor to specific offset in shared buffer (for workers)
int shm_write_tensor(shm_layer_id_t layer_id, int worker_id, const tensor_t* tensor);

// Read entire layer output (combines all workers)
int shm_read_tensor(shm_layer_id_t layer_id, tensor_t* tensor);

// Wait for all workers in layer to complete forward pass
void shm_barrier_forward(shm_layer_id_t layer_id);

// Wait for all workers in layer to complete backward pass
void shm_barrier_backward(shm_layer_id_t layer_id);

// Completion counter functions for multi-worker coordination
int shm_wait_for_forward_complete(shm_layer_id_t layer_id);
int shm_signal_forward_complete(shm_layer_id_t layer_id, int sample_id);
int shm_signal_forward_complete_with_label(shm_layer_id_t layer_id, int sample_id, int label);
int shm_wait_for_backward_complete(shm_layer_id_t layer_id);
int shm_signal_backward_complete(shm_layer_id_t layer_id, int sample_id);

// Head waits for all workers in layer to complete (polls completion counter)
void shm_wait_layer_forward_complete(shm_layer_id_t layer_id);
void shm_wait_layer_backward_complete(shm_layer_id_t layer_id);

// P2P coordination via ready flags
void shm_set_forward_ready(shm_layer_id_t layer_id);
void shm_wait_for_forward_ready(shm_layer_id_t layer_id);
void shm_clear_forward_ready(shm_layer_id_t layer_id);
void shm_consume_forward_ready(shm_layer_id_t layer_id);
void shm_set_backward_ready(shm_layer_id_t layer_id);
void shm_wait_for_backward_ready(shm_layer_id_t layer_id);
void shm_clear_backward_ready(shm_layer_id_t layer_id);
int shm_check_layer_complete(shm_layer_id_t layer_id);
int shm_wait_for_sample_completion(int sample_id);
void shm_signal_sample_complete(int sample_id);
int shm_get_current_sample_id(shm_layer_id_t layer_id);
void shm_set_current_sample_id(shm_layer_id_t layer_id, int sample_id);
int shm_get_current_label(shm_layer_id_t layer_id);
void shm_set_current_label(shm_layer_id_t layer_id, int label);
int shm_increment_samples_sent(void);
int shm_increment_samples_completed(void);
int shm_wait_for_sample_completion(int sample_number);

// Pipeline coordination functions
int shm_init_pipeline_control(void);
void shm_set_phase(pipeline_phase_t phase);
void shm_wait_for_phase(pipeline_phase_t phase);
pipeline_phase_t shm_get_phase(void);
int shm_create_gradient_segments(const model_config_t* model_config);

// Gradient aggregation functions for corrected architecture
int shm_read_complete_layer_output(shm_layer_id_t layer_id, tensor_t* tensor);
int shm_write_gradient_contribution(shm_layer_id_t grad_layer_id, int contributor_id, const tensor_t* gradients);
int shm_read_aggregated_gradients(shm_layer_id_t grad_layer_id, tensor_t* aggregated_gradients);
int shm_read_aggregated_gradient_slice(shm_layer_id_t grad_layer_id, int start_channel, tensor_t* sliced_gradients);

// Cleanup shared memory
void shm_cleanup(void);

// Get offset for specific worker in shared buffer
size_t shm_get_worker_offset(shm_layer_id_t layer_id, int worker_id);

#endif // SHARED_MEMORY_H
