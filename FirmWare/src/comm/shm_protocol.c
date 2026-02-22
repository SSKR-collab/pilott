#include "comm_types.h"
#include "shared_memory.h"
#include <stdlib.h>
#include <string.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <unistd.h>

// Global sequence counter for messages
static uint32_t g_sequence_id = 0;

// Shared memory communication state
typedef struct {
    sem_t* data_ready_sem;     // Signal when data is ready
    sem_t* data_consumed_sem;  // Signal when data is consumed  
    int layer_id;
    int initialized;
} shm_comm_state_t;

static shm_comm_state_t g_comm_state[MAX_SHM_LAYERS] = {0};

uint32_t calculate_checksum(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t checksum = 0;
    
    for (size_t i = 0; i < size; i++) {
        checksum = ((checksum << 1) | (checksum >> 31)) ^ bytes[i];
    }
    
    return checksum;
}

message_t* message_create(message_type_t type, size_t payload_size) {
    message_t* msg = (message_t*)sim_malloc(sizeof(message_t));
    if (!msg) {
        return NULL;
    }
    
    // Initialize header
    msg->header.magic = MESSAGE_MAGIC;
    msg->header.type = type;
    msg->header.payload_size = payload_size;
    msg->header.sequence_id = ++g_sequence_id;
    msg->header.checksum = 0;  // Will be calculated when payload is set
    
    // Allocate payload if needed
    msg->payload = NULL;
    msg->payload_capacity = 0;
    
    if (payload_size > 0) {
        msg->payload = sim_malloc(payload_size);
        if (!msg->payload) {
            sim_free(msg);
            return NULL;
        }
        msg->payload_capacity = payload_size;
        memset(msg->payload, 0, payload_size);
    }
    
    return msg;
}

void message_free(message_t* msg) {
    if (msg) {
        if (msg->payload) {
            sim_free(msg->payload);
        }
        sim_free(msg);
    }
}

int message_set_tensor_payload(message_t* msg, const tensor_t* tensor) {
    if (!msg || !tensor || !tensor->data) {
        log_error("Invalid arguments for message_set_tensor_payload");
        return -1;
    }
    
    // Calculate payload size: 4 ints for dimensions + tensor data
    size_t header_size = 4 * sizeof(int);
    size_t data_size = tensor->batch_size * tensor->channels * tensor->length * sizeof(float);
    size_t total_size = header_size + data_size;
    
    // Reallocate payload if needed
    if (total_size > msg->payload_capacity) {
        void* new_payload = realloc(msg->payload, total_size);
        if (!new_payload) {
            log_error("Failed to allocate %zu bytes for tensor payload", total_size);
            return -1;
        }
        msg->payload = new_payload;
        msg->payload_capacity = total_size;
    }
    
    // Pack tensor dimensions
    int* dims = (int*)msg->payload;
    dims[0] = tensor->batch_size;
    dims[1] = tensor->channels;
    dims[2] = tensor->length;
    dims[3] = data_size;  // Size of data portion
    
    // Copy tensor data
    memcpy((char*)msg->payload + header_size, tensor->data, data_size);
    
    // Update message header
    msg->header.payload_size = total_size;
    msg->header.checksum = calculate_checksum(msg->payload, total_size);
    
    log_debug("Packed tensor (%dx%dx%d) into message payload (%zu bytes)",
              tensor->batch_size, tensor->channels, tensor->length, total_size);
    
    return 0;
}

int message_get_tensor_payload(const message_t* msg, tensor_t* tensor) {
    if (!msg || !tensor || !msg->payload) {
        log_error("Invalid arguments for message_get_tensor_payload");
        return -1;
    }
    
    if (msg->header.payload_size < 4 * sizeof(int)) {
        log_error("Message payload too small for tensor header");
        return -1;
    }
    
    // Validate checksum
    uint32_t expected_checksum = calculate_checksum(msg->payload, msg->header.payload_size);
    if (msg->header.checksum != expected_checksum) {
        log_error("Message checksum mismatch (expected=0x%x, got=0x%x)", 
                  expected_checksum, msg->header.checksum);
        return -1;
    }
    
    // Unpack tensor dimensions
    const int* dims = (const int*)msg->payload;
    int batch_size = dims[0];
    int channels = dims[1];
    int length = dims[2];
    int data_size = dims[3];
    
    // Verify tensor dimensions match
    if (tensor->batch_size != batch_size || 
        tensor->channels != channels || 
        tensor->length != length) {
        log_error("Tensor dimension mismatch: expected %dx%dx%d, got %dx%dx%d",
                  tensor->batch_size, tensor->channels, tensor->length,
                  batch_size, channels, length);
        return -1;
    }
    
    // Copy tensor data
    const char* data_start = (const char*)msg->payload + 4 * sizeof(int);
    memcpy(tensor->data, data_start, data_size);
    
    log_debug("Unpacked tensor (%dx%dx%d) from message payload (%zu bytes)",
              batch_size, channels, length, msg->header.payload_size);
    
    return 0;
}

// Initialize shared memory communication for a layer
int shm_comm_init(int layer_id) {
    if (layer_id >= MAX_SHM_LAYERS || layer_id < 0) {
        log_error("Invalid layer ID for shared memory communication: %d", layer_id);
        return -1;
    }
    
    shm_comm_state_t* state = &g_comm_state[layer_id];
    if (state->initialized) {
        return 0;  // Already initialized
    }
    
    // Create semaphores for synchronization
    char sem_name[64];
    
    snprintf(sem_name, sizeof(sem_name), "/shm_data_ready_%d", layer_id);
    state->data_ready_sem = sem_open(sem_name, O_CREAT, 0644, 0);
    if (state->data_ready_sem == SEM_FAILED) {
        log_error("Failed to create data ready semaphore for layer %d", layer_id);
        return -1;
    }
    
    snprintf(sem_name, sizeof(sem_name), "/shm_data_consumed_%d", layer_id);
    state->data_consumed_sem = sem_open(sem_name, O_CREAT, 0644, 1);
    if (state->data_consumed_sem == SEM_FAILED) {
        log_error("Failed to create data consumed semaphore for layer %d", layer_id);
        sem_close(state->data_ready_sem);
        return -1;
    }
    
    state->layer_id = layer_id;
    state->initialized = 1;
    
    log_info("Shared memory communication initialized for layer %d", layer_id);
    return 0;
}

// Send data via shared memory (write to buffer and signal)
int shm_send_tensor(int layer_id, int worker_id, const tensor_t* tensor) {
    if (!tensor) {
        return -1;
    }
    
    // Initialize communication if needed
    if (shm_comm_init(layer_id) < 0) {
        return -1;
    }
    
    shm_comm_state_t* state = &g_comm_state[layer_id];
    
    // Wait until previous data is consumed
    sem_wait(state->data_consumed_sem);
    
    // Write tensor to shared memory
    if (shm_write_tensor((shm_layer_id_t)layer_id, worker_id, tensor) < 0) {
        sem_post(state->data_consumed_sem);
        return -1;
    }
    
    // Signal that data is ready
    sem_post(state->data_ready_sem);
    
    log_debug("Sent tensor via shared memory: layer=%d, worker=%d", layer_id, worker_id);
    return 0;
}

// Receive data via shared memory (wait for signal and read from buffer)
int shm_recv_tensor(int layer_id, tensor_t* tensor) {
    if (!tensor) {
        return -1;
    }
    
    // Initialize communication if needed
    if (shm_comm_init(layer_id) < 0) {
        return -1;
    }
    
    shm_comm_state_t* state = &g_comm_state[layer_id];
    
    // Wait for data to be ready
    sem_wait(state->data_ready_sem);
    
    // Read tensor from shared memory
    if (shm_read_tensor((shm_layer_id_t)layer_id, tensor) < 0) {
        sem_post(state->data_ready_sem);  // Re-signal if read failed
        return -1;
    }
    
    // Signal that data has been consumed
    sem_post(state->data_consumed_sem);
    
    log_debug("Received tensor via shared memory: layer=%d", layer_id);
    return 0;
}

// Signal that training is complete (for cleanup)
void shm_signal_training_done(void) {
    log_info("Training done signal sent via shared memory");
    
    // Signal all layers
    for (int i = 0; i < MAX_SHM_LAYERS; i++) {
        if (g_comm_state[i].initialized) {
            sem_post(g_comm_state[i].data_ready_sem);
        }
    }
}

// Cleanup shared memory communication
void shm_comm_cleanup(void) {
    for (int i = 0; i < MAX_SHM_LAYERS; i++) {
        shm_comm_state_t* state = &g_comm_state[i];
        if (state->initialized) {
            sem_close(state->data_ready_sem);
            sem_close(state->data_consumed_sem);
            
            char sem_name[64];
            snprintf(sem_name, sizeof(sem_name), "/shm_data_ready_%d", i);
            sem_unlink(sem_name);
            snprintf(sem_name, sizeof(sem_name), "/shm_data_consumed_%d", i);
            sem_unlink(sem_name);
            
            state->initialized = 0;
        }
    }
    
    log_info("Shared memory communication cleanup complete");
}