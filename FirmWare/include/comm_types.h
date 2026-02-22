#ifndef COMM_TYPES_H
#define COMM_TYPES_H

#include "lw_pilot_sim.h"
#include "nn_types.h"
#include <semaphore.h>
#include <fcntl.h>

// Message types (kept for compatibility, now used for shared memory)
typedef enum {
    MSG_RAW_SAMPLE = 1,          // Head → Worker: raw input data
    MSG_CONV_FEATURES = 2,       // Worker → Tail: conv1 output features  
    MSG_CLASSIFICATION_GRAD = 3, // Tail → Worker: gradients for conv1
    MSG_SAMPLE_COMPLETE = 4,     // Processing complete signals
    MSG_TRAINING_DONE = 5,       // End of dataset
    MSG_HEARTBEAT = 6,           // Keep-alive messages
    MSG_ERROR = 7                // Error notifications
} message_type_t;

// Message header
typedef struct {
    uint32_t magic;              // Magic number for validation
    message_type_t type;         // Message type
    uint32_t payload_size;       // Size of payload in bytes
    uint32_t sequence_id;        // Sequence number
    uint32_t checksum;           // Simple checksum
} message_header_t;

// Complete message structure (maintained for compatibility)
typedef struct message {
    message_header_t header;
    void* payload;
    size_t payload_capacity;
} message_t;

// Shared memory communication functions (replacement for socket functions)
int shm_comm_init(int layer_id);
int shm_send_tensor(int layer_id, int worker_id, const tensor_t* tensor);
int shm_recv_tensor(int layer_id, tensor_t* tensor);
void shm_signal_training_done(void);
void shm_comm_cleanup(void);

// Message protocol functions (updated for shared memory)
message_t* message_create(message_type_t type, size_t payload_size);
void message_free(message_t* msg);
int message_set_tensor_payload(message_t* msg, const tensor_t* tensor);
int message_get_tensor_payload(const message_t* msg, tensor_t* tensor);
uint32_t calculate_checksum(const void* data, size_t size);

#define MESSAGE_MAGIC 0x4C575049  // "LWPI" in hex

#endif // COMM_TYPES_H