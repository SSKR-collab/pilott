#ifndef IPC_TENSOR_H
#define IPC_TENSOR_H

#include <semaphore.h>
#include <stddef.h>
#include "nn_types.h"
#include <sys/mman.h>

typedef struct {
    int label;
    int counter;
    int bwd_counter; // Backward pass synchronization counter
    int channels;
    int length;
    int sample_id; // Added for pipeline coordination
    int is_testing; // Flag to indicate if this is a testing sample
    float buffer[];
} ipc_layer_shm_t;

// Open or create shared memory for a layer
int ipc_tensor_open(const char* shm_name, size_t total_size, int create, void** shm_ptr);

// Close/unmap shared memory
int ipc_tensor_close(void* shm_ptr, size_t total_size);

// Open or create a semaphore for a layer
int ipc_sem_open(const char* sem_name, int create, sem_t** sem);

// Close a semaphore
int ipc_sem_close(sem_t* sem);

// Atomically increment the counter and return the new value
int ipc_counter_increment(int* counter);

// Utility to calculate total shared memory size for a layer
size_t ipc_layer_shm_size(int num_workers, size_t tensor_size);

#endif // IPC_TENSOR_H