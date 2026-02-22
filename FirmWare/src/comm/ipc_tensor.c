#include "ipc_tensor.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

// Open or create shared memory for a layer
int ipc_tensor_open(const char* shm_name, size_t total_size, int create, void** shm_ptr) {
    int flags = O_RDWR | (create ? O_CREAT : 0);
    int fd = shm_open(shm_name, flags, 0666);
    if (fd < 0) return -1;
    if (create) {
        if (ftruncate(fd, total_size) < 0) {
            close(fd);
            return -1;
        }
    }
    void* ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return -1;
    *shm_ptr = ptr;
    memset(ptr, 0, total_size); // Initialize memory to zero
    return 0;
}

// Close/unmap shared memory
int ipc_tensor_close(void* shm_ptr, size_t total_size) {
    return munmap(shm_ptr, total_size);
}

// Open or create a semaphore for a layer
int ipc_sem_open(const char* sem_name, int create, sem_t** sem) {
    sem_t* s = sem_open(sem_name, create ? (O_CREAT | O_EXCL) : 0, 0666, 0);
    if (s == SEM_FAILED) {
        // If already exists and not creating, try opening without O_EXCL
        if (!create) s = sem_open(sem_name, 0);
        if (s == SEM_FAILED) return -1;
    }
    *sem = s;
    return 0;
}

// Close a semaphore
int ipc_sem_close(sem_t* sem) {
    return sem_close(sem);
}

// Atomically increment the counter and return the new value
int ipc_counter_increment(int* counter) {
    return __sync_add_and_fetch(counter, 1);
}

// Utility to calculate total shared memory size for a layer
size_t ipc_layer_shm_size(int num_workers, size_t tensor_size) {
    return sizeof(int) /*label*/ + sizeof(int) /*counter*/ + num_workers * tensor_size;
}