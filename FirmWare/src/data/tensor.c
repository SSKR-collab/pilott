#include "nn_types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Global memory tracking
static size_t g_allocated_bytes = 0;
static size_t g_peak_bytes = 0;
static int g_allocation_count = 0;

void* sim_malloc(size_t size) {
    if (g_allocated_bytes + size > MEMORY_LIMIT_BYTES) {
        log_error("Memory allocation failed: would exceed limit (%zu + %zu > %zu)",
                  g_allocated_bytes, size, (size_t)MEMORY_LIMIT_BYTES);
        return NULL;
    }
    
    void* ptr = malloc(size);
    if (ptr) {
        g_allocated_bytes += size;
        g_allocation_count++;
        if (g_allocated_bytes > g_peak_bytes) {
            g_peak_bytes = g_allocated_bytes;
        }
        log_debug("Allocated %zu bytes (total: %zu/%zu)", 
                  size, g_allocated_bytes, (size_t)MEMORY_LIMIT_BYTES);
    }
    return ptr;
}

void sim_free(void* ptr) {
    if (ptr) {
        free(ptr);
        g_allocation_count--;
        log_debug("Freed memory block (remaining allocations: %d)", g_allocation_count);
    }
}

void sim_free_tracked(void* ptr, size_t size) {
    if (ptr) {
        free(ptr);
        g_allocation_count--;
        if (g_allocated_bytes >= size) {
            g_allocated_bytes -= size;
        }
        log_debug("Freed %zu bytes (total: %zu/%zu)", 
                  size, g_allocated_bytes, (size_t)MEMORY_LIMIT_BYTES);
    }
}

void print_memory_usage(void) {
    log_info("Memory usage: current=%zu bytes, peak=%zu bytes, limit=%zu bytes", 
             g_allocated_bytes, g_peak_bytes, (size_t)MEMORY_LIMIT_BYTES);
    log_info("Active allocations: %d", g_allocation_count);
}

tensor_t* tensor_create(int batch_size, int channels, int length) {
    if (batch_size <= 0 || channels <= 0 || length <= 0) {
        log_error("Invalid tensor dimensions: %dx%dx%d", batch_size, channels, length);
        return NULL;
    }
    
    tensor_t* tensor = (tensor_t*)sim_malloc(sizeof(tensor_t));
    if (!tensor) {
        return NULL;
    }
    
    int total_elements = batch_size * channels * length;
    tensor->allocated_size = total_elements * sizeof(float);
    tensor->data = (float*)sim_malloc(tensor->allocated_size);
    
    if (!tensor->data) {
        sim_free(tensor);
        return NULL;
    }
    
    tensor->batch_size = batch_size;
    tensor->channels = channels;
    tensor->length = length;
    tensor->stride = length;  // Default stride for memory layout
    
    // Initialize to zero
    memset(tensor->data, 0, tensor->allocated_size);
    
    log_debug("Created tensor: %dx%dx%d (%d elements, %d bytes)",
              batch_size, channels, length, total_elements, tensor->allocated_size);
    
    return tensor;
}

void tensor_free(tensor_t* tensor) {
    if (tensor) {
        if (tensor->data) {
            sim_free_tracked(tensor->data, tensor->allocated_size);
        }
        sim_free_tracked(tensor, sizeof(tensor_t));
    }
}

void tensor_fill_random(tensor_t* tensor) {
    if (!tensor || !tensor->data) return;
    
    int total_elements = tensor->batch_size * tensor->channels * tensor->length;
    for (int i = 0; i < total_elements; i++) {
        tensor->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Range: [-1, 1]
    }
}

void tensor_fill_zeros(tensor_t* tensor) {
    if (!tensor || !tensor->data) return;
    memset(tensor->data, 0, tensor->channels * tensor->length * sizeof(float));  
}

void tensor_copy(tensor_t* dst, const tensor_t* src) {
    if (!dst || !src || !dst->data || !src->data) return;
    
    if (dst->batch_size != src->batch_size || 
        dst->channels != src->channels || 
        dst->length != src->length) {
        log_error("Tensor dimension mismatch for copy");
        return;
    }
    
    memcpy(dst->data, src->data, src->allocated_size);
}

void tensor_print(const tensor_t* tensor, const char* name) {
    if (!tensor || !tensor->data) {
        printf("%s: NULL tensor\n", name ? name : "tensor");
        return;
    }
    
    printf("%s: shape=(%d,%d,%d)\n", 
           name ? name : "tensor", 
           tensor->batch_size, tensor->channels, tensor->length);
    
    // Print first few elements for debugging
    int max_print = 10;
    int total_elements = tensor->batch_size * tensor->channels * tensor->length;
    int print_count = total_elements < max_print ? total_elements : max_print;
    
    printf("  data[0:%d] = [", print_count - 1);
    for (int i = 0; i < print_count; i++) {
        printf("%.3f", tensor->data[i]);
        if (i < print_count - 1) printf(", ");
    }
    if (total_elements > max_print) {
        printf(", ... (%d more)", total_elements - max_print);
    }
    printf("]\n");
}

// Helper function to get tensor element index
static inline int tensor_index(const tensor_t* tensor, int batch, int channel, int pos) {
    return batch * tensor->channels * tensor->stride + 
           channel * tensor->stride + pos;
}

float tensor_get(const tensor_t* tensor, int batch, int channel, int pos) {
    if (!tensor || !tensor->data) return 0.0f;
    if (batch >= tensor->batch_size || channel >= tensor->channels || pos >= tensor->length) {
        return 0.0f;
    }
    
    return tensor->data[tensor_index(tensor, batch, channel, pos)];
}

void tensor_set(tensor_t* tensor, int batch, int channel, int pos, float value) {
    if (!tensor || !tensor->data) return;
    if (batch >= tensor->batch_size || channel >= tensor->channels || pos >= tensor->length) {
        return;
    }
    
    tensor->data[tensor_index(tensor, batch, channel, pos)] = value;
}