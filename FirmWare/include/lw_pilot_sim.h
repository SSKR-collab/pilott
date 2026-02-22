#ifndef LW_PILOT_SIM_H
#define LW_PILOT_SIM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <math.h>

// Forward declarations
typedef struct tensor tensor_t;
typedef struct device_config device_config_t;
typedef struct message message_t;

// Global constants
#define MAX_DEVICES 15
#define MAX_CLASSES 32
#define MAX_CHANNELS 128
#define MAX_SEQUENCE_LENGTH 1000
#define MEMORY_LIMIT_BYTES (1024 * 1024)  // 1MB per device (for ECG5000 dataset)

// Device roles
typedef enum {
    DEVICE_HEAD = 0,      // Data feeder
    DEVICE_WORKER = 1,    // Conv1D processor
    DEVICE_TAIL = 2       // Classifier
} device_role_t;

// Logging utilities
void log_init(const char* filename);
void log_cleanup(void);
void log_info(const char* format, ...);
void log_error(const char* format, ...);
void log_debug(const char* format, ...);
void set_log_level_debug(void);

#endif // LW_PILOT_SIM_H