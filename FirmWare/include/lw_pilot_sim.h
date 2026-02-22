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
#define DEFAULT_MEMORY_LIMIT_BYTES (256 * 1024)  // 256KB per device (nRF52840 constraint)

// Runtime memory limit (can be overridden via --mem-limit=N or config JSON)
extern size_t MEMORY_LIMIT_BYTES;

// Processing constraint: simulate 64 MHz Cortex-M4 clock
#define PROC_CLOCK_HZ  64000000   // 64 MHz
#define CYCLES_PER_FMUL 1         // single-cycle float multiply on Cortex-M4F
#define CYCLES_PER_FADD 1         // single-cycle float add
extern int g_proc_constraint;      // 0 = disabled, 1 = enabled (--proc-constraint / -p)
void proc_delay_flops(long flops);  // Simulate processing delay proportional to FLOPs

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