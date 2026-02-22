#include "lw_pilot_sim.h"
#include <stdarg.h>
#include <time.h>

static FILE* g_log_file = NULL;
static int g_log_level = 0;  // 0=INFO, 1=DEBUG

void log_init(const char* filename) {
    if (filename) {
        g_log_file = fopen(filename, "w");
        if (!g_log_file) {
            fprintf(stderr, "Failed to open log file: %s\n", filename);
            g_log_file = stderr;
        }
    } else {
        g_log_file = stderr;
    }
}

void log_cleanup(void) {
    if (g_log_file && g_log_file != stderr && g_log_file != stdout) {
        fclose(g_log_file);
    }
    g_log_file = NULL;
}

static void log_message(const char* level, const char* format, va_list args) {
    if (!g_log_file) {
        g_log_file = stderr;
    }
    
    // Get timestamp
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    
    // Print timestamp and level
    fprintf(g_log_file, "[%ld.%03ld] %s: ", 
            ts.tv_sec % 1000, ts.tv_nsec / 1000000, level);
    
    // Print message
    vfprintf(g_log_file, format, args);
    fprintf(g_log_file, "\n");
    fflush(g_log_file);
}

void log_info(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message("INFO", format, args);
    va_end(args);
}

void log_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_message("ERROR", format, args);
    va_end(args);
}

void log_debug(const char* format, ...) {
    if (g_log_level >= 1) {
        va_list args;
        va_start(args, format);
        log_message("DEBUG", format, args);
        va_end(args);
    }
}

void set_log_level_debug(void) {
    g_log_level = 1;
}