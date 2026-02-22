#include "lw_pilot_sim.h"
#include "nn_types.h"
#include "comm_types.h"
#include "config_types.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Forward declarations for device implementations
int run_head_device(int device_id, const char* dataset_name);
int run_worker_device(int device_id);
int run_tail_device(int device_id, int num_classes);

// Global configuration
static int g_device_id = -1;
static device_role_t g_device_role = DEVICE_HEAD;
static char g_dataset_name[256] = "Cricket_X";
static int g_num_classes = 12;
static int g_debug = 0;
static char g_config_file[512] = "";
model_config_t* g_model_config = NULL;  // Global model configuration

// Runtime parameters (override config) - exported for device implementations
int g_in_channels = -1;
int g_out_channels = -1;
int g_kernel_size = -1;
int g_stride = -1;
int g_padding = -1;
int g_layer_id = -1;     // For shared memory: which layer (0-4)
int g_worker_id = -1;    // For shared memory: which worker within layer
int g_num_workers = -1;  // For shared memory: total workers in layer

void print_usage(const char* program_name) {
    printf("Usage: %s [--config=<config_file>] [--id=<device_id>] [options]\\n", program_name);
    printf("Options:\\n");
    printf("  --config=<file>       Device config JSON file\\n");
    printf("  --id=<num>            Device ID\\n");
    printf("  --role=<head|worker|tail>  Device role\\n");
    printf("  --dataset=<name>      Dataset name (default: Cricket_X)\\n");
    printf("  --classes=<num>       Number of classes (default: 12)\\n");
    printf("\\n  Removed Network Ports (now using shared memory):\\n");
    printf("\\n  Layer Parameters (for workers):\\n");
    printf("  --in-channels=<num>      Input channels\\n");
    printf("  --out-channels=<num>     Output channels\\n");
    printf("  --kernel-size=<num>      Convolution kernel size\\n");
    printf("  --stride=<num>           Convolution stride\\n");
    printf("  --padding=<num>          Convolution padding\\n");
    printf("\\n  Shared Memory Parameters (for workers):\\n");
    printf("  --layer-id=<num>         Layer ID (0-4)\\n");
    printf("  --worker-id=<num>        Worker ID within layer (0-based)\\n");
    printf("  --num-workers=<num>      Total workers in this layer\\n");
    printf("\\n  Other:\\n");
    printf("  --debug              Enable debug logging\\n");
    printf("  --help               Show this help message\\n");
    printf("\\nExamples:\\n");
    printf("  %s --config=worker_template.json --id=1 --layer-id=1 --worker-id=0 --num-workers=1\\n", program_name);
    printf("  %s --role=worker --id=1 --layer-id=0 --worker-id=0 --num-workers=1\\n", program_name);
}

int parse_arguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--config=", 9) == 0) {
            strncpy(g_config_file, argv[i] + 9, sizeof(g_config_file) - 1);
        } else if (strncmp(argv[i], "--id=", 5) == 0) {
            g_device_id = atoi(argv[i] + 5);
        } else if (strncmp(argv[i], "--role=", 7) == 0) {
            const char* role = argv[i] + 7;
            if (strcmp(role, "head") == 0) {
                g_device_role = DEVICE_HEAD;
            } else if (strcmp(role, "worker") == 0) {
                g_device_role = DEVICE_WORKER;
            } else if (strcmp(role, "tail") == 0) {
                g_device_role = DEVICE_TAIL;
            } else {
                log_error("Invalid role: %s (must be head, worker, or tail)", role);
                return -1;
            }
        } else if (strncmp(argv[i], "--dataset=", 10) == 0) {
            strncpy(g_dataset_name, argv[i] + 10, sizeof(g_dataset_name) - 1);
        } else if (strncmp(argv[i], "--classes=", 10) == 0) {
            g_num_classes = atoi(argv[i] + 10);
            if (g_num_classes <= 0 || g_num_classes > MAX_CLASSES) {
                log_error("Invalid number of classes: %d", g_num_classes);
                return -1;
            }
        } else if (strncmp(argv[i], "--in-channels=", 14) == 0) {
            g_in_channels = atoi(argv[i] + 14);
        } else if (strncmp(argv[i], "--out-channels=", 15) == 0) {
            g_out_channels = atoi(argv[i] + 15);
        } else if (strncmp(argv[i], "--kernel-size=", 14) == 0) {
            g_kernel_size = atoi(argv[i] + 14);
        } else if (strncmp(argv[i], "--stride=", 9) == 0) {
            g_stride = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--padding=", 10) == 0) {
            g_padding = atoi(argv[i] + 10);
        } else if (strncmp(argv[i], "--layer-id=", 11) == 0) {
            g_layer_id = atoi(argv[i] + 11);
        } else if (strncmp(argv[i], "--worker-id=", 12) == 0) {
            g_worker_id = atoi(argv[i] + 12);
        } else if (strncmp(argv[i], "--num-workers=", 14) == 0) {
            g_num_workers = atoi(argv[i] + 14);
        } else if (strncmp(argv[i], "--in-ch=", 8) == 0) {
            g_in_channels = atoi(argv[i] + 8);
        } else if (strncmp(argv[i], "--out-ch=", 9) == 0) {
            g_out_channels = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--kernel=", 9) == 0) {
            g_kernel_size = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--stride=", 9) == 0) {
            g_stride = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--padding=", 10) == 0) {
            g_padding = atoi(argv[i] + 10);
        } else if (strcmp(argv[i], "--debug") == 0) {
            g_debug = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            log_error("Unknown argument: %s", argv[i]);
            return -1;
        }
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Seed PRNG once so Conv/FC weight initialization is not deterministic across runs.
    srand((unsigned int)(time(NULL) ^ (unsigned int)getpid()));

    // Parse command line arguments
    if (parse_arguments(argc, argv) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    // Load model configuration (all devices need this)
    log_info("Loading model configuration...");
    const char *cfg_path = (g_config_file[0] != '\0') ? g_config_file : "configs/model_config_ecg5000.json";
    g_model_config = load_model_config(cfg_path);

    if (!g_model_config) {
        log_error("Failed to load model config - tried both configs/ and ../configs/ paths");
        return 1;
    }
    log_info("Model config loaded: %s, %d layers", g_model_config->name, g_model_config->num_layers);

    // Use dataset name from config if not overridden via CLI
    if (g_model_config->dataset[0] != '\0' &&
        strcmp(g_dataset_name, "Cricket_X") == 0) {
        strncpy(g_dataset_name, g_model_config->dataset, sizeof(g_dataset_name) - 1);
        log_info("Using dataset from config: %s", g_dataset_name);
    }
   
    if (g_debug) {
        set_log_level_debug();
    }
    
    // Check required parameters for shared memory
    if (g_device_role == DEVICE_WORKER) {
        if (g_layer_id < 0 || g_worker_id < 0 || g_num_workers <= 0) {
            log_error("Worker devices require: --layer-id, --worker-id, --num-workers");
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (g_device_id < 0) {
        log_error("Device ID is required");
        print_usage(argv[0]);
        return 1;
    }
    
    log_info("Device parameters: ID=%d, Role=%d, Layer=%d, Worker=%d/%d",
             g_device_id, g_device_role, g_layer_id, g_worker_id, g_num_workers);

    log_info("Starting LayerWise Pilot Simulation (Shared Memory Communication)");
    
    log_info("Device ID: %d, Role: %s", g_device_id,
             g_device_role == DEVICE_HEAD ? "HEAD" :
             g_device_role == DEVICE_WORKER ? "WORKER" : "TAIL");
    
    if (g_debug) {
        log_info("Debug mode enabled");
    }
    
    // Run device based on role
    int result = 0;
    switch (g_device_role) {
        case DEVICE_HEAD:
            result = run_head_device(g_device_id, g_dataset_name);
            break;
        case DEVICE_WORKER:
            result = run_worker_device(g_device_id);
            break;
        case DEVICE_TAIL:
            result = run_tail_device(g_device_id, g_num_classes);
            break;
        default:
            log_error("Invalid device role: %d", g_device_role); 
            result = 1;
            break;
    }
    
    if (result == 0) {
        log_info("Device %d completed successfully", g_device_id);
    } else {
        log_error("Device %d failed with result %d", g_device_id, result);
    }
    
    if (g_debug) {
        print_memory_usage();
    }
    

    log_cleanup();
    
    return result;
}
