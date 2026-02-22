#ifndef CONFIG_TYPES_H
#define CONFIG_TYPES_H

#include "lw_pilot_sim.h"

#define MAX_CONNECTIONS 10
#define MAX_LAYERS 10
#define MAX_CONFIG_PATH 512

// Layer types
// typedef enum {
//     LAYER_CONV1D,
//     LAYER_POOLING,
//     LAYER_FULLY_CONNECTED,
//     LAYER_ACTIVATION
// } layer_type_t;

// Connection types
// typedef enum {
//     CONN_FORWARD,
//     CONN_BACKWARD
// } connection_type_t;

// Tensor shape
typedef struct {
    int batch_size;  
    int channels;
    int length;
} tensor_shape_t;

// Connection configuration
// typedef struct {
//     int device_id;
//     int port;
//     connection_type_t type;
// } connection_config_t;

// Conv1D layer configuration
typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
} conv1d_layer_config_t;

// Fully connected layer configuration
typedef struct {
    int in_features;
    int out_features;
} fc_layer_config_t;

// Generic layer configuration
// typedef struct {
//     layer_type_t type;
//     union {
//         conv1d_layer_config_t conv1d;
//         fc_layer_config_t fc;
//     } params;
// } layer_config_t;

// Model layer configuration (from model config JSON)
typedef struct {
    int id;
    char type[32];  // "conv1d" or "fc"
    int in_channels;
    int out_channels;  
    int kernel_size;
    int stride;
    int padding;
    int num_devices;
    int input_length;  // For conv1d, this is the length of the input sequence; for fc, this is in_features
    int output_length; // For conv1d, this is the length of the output sequence; for fc, this is out_features
    // FC layer fields
    int in_features;
    int out_features;
} model_layer_t;

// Complete model configuration
typedef struct {
    char name[256];
    char version[32];
    
    // Global settings
    char dataset[64];
    int epochs;
    int num_classes;
    int input_length;
    int memory_limit_bytes;
    float learning_rate;
    // Layer definitions
    int num_layers;
    model_layer_t layers[MAX_LAYERS];
} model_config_t;

typedef struct {
    int device_id;
    device_role_t role;
    
    // Dataset info (for head device)
    char dataset_name[256];
    int epochs;
    int num_classes;
    
    // Layer configurations
    int num_layers;
    // layer_config_t layers[MAX_LAYERS];
    
    // Connection topology
    // int num_upstream;
    // connection_config_t upstream_conns[MAX_CONNECTIONS];
    
    // int num_downstream;
    // connection_config_t downstream_conns[MAX_CONNECTIONS];
    
    // int num_backward;
    // connection_config_t backward_conns[MAX_CONNECTIONS];
    
    // Tensor dimensions
    tensor_shape_t input_shape;
    tensor_shape_t output_shape;
    
    // Memory limit
    size_t memory_limit;
} device_json_config_t;

// Function prototypes
device_json_config_t* load_device_config(const char* config_file);
void free_device_config(device_json_config_t* config);
void print_device_config(const device_json_config_t* config);

// Model config functions
model_config_t* load_model_config(const char* config_file);
void free_model_config(model_config_t* config);
int get_layer_input_channels(const model_config_t* model_config, int layer_id);
int get_layer_output_channels(const model_config_t* model_config, int layer_id);
int get_layer_num_devices(const model_config_t* model_config, int layer_id);
int get_total_conv_layers(const model_config_t* model_config);

// Global configuration variables  
extern device_json_config_t* g_device_config;
extern model_config_t* g_model_config;

#endif // CONFIG_TYPES_H
