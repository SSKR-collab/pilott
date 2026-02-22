#include "config_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Simple JSON parser for our specific config format
// We'll use a simple key-value parser since we control the format

static char* read_file_contents(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        log_error("Failed to open config file: %s", filename);
        return NULL;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* content = (char*)malloc(size + 1);
    if (!content) {
        fclose(f);
        return NULL;
    }
    
    fread(content, 1, size, f);
    content[size] = '\0';
    fclose(f);
    
    return content;
}

static char* find_value(const char* json, const char* key) {
    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\"", key);
    
    char* pos = strstr(json, search_key);
    if (!pos) return NULL;
    
    pos = strchr(pos, ':');
    if (!pos) return NULL;
    pos++;
    
    // Skip whitespace
    while (*pos && isspace(*pos)) pos++;
    
    // Extract value (number, string, or simple structure)
    char* value = (char*)malloc(256);
    int i = 0;
    
    if (*pos == '"') {
        pos++;
        while (*pos && *pos != '"' && i < 255) {
            value[i++] = *pos++;
        }
    } else if (*pos == '[') {
        // Array - just return the bracket content
        int bracket_count = 1;
        value[i++] = *pos++;
        while (*pos && bracket_count > 0 && i < 255) {
            if (*pos == '[') bracket_count++;
            if (*pos == ']') bracket_count--;
            value[i++] = *pos++;
        }
    } else {
        // Number or boolean
        while (*pos && !isspace(*pos) && *pos != ',' && *pos != '}' && i < 255) {
            value[i++] = *pos++;
        }
    }
    
    value[i] = '\0';
    return value;
}

static int parse_int(const char* str) {
    if (!str) return 0;
    return atoi(str);
}

// static void parse_connections(const char* json, const char* section, 
//                               connection_config_t* conns, int* count) {
//     *count = 0;
    
//     char section_key[256];
//     snprintf(section_key, sizeof(section_key), "\"%s\"", section);
    
//     char* pos = strstr(json, section_key);
//     if (!pos) {
//         log_debug("Section '%s' not found in JSON", section);
//         return;
//     }
    
//     log_debug("Found section '%s' in JSON", section);
    
//     pos = strchr(pos, '[');
//     if (!pos) {
//         log_debug("No array found for section '%s'", section);
//         return;
//     }
//     pos++;
    
//     // Find the end of this array to limit our search
//     char* array_end = pos;
//     int bracket_depth = 1;
//     while (*array_end && bracket_depth > 0) {
//         if (*array_end == '[') bracket_depth++;
//         if (*array_end == ']') {
//             bracket_depth--;
//             if (bracket_depth == 0) break;
//         }
//         array_end++;
//     }
    
//     if (!*array_end) {
//         log_debug("Could not find end of array for '%s'", section);
//         return;
//     }
    
//     // Parse each connection object within this array only
//     while (*pos && pos < array_end && *count < MAX_CONNECTIONS) {
//         // Skip to next object
//         while (*pos && *pos != '{' && pos < array_end) pos++;
//         if (*pos != '{' || pos >= array_end) break;
//         pos++;
        
//         // Find the end of this object
//         char* obj_end = strchr(pos, '}');
//         if (!obj_end || obj_end > array_end) break;
        
//         // Extract device_id (within current object only)
//         char* dev_str = pos;
//         while (dev_str < obj_end && strncmp(dev_str, "\"device_id\"", 11) != 0) dev_str++;
//         if (dev_str < obj_end) {
//             dev_str = strchr(dev_str, ':');
//             if (dev_str) {
//                 dev_str++;
//                 conns[*count].device_id = parse_int(dev_str);
//             }
//         }
        
//         // Extract port (within current object only)
//         char* port_str = pos;
//         while (port_str < obj_end && strncmp(port_str, "\"port\"", 6) != 0) port_str++;
//         if (port_str < obj_end) {
//             port_str = strchr(port_str, ':');
//             if (port_str) {
//                 port_str++;
//                 conns[*count].port = parse_int(port_str);
//             }
//         }
        
//         // Extract type (within current object only)
//         char* type_str = pos;
//         while (type_str < obj_end && strncmp(type_str, "\"type\"", 6) != 0) type_str++;
//         if (type_str < obj_end) {
//             type_str = strchr(type_str, '"');
//             if (type_str) {
//                 type_str++;
//                 type_str = strchr(type_str, '"');
//                 if (type_str) {
//                     type_str++;
//                     if (strncmp(type_str, "forward", 7) == 0) {
//                         conns[*count].type = CONN_FORWARD;
//                     } else {
//                         conns[*count].type = CONN_BACKWARD;
//                     }
//                 }
//             }
//         }
        
//         printf("[DEBUG] Parsed connection[%d] for '%s': device_id=%d, port=%d, type=%d\n",
//                *count, section, conns[*count].device_id, conns[*count].port, conns[*count].type);
        
//         (*count)++;
        
//         // Move to next object
//         pos = obj_end + 1;
//     }
    
//     printf("[DEBUG] Total '%s' connections parsed: %d\n", section, *count);
// }

// static void parse_layers(const char* json, layer_config_t* layers, int* count) {
//     *count = 0;
    
//     char* pos = strstr(json, "\"layers\"");
//     if (!pos) return;
    
//     pos = strchr(pos, '[');
//     if (!pos) return;
//     pos++;
    
//     // Parse each layer object
//     while (*pos && *pos != ']' && *count < MAX_LAYERS) {
//         // Skip to next object
//         while (*pos && *pos != '{') pos++;
//         if (*pos != '{') break;
        
//         char layer_obj[512];
//         char* obj_start = pos;
//         int brace_count = 1;
//         pos++;
//         int obj_len = 0;
        
//         while (*pos && brace_count > 0 && obj_len < 511) {
//             if (*pos == '{') brace_count++;
//             if (*pos == '}') brace_count--;
//             layer_obj[obj_len++] = *pos++;
//         }
//         layer_obj[obj_len-1] = '\0';
        
//         // Determine layer type
//         if (strstr(layer_obj, "\"conv1d\"")) {
//             layers[*count].type = LAYER_CONV1D;
            
//             char* val = find_value(layer_obj, "in_channels");
//             layers[*count].params.conv1d.in_channels = parse_int(val);
//             free(val);
            
//             val = find_value(layer_obj, "out_channels");
//             layers[*count].params.conv1d.out_channels = parse_int(val);
//             free(val);
            
//             val = find_value(layer_obj, "kernel_size");
//             layers[*count].params.conv1d.kernel_size = parse_int(val);
//             free(val);
            
//             val = find_value(layer_obj, "stride");
//             layers[*count].params.conv1d.stride = parse_int(val);
//             free(val);
            
//             val = find_value(layer_obj, "padding");
//             layers[*count].params.conv1d.padding = parse_int(val);
//             free(val);
            
//             (*count)++;
//         } else if (strstr(layer_obj, "\"fc\"")) {
//             layers[*count].type = LAYER_FULLY_CONNECTED;
            
//             char* val = find_value(layer_obj, "in_features");
//             layers[*count].params.fc.in_features = parse_int(val);
//             free(val);
            
//             val = find_value(layer_obj, "out_features");
//             layers[*count].params.fc.out_features = parse_int(val);
//             free(val);
            
//             (*count)++;
//         }
//     }
// }

// device_json_config_t* load_device_config(const char* config_file) {
//     log_info("Loading device config from: %s", config_file);
    
//     char* json = read_file_contents(config_file);
//     if (!json) {
//         return NULL;
//     }
    
//     device_json_config_t* config = (device_json_config_t*)malloc(sizeof(device_json_config_t));
//     if (!config) {
//         free(json);
//         return NULL;
//     }
    
//     memset(config, 0, sizeof(device_json_config_t));
    
//     // Parse device_id
//     char* val = find_value(json, "device_id");
//     config->device_id = parse_int(val);
//     free(val);
    
//     // Parse role
//     val = find_value(json, "role");
//     if (val) {
//         if (strcmp(val, "head") == 0) config->role = DEVICE_HEAD;
//         else if (strcmp(val, "worker") == 0) config->role = DEVICE_WORKER;
//         else if (strcmp(val, "tail") == 0) config->role = DEVICE_TAIL;
//         free(val);
//     }
    
//     // Parse dataset name (for head)
//     val = find_value(json, "dataset");
//     if (val) {
//         strncpy(config->dataset_name, val, sizeof(config->dataset_name) - 1);
//         free(val);
//     }
    
//     // Parse epochs
//     val = find_value(json, "epochs");
//     config->epochs = parse_int(val);
//     free(val);
    
//     // Parse num_classes
//     val = find_value(json, "num_classes");
//     config->num_classes = parse_int(val);
//     free(val);
    
//     // Parse memory limit
//     val = find_value(json, "memory_limit");
//     config->memory_limit = parse_int(val);
//     if (config->memory_limit == 0) config->memory_limit = MEMORY_LIMIT_BYTES;
//     free(val);
    
//     // Parse layers
//     parse_layers(json, config->layers, &config->num_layers);
    
//     // Parse connections
//     parse_connections(json, "upstream", config->upstream_conns, &config->num_upstream);
//     printf("[DEBUG] After parsing upstream: num_upstream=%d\n", config->num_upstream);
//     parse_connections(json, "downstream", config->downstream_conns, &config->num_downstream);
//     printf("[DEBUG] After parsing downstream: num_downstream=%d\n", config->num_downstream);
//     parse_connections(json, "backward", config->backward_conns, &config->num_backward);
//     printf("[DEBUG] After parsing backward: num_backward=%d\n", config->num_backward);
    
//     log_debug("Parsed connections: upstream=%d, downstream=%d, backward=%d",
//               config->num_upstream, config->num_downstream, config->num_backward);
    
//     // Parse tensor shapes
//     val = find_value(json, "input_channels");
//     if (val) {
//         config->input_shape.channels = parse_int(val);
//         free(val);
//     }
    
//     val = find_value(json, "input_length");
//     if (val) {
//         config->input_shape.length = parse_int(val);
//         free(val);
//     }
//     config->input_shape.batch_size = 1;
    
//     val = find_value(json, "output_channels");
//     if (val) {
//         config->output_shape.channels = parse_int(val);
//         free(val);
//     }
    
//     val = find_value(json, "output_length");
//     if (val) {
//         config->output_shape.length = parse_int(val);
//         free(val);
//     }
//     config->output_shape.batch_size = 1;
    
//     free(json);
    
//     log_info("Config loaded: Device %d, Role %d, %d layers, %d upstream, %d downstream",
//              config->device_id, config->role, config->num_layers,
//              config->num_upstream, config->num_downstream);
    
//     return config;
// }

// void free_device_config(device_json_config_t* config) {
//     if (config) {
//         free(config);
//     }
// }

// void print_device_config(const device_json_config_t* config) {
//     if (!config) return;
    
//     printf("Device Configuration:\n");
//     printf("  ID: %d\n", config->device_id);
//     printf("  Role: %d\n", config->role);
//     printf("  Layers: %d\n", config->num_layers);
//     printf("  Upstream connections: %d\n", config->num_upstream);
//     printf("  Downstream connections: %d\n", config->num_downstream);
//     printf("  Backward connections: %d\n", config->num_backward);
// }

// Model configuration functions
model_config_t* load_model_config(const char* config_file) {
    char* content = read_file_contents(config_file);
    if (!content) {
        return NULL;
    }
    
    model_config_t* config = (model_config_t*)calloc(1, sizeof(model_config_t));
    if (!config) {
        free(content);
        return NULL;
    }
    
    // Parse basic fields
    char* value = find_value(content, "name");
    if (value) {
        // Remove quotes
        if (value[0] == '"') {
            value++;
            char* end_quote = strchr(value, '"');
            if (end_quote) *end_quote = '\0';
        }
        strncpy(config->name, value, sizeof(config->name) - 1);
        free(value - (value > content && *(value-1) == '"' ? 1 : 0));
    }
    
    value = find_value(content, "version");
    if (value) {
        if (value[0] == '"') {
            value++;
            char* end_quote = strchr(value, '"');
            if (end_quote) *end_quote = '\0';
        }
        strncpy(config->version, value, sizeof(config->version) - 1);
        free(value - (value > content && *(value-1) == '"' ? 1 : 0));
    }
    
    // Global settings
    value = find_value(content, "dataset");
    if (value) {
        if (value[0] == '"') {
            value++;
            char* end_quote = strchr(value, '"');
            if (end_quote) *end_quote = '\0';
        }
        strncpy(config->dataset, value, sizeof(config->dataset) - 1);
        free(value - (value > content && *(value-1) == '"' ? 1 : 0));
    }
    
    value = find_value(content, "epochs");
    if (value) {
        config->epochs = atoi(value);
        free(value);
    }
    value = find_value(content, "learning_rate");
    if (value) {
        config->learning_rate = atof(value);
        free(value);
    }
    value = find_value(content, "num_classes");
    if (value) {
        config->num_classes = atoi(value);
        free(value);
    }
    
    value = find_value(content, "input_length");
    if (value) {
        config->input_length = atoi(value);
        free(value);
    }
    
    value = find_value(content, "memory_limit_bytes");
    if (value) {
        config->memory_limit_bytes = atoi(value);
        free(value);
    }
    
    // Parse layers array - simplified parsing
    char* layers_start = strstr(content, "\"layers\"");
    if (layers_start) {
        layers_start = strchr(layers_start, '[');
        if (layers_start) {
            layers_start++; // Skip '['
            int layer_count = 0;
            
            // Simple layer parsing
            char* layer_pos = strstr(layers_start, "\"id\"");
            while (layer_pos && layer_count < MAX_LAYERS && layer_pos < strstr(layers_start, "]")) {
                model_layer_t* layer = &config->layers[layer_count];
                
                // Parse layer id
                char* id_value = find_value(layer_pos, "id");  
                if (id_value) {
                    layer->id = atoi(id_value);
                    free(id_value);
                }
                
                // Parse layer type
                char* type_value = find_value(layer_pos, "type");
                if (type_value) {
                    if (type_value[0] == '"') {
                        type_value++;
                        char* end_quote = strchr(type_value, '"');
                        if (end_quote) *end_quote = '\0';
                    }
                    strncpy(layer->type, type_value, sizeof(layer->type) - 1);
                    free(type_value - (type_value > layer_pos && *(type_value-1) == '"' ? 1 : 0));
                }
                
                // Parse common fields
                char* val;
                if ((val = find_value(layer_pos, "in_channels"))) {
                    layer->in_channels = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "out_channels"))) {
                    layer->out_channels = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "kernel_size"))) {
                    layer->kernel_size = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "stride"))) {
                    layer->stride = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "padding"))) {
                    layer->padding = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "num_devices"))) {
                    layer->num_devices = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "in_features"))) {
                    layer->in_features = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "out_features"))) {
                    layer->out_features = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "input_length"))) {
                    layer->input_length  = atoi(val);
                    free(val);
                }
                if ((val = find_value(layer_pos, "output_length"))) {
                    layer->output_length = atoi(val);
                    free(val);
                }
                layer_count++;
                
                // Find next layer  
                layer_pos = strstr(layer_pos + 1, "\"id\"");
            }
            log_info("Parsed %d layers from config", layer_count);
            config->num_layers = layer_count;
        }
    }
    
    free(content);
    log_info("Loaded model config: %s v%s, %d layers", config->name, config->version, config->num_layers);
    return config;
}

void free_model_config(model_config_t* config) {
    if (config) {
        free(config);
    }
}

int get_layer_input_channels(const model_config_t* model_config, int layer_id) {
    if (!model_config || layer_id < 0 || layer_id >= model_config->num_layers) {
        return -1;
    }
    
    if (layer_id == 0) {
        // First layer gets input from head (1 channel for time series)
        return 1;
    }
    
    // Previous layer output becomes this layer input
    return model_config->layers[layer_id - 1].out_channels;
}

int get_layer_output_channels(const model_config_t* model_config, int layer_id) {
    if (!model_config || layer_id < 0 || layer_id >= model_config->num_layers) {
        return -1;
    }
    return model_config->layers[layer_id].out_channels;
}

int get_layer_num_devices(const model_config_t* model_config, int layer_id) {
    if (!model_config || layer_id < 0 || layer_id >= model_config->num_layers) {
        return -1;
    }
    return model_config->layers[layer_id].num_devices;
}

int get_total_conv_layers(const model_config_t* model_config) {
    if (!model_config) return 0;
    
    int conv_layers = 0;
    for (int i = 0; i < model_config->num_layers; i++) {
        if (strcmp(model_config->layers[i].type, "conv1d") == 0) {
            conv_layers++;
        }
    }
    return conv_layers;
}
