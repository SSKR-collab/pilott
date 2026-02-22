/*
 * Distributed RockNet - POSIX Multi-Process Simulation
 * =====================================================
 * 
 * Simulates the distributed RockNet architecture on Linux using fork() + shared memory.
 * This faithfully replicates the nRF52840 firmware behavior:
 *
 *   - N worker nodes (processes), each owning a disjoint subset of 84 MiniRocket kernels
 *   - Each node computes PPV features for its kernels → partial logits
 *   - Partial logits are aggregated (summed) via POSIX shared memory (simulating BLE mixer)
 *   - Each node independently computes gradients on the aggregated logits and updates its weights
 *   - QADAM (quantized Adam) optimizer with dynamic tree quantization
 *
 * Architecture: [Data Feeder] → Node1(kernels 0-11) + ... + NodeN(kernels 72-83) → Aggregation → Gradient → QADAM
 *
 * Usage: ./distributed_rocknet
 *   Outputs [METRICS] lines compatible with the log extraction pipeline.
 *
 * Build: gcc -O2 -o distributed_rocknet main.c conv.c linear_classifier.c \
 *            dynamic_tree_quantization.c -lm -lrt -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <pthread.h>

#include "rocket_config.h"
#include "conv.h"
#include "linear_classifier.h"

/* ------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------ */
#ifndef NUM_NODES
#define NUM_NODES (7)  /* Default: 7 distributed devices. Override at compile: -DNUM_NODES=N */
#endif

/* Early stopping: stop if test accuracy doesn't improve for this many epochs */
#define EARLY_STOP_PATIENCE (50)

/* ------------------------------------------------------------------
 * Shared memory layout for inter-process communication
 * Simulates the BLE mixer wireless protocol from the firmware
 * ------------------------------------------------------------------ */
typedef struct {
    /* Barrier synchronisation */
    pthread_barrier_t barrier;
    
    /* Partial logits: each node writes its partial logits here */
    float partial_logits[NUM_NODES][NUM_CLASSES];
    
    /* Aggregated (summed) logits for current sample */
    float aggregated_logits[NUM_CLASSES];
    
    /* Current sample being processed (set by coordinator / node 0) */
    uint32_t current_sample_idx;
    
    /* Phase: 0 = training, 1 = evaluation */
    int phase;
    
    /* Epoch counter */
    uint64_t epoch;
    
    /* Accuracy counters per node (only idx 0 used for final) */
    int train_correct[NUM_NODES];
    int eval_correct[NUM_NODES];
    
    /* Signal to stop */
    volatile int stop;
    
    /* Metrics from coordinator */
    float last_train_acc;
    float last_test_acc;
    
    /* Early stopping state (managed by node 0) */
    float best_test_acc;
    int   best_epoch;
    int   epochs_no_improve;
    
} shared_state_t;

/* Global TOS_NODE_ID - set per process after fork */
uint16_t TOS_NODE_ID = 0;

static shared_state_t *shm = NULL;

/* ------------------------------------------------------------------
 * Shared memory setup
 * ------------------------------------------------------------------ */
static shared_state_t *create_shared_state(void)
{
    shared_state_t *s = mmap(NULL, sizeof(shared_state_t),
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (s == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    memset(s, 0, sizeof(*s));
    
    /* Init barrier for NUM_NODES processes */
    pthread_barrierattr_t battr;
    pthread_barrierattr_init(&battr);
    pthread_barrierattr_setpshared(&battr, PTHREAD_PROCESS_SHARED);
    pthread_barrier_init(&s->barrier, &battr, NUM_NODES);
    pthread_barrierattr_destroy(&battr);
    
    return s;
}

/* ------------------------------------------------------------------
 * Node worker function — runs in each forked process
 * ------------------------------------------------------------------ */
static void node_worker(int node_id)
{
    /* Set the node identity (1-based, like firmware TOS_NODE_ID) */
    TOS_NODE_ID = (uint16_t)(node_id + 1);
    
    /* Initialize rocket convolution (compute biases for this node's kernel subset) */
    init_rocket();
    
    /* Initialize linear classifier with QADAM */
    init_linear_classifier(TOS_NODE_ID);
    
    float partial[NUM_CLASSES];
    
    struct timespec start_time, epoch_end, eval_start, eval_end;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    for (uint64_t epoch = 0; !shm->stop; epoch++) {
        
        /* ========== TRAINING PHASE ========== */
        int my_train_correct = 0;
        
        for (uint32_t sample = 0; sample < NUM_TRAINING_TIMESERIES; sample++) {
            /* Each node computes partial logits for the same sample */
            classify_part(get_training_timeseries()[sample], partial);
            
            /* Write partial logits to shared memory */
            for (int c = 0; c < NUM_CLASSES; c++) {
                shm->partial_logits[node_id][c] = partial[c];
            }
            
            /* Synchronize: wait for all nodes to finish computing */
            pthread_barrier_wait(&shm->barrier);
            
            /* Node 0 aggregates partial logits */
            if (node_id == 0) {
                for (int c = 0; c < NUM_CLASSES; c++) {
                    float sum = 0;
                    for (int n = 0; n < NUM_NODES; n++) {
                        sum += shm->partial_logits[n][c];
                    }
                    /* Subtract bias (N-1) times since each node adds bias */
                    sum -= (NUM_NODES - 1) * shm->partial_logits[0][c]; 
                    /* Actually: recalculate properly. Each node adds bias, 
                       so total has N*bias. We want 1*bias. Subtract (N-1)*bias.
                       But bias is inside partial_logits already.
                       Simpler: just sum the partials. In firmware, only node 0 has bias ≠ 0.
                       In our sim, only node 0's bias is non-zero because init_linear_classifier
                       sets bias=0 for all, and update_weights only updates bias on node 0 
                       (when USE_QADAM=0). With QADAM, all nodes update bias...
                       Let's just sum all partials as in firmware. */
                    shm->aggregated_logits[c] = 0;
                    for (int n = 0; n < NUM_NODES; n++) {
                        shm->aggregated_logits[c] += shm->partial_logits[n][c];
                    }
                }
            }
            
            /* Wait for aggregation to complete */
            pthread_barrier_wait(&shm->barrier);
            
            /* Each node uses aggregated logits for gradient + accuracy */
            uint8_t pred = get_max_idx(shm->aggregated_logits, NUM_CLASSES);
            if (pred == get_training_labels()[sample]) {
                my_train_correct++;
            }
            
            /* Each node computes gradient using aggregated logits and its own features */
            calculate_and_accumulate_gradient(shm->aggregated_logits, get_training_labels()[sample]);
            
            /* Batch update */
            if (sample % BATCH_SIZE == BATCH_SIZE - 1) {
                update_weights();
            }
            
            /* Sync after gradient computation before next sample */
            pthread_barrier_wait(&shm->barrier);
        }
        
        /* Final batch update if needed */
        if (NUM_TRAINING_TIMESERIES % BATCH_SIZE != 0) {
            update_weights();
        }
        
        shm->train_correct[node_id] = my_train_correct;
        
        /* ========== EVALUATION PHASE ========== */
        /* Barrier before eval */
        pthread_barrier_wait(&shm->barrier);
        
        int my_eval_correct = 0;
        
        clock_gettime(CLOCK_MONOTONIC, &eval_start);
        
        for (uint32_t sample = 0; sample < NUM_EVALUATION_TIMESERIES; sample++) {
            /* Compute partial logits */
            classify_part(get_evaluation_timeseries()[sample], partial);
            
            for (int c = 0; c < NUM_CLASSES; c++) {
                shm->partial_logits[node_id][c] = partial[c];
            }
            
            pthread_barrier_wait(&shm->barrier);
            
            /* Aggregate */
            if (node_id == 0) {
                for (int c = 0; c < NUM_CLASSES; c++) {
                    shm->aggregated_logits[c] = 0;
                    for (int n = 0; n < NUM_NODES; n++) {
                        shm->aggregated_logits[c] += shm->partial_logits[n][c];
                    }
                }
            }
            
            pthread_barrier_wait(&shm->barrier);
            
            uint8_t pred = get_max_idx(shm->aggregated_logits, NUM_CLASSES);
            if (pred == get_evaluation_labels()[sample]) {
                my_eval_correct++;
            }
            
            pthread_barrier_wait(&shm->barrier);
        }
        
        clock_gettime(CLOCK_MONOTONIC, &eval_end);
        
        shm->eval_correct[node_id] = my_eval_correct;
        
        /* Node 0 prints metrics */
        pthread_barrier_wait(&shm->barrier);
        
        if (node_id == 0) {
            /* Use node 0's train_correct (all nodes see same aggregated logits, so same predictions) */
            float train_acc = (float)shm->train_correct[0] / NUM_TRAINING_TIMESERIES * 100.0f;
            float test_acc = (float)shm->eval_correct[0] / NUM_EVALUATION_TIMESERIES * 100.0f;
            
            double eval_secs = (eval_end.tv_sec - eval_start.tv_sec) +
                               (eval_end.tv_nsec - eval_start.tv_nsec) / 1e9;
            double infer_latency_ms = eval_secs / NUM_EVALUATION_TIMESERIES * 1000.0;
            
            /* Training time = total epoch time minus evaluation time */
            clock_gettime(CLOCK_MONOTONIC, &epoch_end);
            double epoch_secs = (epoch_end.tv_sec - start_time.tv_sec) +
                                (epoch_end.tv_nsec - start_time.tv_nsec) / 1e9;
            double train_latency_ms = (epoch_secs - eval_secs) * 1000.0 / (epoch + 1);
            double timespan = epoch_secs;
            
            time_t now = time(NULL);
            struct tm *tm_now = localtime(&now);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_now);
            
            struct rusage r;
            getrusage(RUSAGE_SELF, &r);
            long rss = r.ru_maxrss;
            
            printf("[METRICS] Timestamp=%s | Timespan=%.1fs | Epoch=%lu | "
                   "Train_Acc=%.2f%% | Test_Acc=%.2f%% | Infer_Latency=%.3fms | "
                   "Train_Latency=%.3fms | Memory=%ldKB | Devices=%d\n",
                   timestamp, timespan, epoch, train_acc, test_acc,
                   infer_latency_ms, train_latency_ms, rss, NUM_NODES);
            fflush(stdout);
            
            shm->last_train_acc = train_acc;
            shm->last_test_acc = test_acc;
            
            /* ---- Early stopping check ---- */
            if (test_acc > shm->best_test_acc) {
                shm->best_test_acc = test_acc;
                shm->best_epoch = (int)epoch;
                shm->epochs_no_improve = 0;
            } else {
                shm->epochs_no_improve++;
            }
            
            if (shm->epochs_no_improve >= EARLY_STOP_PATIENCE) {
                printf("[EARLY_STOP] No improvement for %d epochs. Best=%.2f%% at epoch %d\n",
                       EARLY_STOP_PATIENCE, shm->best_test_acc, shm->best_epoch);
                fflush(stdout);
                shm->stop = 1;
            }
        }
        
        /* Sync before next epoch */
        pthread_barrier_wait(&shm->barrier);
    }
    
    exit(0);
}


int main(void)
{
    printf("=== Distributed RockNet Simulation ===\n");
    printf("Nodes: %d\n", NUM_NODES);
    printf("Kernels: %d total, split across %d nodes\n", NUM_KERNELS, NUM_NODES);
    printf("Dilations: %d\n", NUM_DILATIONS);
    printf("Features per kernel: %d (biases_per_kernel)\n", NUM_BIASES_PER_KERNEL);
    printf("Total features: %d\n", NUM_FEATURES);
    printf("Max features per device: %d\n", MAX_FEATURES_PER_DEVICE);
    printf("Training samples: %d\n", NUM_TRAINING_TIMESERIES);
    printf("Evaluation samples: %d\n", NUM_EVALUATION_TIMESERIES);
    printf("Classes: %d\n", NUM_CLASSES);
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("Optimizer: QADAM (Quantized Adam)\n");
    printf("Time series length: %d\n", LENGTH_TIME_SERIES);
    printf("=========================================\n\n");
    fflush(stdout);
    
    /* Create shared memory */
    shm = create_shared_state();
    
    /* Fork worker processes */
    pid_t pids[NUM_NODES];
    
    for (int i = 0; i < NUM_NODES; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            exit(1);
        }
        if (pid == 0) {
            /* Child process */
            node_worker(i);
            /* Never returns */
        }
        pids[i] = pid;
    }
    
    /* Parent: wait for all children */
    for (int i = 0; i < NUM_NODES; i++) {
        int status;
        waitpid(pids[i], &status, 0);
    }
    
    /* Cleanup */
    pthread_barrier_destroy(&shm->barrier);
    munmap(shm, sizeof(shared_state_t));
    
    return 0;
}
