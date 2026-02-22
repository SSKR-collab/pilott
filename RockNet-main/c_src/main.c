#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include "conv.h"

#include "rocket_config.h"
#include "linear_classifier.h"

static float cummulative[NUM_CLASSES];

static long get_rss_kb(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss;  /* KB on Linux */
}

void train()
{
    printf("Starting training\n");
    float accuracy_filtered = 0;
    const float gamma = 0.01;

    struct timespec start_time, epoch_end, eval_start, eval_end;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    for (uint64_t epoch = 0; epoch < 100000; epoch++) {
        /* ---- Training pass (with accuracy tracking) ---- */
        int train_correct = 0;
        for (uint32_t batch_idx = 0; batch_idx < NUM_TRAINING_TIMESERIES; batch_idx++) {
            classify_part(get_training_timeseries()[batch_idx], cummulative);

            /* Track train accuracy */
            uint8_t train_pred = get_max_idx(cummulative, NUM_CLASSES);
            if (train_pred == get_training_labels()[batch_idx]) {
                train_correct++;
            }

            calculate_and_accumulate_gradient(cummulative, get_training_labels()[batch_idx]);

            if (batch_idx % BATCH_SIZE == BATCH_SIZE-1) {
                update_weights();
            }
        }
        if (NUM_TRAINING_TIMESERIES % BATCH_SIZE != 0) {
            update_weights();
        }
        float train_acc = (float)train_correct / NUM_TRAINING_TIMESERIES * 100.0f;

        /* ---- Evaluation pass (timed for inference latency) ---- */
        clock_gettime(CLOCK_MONOTONIC, &eval_start);
        float acc = 0;
        for (uint32_t batch_idx = 0; batch_idx < NUM_EVALUATION_TIMESERIES; batch_idx++) {
            classify_part(get_evaluation_timeseries()[batch_idx], cummulative);

            uint8_t pred_idx = get_max_idx(cummulative, NUM_CLASSES);

            if (pred_idx == get_evaluation_labels()[batch_idx]) {
                acc++;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &eval_end);

        float test_acc = acc / NUM_EVALUATION_TIMESERIES * 100.0f;
        double eval_secs = (eval_end.tv_sec - eval_start.tv_sec) + (eval_end.tv_nsec - eval_start.tv_nsec) / 1e9;
        double infer_latency_ms = eval_secs / NUM_EVALUATION_TIMESERIES * 1000.0;

        clock_gettime(CLOCK_MONOTONIC, &epoch_end);
        double timespan = (epoch_end.tv_sec - start_time.tv_sec) + (epoch_end.tv_nsec - start_time.tv_nsec) / 1e9;

        time_t now = time(NULL);
        struct tm *t = localtime(&now);
        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", t);

        long rss = get_rss_kb();

        printf("[METRICS] Timestamp=%s | Timespan=%.1fs | Epoch=%lu | Train_Acc=%.2f%% | Test_Acc=%.2f%% | Infer_Latency=%.3fms | Memory=%ldKB\n",
               timestamp, timespan, epoch, train_acc, test_acc, infer_latency_ms, rss);
    }
}

int main()
{
    init_rocket();

    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%f\n", get_biases()[i]);
    }

    init_linear_classifier(0);
    train();
}