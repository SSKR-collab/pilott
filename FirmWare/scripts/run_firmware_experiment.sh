#!/bin/bash
set -euo pipefail
# =============================================================================
#  FirmWare CNN — Generic Experiment Runner
# =============================================================================
#
#  Runs a single FirmWare CNN experiment for a given config JSON.
#  Automatically determines device layout from the config using python3.
#
#  Features:
#    • Early stopping: monitors tail log; kills run if accuracy stagnates 50 epochs
#    • Metrics logging: timestamp, timespan, train/test accuracy, latency, memory
#    • Structured CSV output for analysis
#    • Automatic timeout (configurable, default 4 hours)
#    • Clean shared memory on exit
#
#  Usage:
#    ./run_firmware_experiment.sh <config.json> [UCR_DATA_ROOT] [TIMEOUT_SEC]
#
#  Example:
#    ./run_firmware_experiment.sh configs/model_config_cricket_x_2L_N9.json \
#                                /path/to/UCR_DATA 14400
# =============================================================================

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.json> [UCR_DATA_ROOT] [TIMEOUT_SEC]"
    exit 1
fi

CONFIG_FILE="$1"
UCR_ROOT="${2:-/mnt/c/Users/GANESH KUMAR/Downloads/Pilot}"
RUN_TIMEOUT="${3:-14400}"  # 4 hours default

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRMWARE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEVICE_BIN="$FIRMWARE_DIR/build/device"

# Resolve config path (relative to FirmWare dir or absolute)
if [[ ! -f "$CONFIG_FILE" ]]; then
    CONFIG_FILE="$FIRMWARE_DIR/$CONFIG_FILE"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [[ ! -f "$DEVICE_BIN" ]]; then
    echo "ERROR: Device binary not found at $DEVICE_BIN"
    echo "Build first: cd $FIRMWARE_DIR && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

export UCR_DATA_ROOT="$UCR_ROOT"

# =============================================================================
#  Parse config JSON with Python to get device layout
# =============================================================================
DEVICE_LAYOUT=$(python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)

dataset = cfg['global']['dataset']
num_classes = cfg['global']['num_classes']
conv_layers = [l for l in cfg['layers'] if l['type'] == 'conv1d']
fc_layer = [l for l in cfg['layers'] if l['type'] == 'fc'][0]

num_conv = len(conv_layers)
workers_per_layer = [l['num_devices'] for l in conv_layers]
total_workers = sum(workers_per_layer)
total_devices = total_workers + 2  # +1 head +1 tail

print(f'{dataset}|{num_classes}|{num_conv}|{\"|\".join(str(w) for w in workers_per_layer)}|{total_devices}')
")

IFS='|' read -r DATASET NUM_CLASSES NUM_CONV_LAYERS WORKERS_STR TOTAL_DEVICES <<< "$DEVICE_LAYOUT"

# Parse workers per layer into array
IFS='|' read -ra WORKERS_PER_LAYER <<< "$WORKERS_STR"

# Derive experiment name from config filename
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .json)
EXPERIMENT_NAME="${CONFIG_BASENAME}"
LOG_DIR="$FIRMWARE_DIR/results/${EXPERIMENT_NAME}"
RESULTS_CSV="$FIRMWARE_DIR/results/${EXPERIMENT_NAME}/metrics.csv"

mkdir -p "$LOG_DIR"

# =============================================================================
#  Cleanup function
# =============================================================================
cleanup() {
    echo ""
    echo "[CLEANUP] Stopping all devices..."
    pkill -P $$ 2>/dev/null || true
    sleep 1
    # Clean up POSIX shared memory and semaphores
    rm -f /dev/shm/ipc_tensor_* /dev/shm/sem.ipc_sem_* 2>/dev/null || true
    rm -f /dev/shm/lw_pilot_* /dev/shm/sem.lw_* 2>/dev/null || true
    # Clean up SysV shared memory
    ipcs -m 2>/dev/null | grep "0x000012" | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[CLEANUP] Complete"
}

trap cleanup EXIT INT TERM

# =============================================================================
#  Print experiment header
# =============================================================================
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  FirmWare CNN Experiment                                           ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Config:     $CONFIG_BASENAME"
echo "║  Dataset:    $DATASET ($NUM_CLASSES classes)"
echo "║  Conv Layers: $NUM_CONV_LAYERS"
echo "║  Total Devices: $TOTAL_DEVICES (1 Head + ${WORKERS_STR//|/+} Workers + 1 Tail)"
echo "║  Timeout:    ${RUN_TIMEOUT}s"
echo "║  Log Dir:    $LOG_DIR/"
echo "║  Started:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Pre-clean shared memory
rm -f /dev/shm/ipc_tensor_* /dev/shm/sem.ipc_sem_* 2>/dev/null || true
rm -f /dev/shm/lw_pilot_* /dev/shm/sem.lw_* 2>/dev/null || true
ipcs -m 2>/dev/null | grep "0x000012" | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null || true

# =============================================================================
#  Initialize CSV results file
# =============================================================================
if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "Timestamp,Timespan_s,Epoch,Train_Acc,Test_Acc,Infer_Latency_ms,Memory_KB,Config,Dataset,Devices,Layers" \
         > "$RESULTS_CSV"
fi

# Record start time
START_TIME=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "$START_TIMESTAMP" > "$LOG_DIR/start_time.txt"

# =============================================================================
#  Launch all devices
# =============================================================================
echo "[LAUNCH] Starting $TOTAL_DEVICES devices..."

DEVICE_ID=0

# --- Head Device (ID 0) ---
echo "  [DEVICE $DEVICE_ID] Head (dataset feeder) ..."
"$DEVICE_BIN" --config="$CONFIG_FILE" --id=$DEVICE_ID --role=head --dataset="$DATASET" \
    > "$LOG_DIR/device_$(printf '%02d' $DEVICE_ID)_head.log" 2>&1 &
HEAD_PID=$!
DEVICE_ID=$((DEVICE_ID + 1))
sleep 1

# --- Worker Devices ---
for LAYER_IDX in $(seq 0 $((NUM_CONV_LAYERS - 1))); do
    NUM_W=${WORKERS_PER_LAYER[$LAYER_IDX]}
    for WORKER_IDX in $(seq 0 $((NUM_W - 1))); do
        echo "  [DEVICE $DEVICE_ID] Layer $LAYER_IDX Worker $WORKER_IDX/$NUM_W ..."
        "$DEVICE_BIN" --config="$CONFIG_FILE" --id=$DEVICE_ID --role=worker \
            --layer-id=$LAYER_IDX --worker-id=$WORKER_IDX --num-workers=$NUM_W \
            > "$LOG_DIR/device_$(printf '%02d' $DEVICE_ID)_L${LAYER_IDX}_W${WORKER_IDX}.log" 2>&1 &
        DEVICE_ID=$((DEVICE_ID + 1))
        sleep 0.3
    done
    sleep 0.5
done

# --- Tail Device ---
TAIL_ID=$DEVICE_ID
echo "  [DEVICE $TAIL_ID] Tail classifier ($NUM_CLASSES classes) ..."
"$DEVICE_BIN" --config="$CONFIG_FILE" --id=$TAIL_ID --role=tail --classes=$NUM_CLASSES \
    > "$LOG_DIR/device_$(printf '%02d' $TAIL_ID)_tail.log" 2>&1 &
TAIL_PID=$!

echo ""
echo "[LAUNCH] All $TOTAL_DEVICES devices started (Head PID: $HEAD_PID, Tail PID: $TAIL_PID)"
echo ""

# =============================================================================
#  Timeout killer
# =============================================================================
(
    sleep "$RUN_TIMEOUT"
    echo ""
    echo "[TIMEOUT] Run duration ${RUN_TIMEOUT}s reached."
    pkill -P $$ 2>/dev/null || true
    pkill -f "build/device" 2>/dev/null || true
) &
TIMEOUT_PID=$!

# =============================================================================
#  Early stopping monitor (50 epochs with no improvement)
#  Monitors the tail log for [METRICS] lines, extracts Test_Acc,
#  kills run if 50 consecutive epochs show no improvement.
# =============================================================================
EARLY_STOP_PATIENCE=50
TAIL_LOG="$LOG_DIR/device_$(printf '%02d' $TAIL_ID)_tail.log"

(
    sleep 10  # Wait for initial startup
    BEST_ACC=0.0
    NO_IMPROVE_COUNT=0
    PREV_EPOCH=0

    while true; do
        sleep 5  # Check every 5 seconds
        
        # Get latest metrics line
        LATEST=$(grep "\[METRICS\]" "$TAIL_LOG" 2>/dev/null | tail -1 || true)
        if [[ -z "$LATEST" ]]; then
            continue
        fi

        # Extract epoch and test accuracy
        EPOCH=$(echo "$LATEST" | grep -oP 'Epoch=\K[0-9]+' || echo "0")
        TEST_ACC=$(echo "$LATEST" | grep -oP 'Test_Acc=\K[0-9.]+' || echo "0")
        
        if [[ "$EPOCH" == "$PREV_EPOCH" ]]; then
            continue  # Same epoch, skip
        fi
        PREV_EPOCH="$EPOCH"

        # Parse test accuracy and CSV line
        TIMESTAMP=$(echo "$LATEST" | grep -oP 'Timestamp=\K[0-9-]+ [0-9:]+' || echo "")
        TIMESPAN=$(echo "$LATEST" | grep -oP 'Timespan=\K[0-9.]+' || echo "0")
        TRAIN_ACC=$(echo "$LATEST" | grep -oP 'Train_Acc=\K[0-9.]+' || echo "0")
        INFER_LAT=$(echo "$LATEST" | grep -oP 'Infer_Latency=\K[0-9.]+' || echo "0")
        MEMORY=$(echo "$LATEST" | grep -oP 'Memory=\K[0-9]+' || echo "0")

        # Append to CSV
        echo "$TIMESTAMP,$TIMESPAN,$EPOCH,$TRAIN_ACC,$TEST_ACC,$INFER_LAT,$MEMORY,$CONFIG_BASENAME,$DATASET,$TOTAL_DEVICES,$NUM_CONV_LAYERS" \
            >> "$RESULTS_CSV"

        # Check early stopping
        IMPROVED=$(python3 -c "print(1 if float('$TEST_ACC') > float('$BEST_ACC') else 0)" 2>/dev/null || echo "0")
        if [[ "$IMPROVED" == "1" ]]; then
            BEST_ACC="$TEST_ACC"
            NO_IMPROVE_COUNT=0
        else
            NO_IMPROVE_COUNT=$((NO_IMPROVE_COUNT + 1))
        fi

        if [[ $NO_IMPROVE_COUNT -ge $EARLY_STOP_PATIENCE ]]; then
            echo ""
            echo "[EARLY_STOP] No improvement for $EARLY_STOP_PATIENCE epochs (best=$BEST_ACC%)"
            echo "[EARLY_STOP] Terminating experiment at epoch $EPOCH"
            
            # Record final results
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
            echo "" >> "$LOG_DIR/summary.txt"
            echo "EARLY_STOPPED=true" >> "$LOG_DIR/summary.txt"
            echo "BEST_TEST_ACC=$BEST_ACC" >> "$LOG_DIR/summary.txt"
            echo "FINAL_EPOCH=$EPOCH" >> "$LOG_DIR/summary.txt"
            echo "DURATION_SEC=$DURATION" >> "$LOG_DIR/summary.txt"
            
            pkill -P $$ 2>/dev/null || true
            pkill -f "build/device" 2>/dev/null || true
            exit 0
        fi
    done
) &
MONITOR_PID=$!

# =============================================================================
#  Live monitoring — show METRICS lines from tail
# =============================================================================
echo "[MONITOR] Watching tail log for metrics (Ctrl+C to stop) ..."
echo "─────────────────────────────────────────────────────────────"

tail -f "$TAIL_LOG" 2>/dev/null | grep --line-buffered "\[METRICS\]\|EARLY_STOP\|FATAL" &
LIVE_PID=$!

# Wait for head process
wait $HEAD_PID 2>/dev/null || true
HEAD_EXIT=$?

# Kill monitors
kill $LIVE_PID 2>/dev/null || true
kill $MONITOR_PID 2>/dev/null || true
kill $TIMEOUT_PID 2>/dev/null || true

# =============================================================================
#  Finalize
# =============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Extract final metrics
FINAL_METRICS=$(grep "\[METRICS\]" "$TAIL_LOG" 2>/dev/null | tail -1 || echo "")
FINAL_ACC=$(echo "$FINAL_METRICS" | grep -oP 'Test_Acc=\K[0-9.]+' || echo "N/A")
FINAL_TRAIN_ACC=$(echo "$FINAL_METRICS" | grep -oP 'Train_Acc=\K[0-9.]+' || echo "N/A")
TOTAL_EPOCHS=$(grep -c "\[METRICS\]" "$TAIL_LOG" 2>/dev/null || echo "0")
BEST_ACC_LINE=$(grep "\[EARLY_STOP\] New best" "$TAIL_LOG" 2>/dev/null | tail -1 || echo "")
BEST_ACC=$(echo "$BEST_ACC_LINE" | grep -oP '[0-9.]+(?=%)' | head -1 || echo "$FINAL_ACC")

# Write summary
cat > "$LOG_DIR/summary.txt" <<EOF
═══════════════════════════════════════════════════
  Experiment Summary: $EXPERIMENT_NAME
═══════════════════════════════════════════════════
Config:       $CONFIG_BASENAME
Dataset:      $DATASET ($NUM_CLASSES classes)
Architecture: ${NUM_CONV_LAYERS}-layer CNN, $TOTAL_DEVICES devices
Start:        $START_TIMESTAMP
End:          $END_TIMESTAMP
Duration:     ${DURATION}s ($(printf '%dh %dm %ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60))))
Epochs:       $TOTAL_EPOCHS
Best Test Acc: ${BEST_ACC}%
Final Test Acc: ${FINAL_ACC}%
Final Train Acc: ${FINAL_TRAIN_ACC}%
Head Exit Code: $HEAD_EXIT
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Experiment Complete                                               ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Duration:       ${DURATION}s"
echo "║  Total Epochs:   $TOTAL_EPOCHS"
echo "║  Best Test Acc:  ${BEST_ACC}%"
echo "║  Final Test Acc: ${FINAL_ACC}%"
echo "║  Results CSV:    $RESULTS_CSV"
echo "║  Summary:        $LOG_DIR/summary.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"

wait 2>/dev/null || true
exit 0
