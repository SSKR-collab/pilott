#!/bin/bash
set -e

# ============================================================
# Run Firmware CNN for Cricket_X with kernel_size=9 config
# Config: model_config_nrf52840_realistic_k9.json
# Architecture: 7 devices (1 Head + 2 L0 Workers + 3 L1 Workers + 1 Tail)
# Dataset: Cricket_X (12 classes, 300 time steps)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_BIN="$SCRIPT_DIR/build/device"
MODEL_CONFIG="$SCRIPT_DIR/configs/model_config_nrf52840_realistic_k9.json"
DATASET="Cricket_X"
NUM_CLASSES=12
LOG_DIR="$SCRIPT_DIR/logs/cricket_x_k9"
RUN_DURATION=3600  # 1 hour

# Set UCR data root
export UCR_DATA_ROOT="/mnt/c/Users/GANESH KUMAR/Downloads/LLBPP"

# Create log directory
mkdir -p "$LOG_DIR"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Kill all background device processes
    pkill -P $$ 2>/dev/null || true
    # Clean up shared memory segments
    rm -f /dev/shm/lw_pilot_* /dev/shm/sem.lw_* /dev/shm/ipc_* /dev/shm/sem.ipc_* 2>/dev/null || true
    ipcs -m 2>/dev/null | grep "0x000012" | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

echo "=========================================="
echo "Firmware CNN - Cricket_X (kernel_size=9)"
echo "=========================================="
echo "Config: $MODEL_CONFIG"
echo "Dataset: $DATASET ($NUM_CLASSES classes)"
echo "Architecture: 7 devices"
echo "  Layer 0: Conv1D 1→32ch, kernel=9, stride=1, pad=2 → output=296"
echo "  Layer 1: Conv1D 32→48ch, kernel=5, stride=2, pad=2 → output=148"  
echo "  Layer 2: FC (avg+max pool) 96→12"
echo "  Workers: 2 (Layer 0) + 3 (Layer 1) = 5"
echo "Run duration: ${RUN_DURATION}s (1 hour)"
echo "Log directory: $LOG_DIR/"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Clean up shared memory from previous runs
rm -f /dev/shm/lw_pilot_* /dev/shm/sem.lw_* /dev/shm/ipc_* /dev/shm/sem.ipc_* 2>/dev/null || true

START_TIME=$(date +%s)
echo "$START_TIME" > "$LOG_DIR/start_time.txt"

echo "Starting devices..."
echo ""

# Launch Head Device (ID 0)
echo "[DEVICE 0] Starting Head device..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=0 --role=head --dataset="$DATASET" \
    > "$LOG_DIR/device_00_head.log" 2>&1 &
HEAD_PID=$!
sleep 1

# Launch Layer 0 Workers (2 devices: 16 channels each = 32 total)
echo "[DEVICE 1] Starting Layer 0 Worker 0/2..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=1 --role=worker --layer-id=0 --worker-id=0 --num-workers=2 \
    > "$LOG_DIR/device_01_layer0_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 2] Starting Layer 0 Worker 1/2..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=2 --role=worker --layer-id=0 --worker-id=1 --num-workers=2 \
    > "$LOG_DIR/device_02_layer0_w1.log" 2>&1 &
sleep 0.5

# Launch Layer 1 Workers (3 devices: 16 channels each = 48 total)
echo "[DEVICE 3] Starting Layer 1 Worker 0/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=3 --role=worker --layer-id=1 --worker-id=0 --num-workers=3 \
    > "$LOG_DIR/device_03_layer1_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 4] Starting Layer 1 Worker 1/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=4 --role=worker --layer-id=1 --worker-id=1 --num-workers=3 \
    > "$LOG_DIR/device_04_layer1_w1.log" 2>&1 &
sleep 0.3

echo "[DEVICE 5] Starting Layer 1 Worker 2/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=5 --role=worker --layer-id=1 --worker-id=2 --num-workers=3 \
    > "$LOG_DIR/device_05_layer1_w2.log" 2>&1 &
sleep 0.5

# Launch Tail Device (ID 6)
echo "[DEVICE 6] Starting Tail classifier (${NUM_CLASSES} classes)..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=6 --role=tail --classes=$NUM_CLASSES \
    > "$LOG_DIR/device_06_tail.log" 2>&1 &
TAIL_PID=$!
sleep 1

echo ""
echo "All 7 devices launched!"
echo "Head PID: $HEAD_PID, Tail PID: $TAIL_PID"
echo "Logs: $LOG_DIR/"
echo ""

# Set up auto-kill after RUN_DURATION seconds
(
    sleep $RUN_DURATION
    echo ""
    echo "=========================================="
    echo "Time limit ($RUN_DURATION seconds) reached!"
    echo "Stopping all devices at $(date)..."
    echo "=========================================="
    pkill -P $HEAD_PID 2>/dev/null || true
    kill $HEAD_PID 2>/dev/null || true
    pkill -f "./build/device" 2>/dev/null || true
) &
KILLER_PID=$!
echo "Auto-kill timer set for ${RUN_DURATION}s (PID: $KILLER_PID)"
echo ""

echo "Monitoring tail device for metrics (Ctrl+C to stop early)..."
echo "=========================================="
echo ""

# Monitor the tail log for [METRICS] lines in real time
tail -f "$LOG_DIR/device_06_tail.log" 2>/dev/null | grep --line-buffered "\[METRICS\]\|Accuracy\|completed\|Error\|FATAL" &
MONITOR_PID=$!

# Wait for head device to complete (or be killed by timeout)
wait $HEAD_PID 2>/dev/null
HEAD_EXIT=$?

# Kill monitor
kill $MONITOR_PID 2>/dev/null || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "$END_TIME" > "$LOG_DIR/end_time.txt"

echo ""
echo "=========================================="
echo "Run completed!"
echo "Duration: ${ELAPSED}s"
echo "Head exit code: $HEAD_EXIT"
echo "=========================================="
echo ""

# Wait for remaining devices
echo "Waiting for remaining devices to finish..."
wait 2>/dev/null || true

echo ""
echo "Log files:"
ls -la "$LOG_DIR/"
echo ""
echo "Quick results (METRICS lines from tail):"
grep "\[METRICS\]" "$LOG_DIR/device_06_tail.log" 2>/dev/null | tail -5 || echo "  No METRICS lines found yet"
echo ""
echo "Total epochs completed:"
grep -c "\[METRICS\]" "$LOG_DIR/device_06_tail.log" 2>/dev/null || echo "  0"
