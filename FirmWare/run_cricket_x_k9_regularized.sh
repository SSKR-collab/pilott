#!/bin/bash
set -e

# ============================================================
# Run Firmware CNN for Cricket_X with kernel_size=9 + Regularization
# Config: model_config_nrf52840_realistic_k9.json
# Regularization: L2 weight decay (λ=0.001), Dropout (0.4), Early Stopping (patience=30)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_BIN="$SCRIPT_DIR/build/device"
MODEL_CONFIG="$SCRIPT_DIR/configs/model_config_nrf52840_realistic_k9.json"
DATASET="Cricket_X"
NUM_CLASSES=12
LOG_DIR="$SCRIPT_DIR/logs/cricket_x_k9_regularized"
RUN_DURATION=3600  # 1 hour

export UCR_DATA_ROOT="/mnt/c/Users/GANESH KUMAR/Downloads/LLBPP"

mkdir -p "$LOG_DIR"

cleanup() {
    echo ""
    echo "Cleaning up..."
    pkill -P $$ 2>/dev/null || true
    rm -f /dev/shm/lw_pilot_* /dev/shm/sem.lw_* /dev/shm/ipc_* /dev/shm/sem.ipc_* 2>/dev/null || true
    ipcs -m 2>/dev/null | grep "0x000012" | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

echo "=========================================="
echo "Firmware CNN - Cricket_X (k=9) + REGULARIZATION"
echo "=========================================="
echo "Config: $MODEL_CONFIG"
echo "Dataset: $DATASET ($NUM_CLASSES classes)"
echo "Regularization:"
echo "  L2 Weight Decay: λ=0.001"
echo "  Dropout: 0.4 (before FC layer)"
echo "  Early Stopping: patience=30 epochs"
echo "Architecture: 7 devices"
echo "  Layer 0: Conv1D 1→32ch, kernel=9, stride=1, pad=2 → 296"
echo "  Layer 1: Conv1D 32→48ch, kernel=5, stride=2, pad=2 → 148"  
echo "  Layer 2: FC (avg+max pool) 96→12 (with dropout)"
echo "Run duration: ${RUN_DURATION}s (1 hour)"
echo "Log directory: $LOG_DIR/"
echo "Started at: $(date)"
echo "=========================================="
echo ""

rm -f /dev/shm/lw_pilot_* /dev/shm/sem.lw_* /dev/shm/ipc_* /dev/shm/sem.ipc_* 2>/dev/null || true

START_TIME=$(date +%s)
echo "$START_TIME" > "$LOG_DIR/start_time.txt"

echo "Starting devices..."

# Head
echo "[DEVICE 0] Head..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=0 --role=head --dataset="$DATASET" \
    > "$LOG_DIR/device_00_head.log" 2>&1 &
HEAD_PID=$!
sleep 1

# Layer 0 Workers (2)
echo "[DEVICE 1] Layer 0 Worker 0/2..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=1 --role=worker --layer-id=0 --worker-id=0 --num-workers=2 \
    > "$LOG_DIR/device_01_layer0_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 2] Layer 0 Worker 1/2..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=2 --role=worker --layer-id=0 --worker-id=1 --num-workers=2 \
    > "$LOG_DIR/device_02_layer0_w1.log" 2>&1 &
sleep 0.5

# Layer 1 Workers (3)
echo "[DEVICE 3] Layer 1 Worker 0/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=3 --role=worker --layer-id=1 --worker-id=0 --num-workers=3 \
    > "$LOG_DIR/device_03_layer1_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 4] Layer 1 Worker 1/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=4 --role=worker --layer-id=1 --worker-id=1 --num-workers=3 \
    > "$LOG_DIR/device_04_layer1_w1.log" 2>&1 &
sleep 0.3

echo "[DEVICE 5] Layer 1 Worker 2/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=5 --role=worker --layer-id=1 --worker-id=2 --num-workers=3 \
    > "$LOG_DIR/device_05_layer1_w2.log" 2>&1 &
sleep 0.5

# Tail
echo "[DEVICE 6] Tail classifier..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=6 --role=tail --classes=$NUM_CLASSES \
    > "$LOG_DIR/device_06_tail.log" 2>&1 &
TAIL_PID=$!
sleep 1

echo ""
echo "All 7 devices launched!"
echo "Head PID: $HEAD_PID, Tail PID: $TAIL_PID"

# Auto-kill
(
    sleep $RUN_DURATION
    echo "Time limit reached. Stopping..."
    pkill -f "./build/device" 2>/dev/null || true
) &
KILLER_PID=$!
echo "Auto-kill timer: ${RUN_DURATION}s (PID: $KILLER_PID)"
echo ""

echo "Monitoring for METRICS and EARLY_STOP events..."
echo "=========================================="

tail -f "$LOG_DIR/device_06_tail.log" 2>/dev/null | grep --line-buffered "\[METRICS\]\|EARLY_STOP\|Error\|FATAL" &
MONITOR_PID=$!

wait $HEAD_PID 2>/dev/null
HEAD_EXIT=$?

kill $MONITOR_PID 2>/dev/null || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "$END_TIME" > "$LOG_DIR/end_time.txt"

echo ""
echo "=========================================="
echo "Run completed! Duration: ${ELAPSED}s"
echo "=========================================="
echo ""

wait 2>/dev/null || true

echo "Quick results:"
grep "\[METRICS\]" "$LOG_DIR/device_06_tail.log" 2>/dev/null | tail -5 || echo "  No METRICS"
echo ""
echo "Early stopping events:"
grep "EARLY_STOP" "$LOG_DIR/device_06_tail.log" 2>/dev/null | tail -5 || echo "  None"
echo ""
echo "Total epochs:"
grep -c "\[METRICS\]" "$LOG_DIR/device_06_tail.log" 2>/dev/null || echo "  0"
