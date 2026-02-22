#!/bin/bash

# Distributed CNN Test Script
# Launches 11-device pipeline: 1 Head + 9 Workers (2+3+4) + 1 Tail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_BIN="$SCRIPT_DIR/build/device"
MODEL_CONFIG="$SCRIPT_DIR/configs/model_config_cricket_x.json"
DATASET="Cricket_X"
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Cleanup function
cleanup() {
    echo "Cleaning up shared memory segments..."
    ipcs -m | grep "0x000012" | awk '{print $2}' | xargs -r ipcrm -m 2>/dev/null
    echo "Terminating all device processes..."
    pkill -P $$ 2>/dev/null
    wait 2>/dev/null
    echo "Cleanup complete"
}

trap cleanup EXIT INT TERM

echo "=========================================="
echo "Distributed CNN Pipeline Test"
echo "=========================================="
echo "Model: nRF52840_UniformCNN v2.0"
echo "Dataset: Cricket_X (12 classes)"
echo "Architecture: 7 devices (1 Head + 5 Workers + 1 Tail)"
echo "  Layer 1: 2 workers (32 channels)"
echo "  Layer 2: 3 workers (48 channels)"
echo "=========================================="
echo ""

# Clean up any existing shared memory from previous runs
cleanup > /dev/null 2>&1

echo "Starting devices..."
echo ""

# Launch Head Device (ID 0)
echo "[DEVICE 0] Starting Head device..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=0 --role=head --dataset="$DATASET" \
    > "$LOG_DIR/device_00_head.log" 2>&1 &
HEAD_PID=$!
sleep 1

# Launch Layer 1 Workers (2 devices)
echo "[DEVICE 1] Starting Layer 1 Worker 0/2..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=1 --role=worker --layer-id=0 --worker-id=0 --num-workers=2 \
    > "$LOG_DIR/device_01_layer0_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 2] Starting Layer 1 Worker 1/2..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=2 --role=worker --layer-id=0 --worker-id=1 --num-workers=2 \
    > "$LOG_DIR/device_02_layer0_w1.log" 2>&1 &
sleep 0.5

# Launch Layer 2 Workers (3 devices)
echo "[DEVICE 3] Starting Layer 2 Worker 0/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=3 --role=worker --layer-id=1 --worker-id=0 --num-workers=3 \
    > "$LOG_DIR/device_03_layer1_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 4] Starting Layer 2 Worker 1/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=4 --role=worker --layer-id=1 --worker-id=1 --num-workers=3 \
    > "$LOG_DIR/device_04_layer1_w1.log" 2>&1 &
sleep 0.3

echo "[DEVICE 5] Starting Layer 2 Worker 2/3..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=5 --role=worker --layer-id=1 --worker-id=2 --num-workers=3 \
    > "$LOG_DIR/device_05_layer1_w2.log" 2>&1 &
sleep 0.5

# Launch Tail Device (ID 6) - connects directly to Layer 2 output
echo "[DEVICE 6] Starting Tail classifier..."
"$DEVICE_BIN" --config="$MODEL_CONFIG" --id=6 --role=tail --classes=12 \
    > "$LOG_DIR/device_06_tail.log" 2>&1 &
TAIL_PID=$!
sleep 1

echo ""
echo "All devices launched!"
echo "Logs are being written to: $LOG_DIR/"
echo ""
echo "Monitoring head device progress (Ctrl+C to stop)..."
echo "=========================================="
echo ""

# Monitor head device (it drives the pipeline)
tail -f "$LOG_DIR/device_00_head.log" &
TAIL_LOG_PID=$!

# Wait for head device to complete
wait $HEAD_PID
HEAD_EXIT=$?

# Kill the tail monitoring
kill $TAIL_LOG_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Head device completed with exit code: $HEAD_EXIT"
echo ""
echo "Waiting for all devices to finish..."
wait

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Log files available in: $LOG_DIR/"
echo ""
echo "Quick results:"
echo "  Head device: $LOG_DIR/device_00_head.log"
echo "  Tail device: $LOG_DIR/device_06_tail.log"
echo ""
echo "To view tail classifier statistics:"
echo "  grep 'Final Tail\|Accuracy\|Loss' $LOG_DIR/device_06_tail.log"
echo ""
