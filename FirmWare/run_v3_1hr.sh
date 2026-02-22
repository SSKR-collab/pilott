#!/bin/bash
set -e

DEVICE_BIN="./build/device"
LOG_DIR="./logs/ecg5000_v3_1hr"
mkdir -p "$LOG_DIR"

echo "Starting FirmWare CNN v3 (2-layer) at $(date)"
date +%s > /tmp/firmware_v3_start.txt

# Clean shared memory
rm -f /dev/shm/lw_pilot_* /dev/shm/sem.lw_* /dev/shm/ipc_* /dev/shm/sem.ipc_* 2>/dev/null

# Launch Head (Device 0)
echo "[DEVICE 0] Head feeder"
"$DEVICE_BIN" --id=0 --role=head > "$LOG_DIR/device_00_head.log" 2>&1 &
HEAD_PID=$!
echo $HEAD_PID > /tmp/firmware_v3_head_pid.txt
sleep 1

# Launch Layer 1 Workers (2 devices)
echo "[DEVICE 1] Layer 1 Worker 0/2"
"$DEVICE_BIN" --id=1 --role=worker --layer-id=0 --worker-id=0 --num-workers=2 > "$LOG_DIR/device_01_layer0_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 2] Layer 1 Worker 1/2"
"$DEVICE_BIN" --id=2 --role=worker --layer-id=0 --worker-id=1 --num-workers=2 > "$LOG_DIR/device_02_layer0_w1.log" 2>&1 &
sleep 0.5

# Launch Layer 2 Workers (3 devices)
echo "[DEVICE 3] Layer 2 Worker 0/3"
"$DEVICE_BIN" --id=3 --role=worker --layer-id=1 --worker-id=0 --num-workers=3 > "$LOG_DIR/device_03_layer1_w0.log" 2>&1 &
sleep 0.3

echo "[DEVICE 4] Layer 2 Worker 1/3"
"$DEVICE_BIN" --id=4 --role=worker --layer-id=1 --worker-id=1 --num-workers=3 > "$LOG_DIR/device_04_layer1_w1.log" 2>&1 &
sleep 0.3

echo "[DEVICE 5] Layer 2 Worker 2/3"
"$DEVICE_BIN" --id=5 --role=worker --layer-id=1 --worker-id=2 --num-workers=3 > "$LOG_DIR/device_05_layer1_w2.log" 2>&1 &
sleep 0.5

# Launch Tail (Device 6)
echo "[DEVICE 6] Tail classifier (5 classes)"
"$DEVICE_BIN" --id=6 --role=tail --classes=5 > "$LOG_DIR/device_06_tail.log" 2>&1 &
TAIL_PID=$!
echo $TAIL_PID > /tmp/firmware_v3_tail_pid.txt
sleep 1

echo ""
echo "All 7 devices launched!"
echo "Head PID: $HEAD_PID, Tail PID: $TAIL_PID"
echo "Logs: $LOG_DIR/"
echo "Will kill at: $(date -d @$(($(cat /tmp/firmware_v3_start.txt) + 3600)))"

# Set up auto-kill after 3600 seconds
(
    sleep 3600
    echo "1-hour timeout reached. Killing all devices..."
    pkill -P $HEAD_PID 2>/dev/null
    kill $HEAD_PID 2>/dev/null
    pkill -f "./build/device" 2>/dev/null
    echo "All devices killed at $(date)"
) &
KILLER_PID=$!
echo "Auto-kill timer PID: $KILLER_PID"
echo $KILLER_PID > /tmp/firmware_v3_killer_pid.txt

echo "FirmWare CNN v3 is running in background. Check logs with:"
echo "  tail -f $LOG_DIR/device_06_tail.log"
