#!/bin/bash
set -euo pipefail
# =============================================================================
#  RockNet Distributed — Single Experiment Runner
# =============================================================================
#
#  Runs a single RockNet distributed experiment for a given dataset and N nodes.
#  Steps:
#    1. Generate rocket_config.{h,c} via generate_distributed_config.py
#    2. Compile the distributed_sim binary with -DNUM_NODES=N
#    3. Run the simulation
#    4. Collect metrics (early stopping via C code, 50-epoch patience)
#
#  Features:
#    • Early stopping built into the C code (50 epochs no improvement)
#    • Metrics: timestamp, timespan, train/test accuracy, infer/train latency, memory
#    • Automatic timeout
#    • Structured CSV output
#
#  Usage:
#    ./run_rocknet_experiment.sh [-p] <dataset_name> <num_nodes> [UCR_DATA_ROOT] [TIMEOUT]
#
#  Options:
#    -p   Enable processing constraint (64 MHz simulated clock)
#         Memory constraint (256 KB/device) is ALWAYS active.
#
#  Example:
#    ./run_rocknet_experiment.sh Cricket_X 9 /path/to/UCR_DATA 7200
#    ./run_rocknet_experiment.sh -p Cricket_X 9
# =============================================================================

PROC_FLAG=""
if [[ "${1:-}" == "-p" ]]; then
    PROC_FLAG="-p"
    shift
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 [-p] <dataset_name> <num_nodes> [UCR_DATA_ROOT] [TIMEOUT_SEC]"
    echo "  -p: Enable processing constraint (64 MHz)"
    echo "  dataset_name: Cricket_X, FaceAll, ECG5000, Coffee"
    echo "  num_nodes:    7, 9, 11, 13, 15"
    exit 1
fi

DATASET="$1"
NUM_NODES="$2"
UCR_ROOT="${3:-/mnt/c/Users/GANESH KUMAR/Downloads/Pilot}"
RUN_TIMEOUT="${4:-7200}"  # 2 hours default

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCKNET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_SIM_DIR="$ROCKNET_DIR/c_src/distributed_sim"

export UCR_DATA_ROOT="$UCR_ROOT"

EXPERIMENT_NAME="rocknet_${DATASET}_N${NUM_NODES}"
LOG_DIR="$ROCKNET_DIR/results/${EXPERIMENT_NAME}"
RESULTS_CSV="$LOG_DIR/metrics.csv"
BINARY="$DIST_SIM_DIR/distributed_rocknet_${DATASET}_N${NUM_NODES}"

mkdir -p "$LOG_DIR"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  RockNet Distributed Experiment                                    ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Dataset:     $DATASET"
echo "║  Num Nodes:   $NUM_NODES"
echo "║  Memory:      256 KB/device (always active)"
if [[ -n "$PROC_FLAG" ]]; then
    echo "║  Processing:  64 MHz constraint ENABLED"
else
    echo "║  Processing:  unconstrained"
fi
echo "║  Timeout:     ${RUN_TIMEOUT}s"
echo "║  Log Dir:     $LOG_DIR/"
echo "║  UCR Data:    $UCR_ROOT"
echo "║  Started:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
#  Step 1: Generate config
# =============================================================================
echo "[STEP 1] Generating rocket_config for $DATASET with $NUM_NODES nodes ..."

# Check if python dependencies are available
if ! python3 -c "import numpy, pandas, sympy" 2>/dev/null; then
    echo "ERROR: Python dependencies missing. Install: pip3 install numpy pandas sympy"
    exit 1
fi

cd "$ROCKNET_DIR"
python3 generate_distributed_config.py "$DATASET" "$NUM_NODES" 2>&1 | tee "$LOG_DIR/config_gen.log"

if [[ ! -f "$DIST_SIM_DIR/rocket_config.h" ]] || [[ ! -f "$DIST_SIM_DIR/rocket_config.c" ]]; then
    echo "ERROR: Config generation failed — rocket_config.{h,c} not found"
    exit 1
fi
echo "  ✓ Config generated"
echo ""

# =============================================================================
#  Step 2: Compile
# =============================================================================
echo "[STEP 2] Compiling distributed_sim (NUM_NODES=$NUM_NODES) ..."

cd "$DIST_SIM_DIR"

# Build command — include all required source files
gcc -O2 -DNUM_NODES="$NUM_NODES" \
    -o "$BINARY" \
    main.c \
    rocket_config.c \
    conv.c \
    linear_classifier.c \
    dynamic_tree_quantization.c \
    -I. \
    -lm -lrt -lpthread \
    2>&1 | tee "$LOG_DIR/compile.log"

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: Compilation failed — binary not found at $BINARY"
    exit 1
fi
echo "  ✓ Compiled: $BINARY"
echo ""

# =============================================================================
#  Step 3: Run experiment
# =============================================================================
echo "[STEP 3] Running distributed RockNet ($NUM_NODES nodes) ..."

# Initialize CSV
echo "Timestamp,Timespan_s,Epoch,Train_Acc,Test_Acc,Infer_Latency_ms,Train_Latency_ms,Memory_KB,Devices,Dataset" \
     > "$RESULTS_CSV"

START_TIME=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Run with timeout, collecting output
if [[ -n "$PROC_FLAG" ]]; then
    export PROC_CONSTRAINT=1
fi
timeout "$RUN_TIMEOUT" "$BINARY" 2>&1 | tee "$LOG_DIR/run.log" &
RUN_PID=$!

# Background process to parse metrics lines and write to CSV
(
    sleep 3  # Wait for startup
    while kill -0 $RUN_PID 2>/dev/null; do
        sleep 2
        
        # Extract all METRICS lines and convert to CSV
        grep "\[METRICS\]" "$LOG_DIR/run.log" 2>/dev/null | while IFS= read -r line; do
            TS=$(echo "$line" | grep -oP 'Timestamp=\K[0-9-]+ [0-9:]+' || echo "")
            TSPAN=$(echo "$line" | grep -oP 'Timespan=\K[0-9.]+' || echo "0")
            EP=$(echo "$line" | grep -oP 'Epoch=\K[0-9]+' || echo "0")
            TRAIN=$(echo "$line" | grep -oP 'Train_Acc=\K[0-9.]+' || echo "0")
            TEST=$(echo "$line" | grep -oP 'Test_Acc=\K[0-9.]+' || echo "0")
            INFER=$(echo "$line" | grep -oP 'Infer_Latency=\K[0-9.]+' || echo "0")
            TLAT=$(echo "$line" | grep -oP 'Train_Latency=\K[0-9.]+' || echo "0")
            MEM=$(echo "$line" | grep -oP 'Memory=\K[0-9]+' || echo "0")
            DEV=$(echo "$line" | grep -oP 'Devices=\K[0-9]+' || echo "$NUM_NODES")
            
            echo "$TS,$TSPAN,$EP,$TRAIN,$TEST,$INFER,$TLAT,$MEM,$DEV,$DATASET"
        done > "$LOG_DIR/metrics_tmp.csv" 2>/dev/null
        
        # Replace CSV data section (keep header)
        if [[ -f "$LOG_DIR/metrics_tmp.csv" ]] && [[ -s "$LOG_DIR/metrics_tmp.csv" ]]; then
            head -1 "$RESULTS_CSV" > "$RESULTS_CSV.tmp"
            cat "$LOG_DIR/metrics_tmp.csv" >> "$RESULTS_CSV.tmp"
            mv "$RESULTS_CSV.tmp" "$RESULTS_CSV"
        fi
    done
) &
CSV_PID=$!

# Wait for run to complete
wait $RUN_PID 2>/dev/null
RUN_EXIT=$?

kill $CSV_PID 2>/dev/null || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Final CSV extraction
grep "\[METRICS\]" "$LOG_DIR/run.log" 2>/dev/null | while IFS= read -r line; do
    TS=$(echo "$line" | grep -oP 'Timestamp=\K[0-9-]+ [0-9:]+' || echo "")
    TSPAN=$(echo "$line" | grep -oP 'Timespan=\K[0-9.]+' || echo "0")
    EP=$(echo "$line" | grep -oP 'Epoch=\K[0-9]+' || echo "0")
    TRAIN=$(echo "$line" | grep -oP 'Train_Acc=\K[0-9.]+' || echo "0")
    TEST=$(echo "$line" | grep -oP 'Test_Acc=\K[0-9.]+' || echo "0")
    INFER=$(echo "$line" | grep -oP 'Infer_Latency=\K[0-9.]+' || echo "0")
    TLAT=$(echo "$line" | grep -oP 'Train_Latency=\K[0-9.]+' || echo "0")
    MEM=$(echo "$line" | grep -oP 'Memory=\K[0-9]+' || echo "0")
    DEV=$(echo "$line" | grep -oP 'Devices=\K[0-9]+' || echo "$NUM_NODES")
    echo "$TS,$TSPAN,$EP,$TRAIN,$TEST,$INFER,$TLAT,$MEM,$DEV,$DATASET"
done > "$LOG_DIR/metrics_final.csv" 2>/dev/null || true

if [[ -f "$LOG_DIR/metrics_final.csv" ]] && [[ -s "$LOG_DIR/metrics_final.csv" ]]; then
    head -1 "$RESULTS_CSV" > "$RESULTS_CSV.tmp"
    cat "$LOG_DIR/metrics_final.csv" >> "$RESULTS_CSV.tmp"
    mv "$RESULTS_CSV.tmp" "$RESULTS_CSV"
fi
rm -f "$LOG_DIR/metrics_tmp.csv" "$LOG_DIR/metrics_final.csv"

# =============================================================================
#  Summary
# =============================================================================
TOTAL_EPOCHS=$(grep -c "\[METRICS\]" "$LOG_DIR/run.log" 2>/dev/null || echo "0")
BEST_LINE=$(grep "\[METRICS\]" "$LOG_DIR/run.log" 2>/dev/null | \
    sed 's/.*Test_Acc=\([0-9.]*\).*/\1/' | sort -rn | head -1 || echo "0")
EARLY_STOPPED=$(grep -c "EARLY_STOP" "$LOG_DIR/run.log" 2>/dev/null || echo "0")

cat > "$LOG_DIR/summary.txt" <<EOF
═══════════════════════════════════════════════════
  RockNet Experiment Summary: $EXPERIMENT_NAME
═══════════════════════════════════════════════════
Dataset:        $DATASET
Nodes:          $NUM_NODES
Start:          $START_TIMESTAMP
End:            $(date '+%Y-%m-%d %H:%M:%S')
Duration:       ${DURATION}s ($(printf '%dh %dm %ds' $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60))))
Epochs:         $TOTAL_EPOCHS
Best Test Acc:  ${BEST_LINE}%
Early Stopped:  $([ "$EARLY_STOPPED" -gt 0 ] && echo "Yes" || echo "No")
Exit Code:      $RUN_EXIT
EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Experiment Complete                                               ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  Duration:       ${DURATION}s"
echo "║  Total Epochs:   $TOTAL_EPOCHS"
echo "║  Best Test Acc:  ${BEST_LINE}%"
echo "║  Early Stopped:  $([ "$EARLY_STOPPED" -gt 0 ] && echo "Yes" || echo "No")"
echo "║  Results CSV:    $RESULTS_CSV"
echo "║  Summary:        $LOG_DIR/summary.txt"
echo "╚══════════════════════════════════════════════════════════════════════╝"

exit 0
