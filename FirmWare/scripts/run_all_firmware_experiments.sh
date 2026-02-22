#!/bin/bash
set -euo pipefail
# =============================================================================
#  FirmWare CNN — Run ALL Experiments
# =============================================================================
#
#  Systematically runs FirmWare CNN experiments across:
#    • Datasets:   Cricket_X, FaceAll, ECG5000, Coffee
#    • Devices:    N = 7, 9, 11, 13, 15
#    • Architectures: 2-layer CNN, 3-layer CNN
#
#  Total experiments: 4 datasets × 5 N-values × 2 architectures = 40
#
#  Features:
#    • Builds the project first (cmake + make)
#    • Generates all configs if not already present
#    • Runs experiments sequentially with full cleanup between runs
#    • Early stopping (50 epochs no improvement → terminate that run)
#    • Collects all results into a master CSV at the end
#    • Skips experiments that already have results
#
#  Usage:
#    ./run_all_firmware_experiments.sh [UCR_DATA_ROOT] [TIMEOUT_PER_RUN]
#
#  Examples:
#    ./run_all_firmware_experiments.sh
#    ./run_all_firmware_experiments.sh /path/to/UCR_DATA 7200
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRMWARE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UCR_ROOT="${1:-/mnt/c/Users/GANESH KUMAR/Downloads/Pilot}"
TIMEOUT_PER_RUN="${2:-14400}"  # 4 hours per run

MASTER_LOG="$FIRMWARE_DIR/results/master_results.csv"
MASTER_SUMMARY="$FIRMWARE_DIR/results/experiment_log.txt"
RUN_SCRIPT="$SCRIPT_DIR/run_firmware_experiment.sh"

# Datasets and their properties
DATASETS=("Cricket_X" "FaceAll" "ECG5000" "Coffee")
N_VALUES=(7 9 11 13 15)
LAYER_COUNTS=(2 3)

mkdir -p "$FIRMWARE_DIR/results"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  FirmWare CNN — Full Experiment Suite                                      ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Datasets:      ${DATASETS[*]}"
echo "║  Device counts: ${N_VALUES[*]}"
echo "║  Architectures: 2-layer, 3-layer"
echo "║  Total runs:    $((${#DATASETS[@]} * ${#N_VALUES[@]} * ${#LAYER_COUNTS[@]}))"
echo "║  Timeout/run:   ${TIMEOUT_PER_RUN}s"
echo "║  UCR Data:      $UCR_ROOT"
echo "║  Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
#  Step 1: Build the project
# =============================================================================
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 1: Building FirmWare CNN ..."
echo "═══════════════════════════════════════════════════════════"

cd "$FIRMWARE_DIR"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 2>&1 | tail -5
make -j"$(nproc)" 2>&1 | tail -5

if [[ ! -f "$FIRMWARE_DIR/build/device" ]]; then
    echo "ERROR: Build failed — device binary not found"
    exit 1
fi
echo "  ✓ Build successful"
echo ""

# =============================================================================
#  Step 2: Generate all configs
# =============================================================================
echo "═══════════════════════════════════════════════════════════"
echo "  STEP 2: Generating all configurations ..."
echo "═══════════════════════════════════════════════════════════"

cd "$FIRMWARE_DIR"
python3 generate_all_configs.py 2>&1 | tail -5
echo "  ✓ All configs generated"
echo ""

# =============================================================================
#  Step 3: Initialize master CSV
# =============================================================================
if [[ ! -f "$MASTER_LOG" ]]; then
    echo "Timestamp,Timespan_s,Epoch,Train_Acc,Test_Acc,Infer_Latency_ms,Memory_KB,Config,Dataset,Devices,Layers" \
         > "$MASTER_LOG"
fi

SUITE_START=$(date +%s)
echo "$(date '+%Y-%m-%d %H:%M:%S') - Suite started" >> "$MASTER_SUMMARY"

# =============================================================================
#  Step 4: Run all experiments
# =============================================================================
RUN_COUNT=0
TOTAL_RUNS=$((${#DATASETS[@]} * ${#N_VALUES[@]} * ${#LAYER_COUNTS[@]}))
COMPLETED=0
FAILED=0
SKIPPED=0

for DATASET in "${DATASETS[@]}"; do
    for NUM_LAYERS in "${LAYER_COUNTS[@]}"; do
        for N in "${N_VALUES[@]}"; do
            RUN_COUNT=$((RUN_COUNT + 1))
            DS_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
            CONFIG_NAME="model_config_${DS_LOWER}_${NUM_LAYERS}L_N${N}"
            CONFIG_FILE="$FIRMWARE_DIR/configs/${CONFIG_NAME}.json"

            echo ""
            echo "═══════════════════════════════════════════════════════════"
            echo "  RUN $RUN_COUNT / $TOTAL_RUNS"
            echo "  Config: ${CONFIG_NAME}"
            echo "  Dataset: $DATASET | Layers: $NUM_LAYERS | Devices: $N"
            echo "═══════════════════════════════════════════════════════════"

            # Check if config exists
            if [[ ! -f "$CONFIG_FILE" ]]; then
                echo "  WARNING: Config file not found, skipping: $CONFIG_FILE"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            # Check if results already exist (skip if summary exists)
            RESULT_SUMMARY="$FIRMWARE_DIR/results/${CONFIG_NAME}/summary.txt"
            if [[ -f "$RESULT_SUMMARY" ]]; then
                echo "  SKIP: Results already exist at $RESULT_SUMMARY"
                SKIPPED=$((SKIPPED + 1))
                
                # Still collect into master CSV if metrics exist
                METRICS_CSV="$FIRMWARE_DIR/results/${CONFIG_NAME}/metrics.csv"
                if [[ -f "$METRICS_CSV" ]]; then
                    tail -n +2 "$METRICS_CSV" >> "$MASTER_LOG" 2>/dev/null || true
                fi
                continue
            fi

            # Run the experiment
            RUN_START=$(date +%s)
            echo "  Starting at $(date '+%H:%M:%S') ..."

            if bash "$RUN_SCRIPT" "$CONFIG_FILE" "$UCR_ROOT" "$TIMEOUT_PER_RUN"; then
                COMPLETED=$((COMPLETED + 1))
                echo "  ✓ Completed successfully"
            else
                FAILED=$((FAILED + 1))
                echo "  ✗ Failed (exit code: $?)"
            fi

            RUN_END=$(date +%s)
            RUN_DURATION=$((RUN_END - RUN_START))
            echo "  Duration: ${RUN_DURATION}s"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ${CONFIG_NAME}: duration=${RUN_DURATION}s" >> "$MASTER_SUMMARY"

            # Collect metrics into master CSV
            METRICS_CSV="$FIRMWARE_DIR/results/${CONFIG_NAME}/metrics.csv"
            if [[ -f "$METRICS_CSV" ]]; then
                tail -n +2 "$METRICS_CSV" >> "$MASTER_LOG" 2>/dev/null || true
            fi

            # Cooldown between runs
            echo "  Cooling down (5s) ..."
            sleep 5
        done
    done
done

# =============================================================================
#  Step 5: Final summary
# =============================================================================
SUITE_END=$(date +%s)
SUITE_DURATION=$((SUITE_END - SUITE_START))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  FULL SUITE COMPLETED                                                      ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Total Runs:    $TOTAL_RUNS"
echo "║  Completed:     $COMPLETED"
echo "║  Failed:        $FAILED"
echo "║  Skipped:       $SKIPPED"
echo "║  Total Duration: $(printf '%dh %dm %ds' $((SUITE_DURATION/3600)) $((SUITE_DURATION%3600/60)) $((SUITE_DURATION%60)))"
echo "║  Master CSV:    $MASTER_LOG"
echo "║  Experiment Log: $MASTER_SUMMARY"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Print best results per dataset
echo "Best Results by Dataset:"
echo "─────────────────────────────────────────────────────────────"
for DATASET in "${DATASETS[@]}"; do
    BEST=$(grep "$DATASET" "$MASTER_LOG" 2>/dev/null | sort -t',' -k5 -rn | head -1 || echo "")
    if [[ -n "$BEST" ]]; then
        ACC=$(echo "$BEST" | cut -d',' -f5)
        CFG=$(echo "$BEST" | cut -d',' -f8)
        echo "  $DATASET: Best Test Acc = ${ACC}% ($CFG)"
    fi
done

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - Suite finished: completed=$COMPLETED failed=$FAILED skipped=$SKIPPED duration=${SUITE_DURATION}s" >> "$MASTER_SUMMARY"
