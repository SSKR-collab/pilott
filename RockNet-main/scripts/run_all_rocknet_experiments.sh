#!/bin/bash
set -euo pipefail
# =============================================================================
#  RockNet Distributed — Run ALL Experiments
# =============================================================================
#
#  Systematically runs RockNet distributed experiments across:
#    • Datasets:   Cricket_X, FaceAll, ECG5000, Coffee
#    • Devices:    N = 7, 9, 11, 13, 15
#
#  Total experiments: 4 datasets × 5 N-values = 20
#
#  For each (dataset, N):
#    1. Generate rocket_config.{h,c} with split_kernels(N, 84)
#    2. Compile distributed_sim with -DNUM_NODES=N
#    3. Run the simulation (early stopping: 50 epochs patience)
#    4. Collect metrics into master CSV
#
#  RockNet Kernel Split (84 kernels ÷ N nodes):
#    N=7:  12 kernels/node → 12×6×19 = 1368 features/device
#    N=9:  ~9-10 kernels/node (84/9=9 r3, so 3 nodes get 10, 6 get 9)
#    N=11: ~7-8 kernels/node
#    N=13: ~6-7 kernels/node
#    N=15: ~5-6 kernels/node
#
#  Usage:
#    ./run_all_rocknet_experiments.sh [UCR_DATA_ROOT] [TIMEOUT_PER_RUN]
#
#  Examples:
#    ./run_all_rocknet_experiments.sh
#    ./run_all_rocknet_experiments.sh /path/to/UCR_DATA 3600
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCKNET_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UCR_ROOT="${1:-/mnt/c/Users/GANESH KUMAR/Downloads/Pilot}"
TIMEOUT_PER_RUN="${2:-7200}"  # 2 hours per run

RUN_SCRIPT="$SCRIPT_DIR/run_rocknet_experiment.sh"
MASTER_LOG="$ROCKNET_DIR/results/master_rocknet_results.csv"
MASTER_SUMMARY="$ROCKNET_DIR/results/rocknet_experiment_log.txt"

DATASETS=("Cricket_X" "FaceAll" "ECG5000" "Coffee")
N_VALUES=(7 9 11 13 15)

mkdir -p "$ROCKNET_DIR/results"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  RockNet Distributed — Full Experiment Suite                               ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Datasets:      ${DATASETS[*]}"
echo "║  Device counts: ${N_VALUES[*]}"
echo "║  Total runs:    $((${#DATASETS[@]} * ${#N_VALUES[@]}))"
echo "║  Timeout/run:   ${TIMEOUT_PER_RUN}s"
echo "║  UCR Data:      $UCR_ROOT"
echo "║  Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
#  Check dependencies
# =============================================================================
echo "[CHECK] Verifying dependencies ..."

# Check gcc
if ! command -v gcc &>/dev/null; then
    echo "ERROR: gcc not found. Install: sudo apt install build-essential"
    exit 1
fi

# Check python3 + numpy + pandas + sympy
if ! python3 -c "import numpy, pandas, sympy" 2>/dev/null; then
    echo "WARNING: Python dependencies missing. Attempting install..."
    pip3 install numpy pandas sympy 2>/dev/null || {
        echo "ERROR: Could not install Python dependencies."
        echo "Install manually: pip3 install numpy pandas sympy"
        exit 1
    }
fi

echo "  ✓ All dependencies OK"
echo ""

# =============================================================================
#  Initialize master CSV
# =============================================================================
if [[ ! -f "$MASTER_LOG" ]]; then
    echo "Timestamp,Timespan_s,Epoch,Train_Acc,Test_Acc,Infer_Latency_ms,Train_Latency_ms,Memory_KB,Devices,Dataset" \
         > "$MASTER_LOG"
fi

SUITE_START=$(date +%s)
echo "$(date '+%Y-%m-%d %H:%M:%S') - RockNet suite started" >> "$MASTER_SUMMARY"

# =============================================================================
#  Run all experiments
# =============================================================================
RUN_COUNT=0
TOTAL_RUNS=$((${#DATASETS[@]} * ${#N_VALUES[@]}))
COMPLETED=0
FAILED=0
SKIPPED=0

for DATASET in "${DATASETS[@]}"; do
    for N in "${N_VALUES[@]}"; do
        RUN_COUNT=$((RUN_COUNT + 1))
        EXPERIMENT_NAME="rocknet_${DATASET}_N${N}"

        echo ""
        echo "═══════════════════════════════════════════════════════════"
        echo "  RUN $RUN_COUNT / $TOTAL_RUNS: $EXPERIMENT_NAME"
        echo "═══════════════════════════════════════════════════════════"

        # Check if results already exist
        RESULT_SUMMARY="$ROCKNET_DIR/results/${EXPERIMENT_NAME}/summary.txt"
        if [[ -f "$RESULT_SUMMARY" ]]; then
            echo "  SKIP: Results already exist at $RESULT_SUMMARY"
            SKIPPED=$((SKIPPED + 1))
            
            # Still collect metrics into master CSV
            METRICS_CSV="$ROCKNET_DIR/results/${EXPERIMENT_NAME}/metrics.csv"
            if [[ -f "$METRICS_CSV" ]]; then
                tail -n +2 "$METRICS_CSV" >> "$MASTER_LOG" 2>/dev/null || true
            fi
            continue
        fi

        # Run the experiment
        RUN_START=$(date +%s)
        echo "  Starting at $(date '+%H:%M:%S') ..."

        if bash "$RUN_SCRIPT" "$DATASET" "$N" "$UCR_ROOT" "$TIMEOUT_PER_RUN"; then
            COMPLETED=$((COMPLETED + 1))
            echo "  ✓ Completed"
        else
            FAILED=$((FAILED + 1))
            echo "  ✗ Failed (exit code: $?)"
        fi

        RUN_END=$(date +%s)
        RUN_DURATION=$((RUN_END - RUN_START))
        echo "  Duration: ${RUN_DURATION}s"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - ${EXPERIMENT_NAME}: duration=${RUN_DURATION}s" >> "$MASTER_SUMMARY"

        # Collect metrics
        METRICS_CSV="$ROCKNET_DIR/results/${EXPERIMENT_NAME}/metrics.csv"
        if [[ -f "$METRICS_CSV" ]]; then
            tail -n +2 "$METRICS_CSV" >> "$MASTER_LOG" 2>/dev/null || true
        fi

        # Cooldown
        sleep 3
    done
done

# =============================================================================
#  Final summary
# =============================================================================
SUITE_END=$(date +%s)
SUITE_DURATION=$((SUITE_END - SUITE_START))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  ROCKNET FULL SUITE COMPLETED                                              ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Total Runs:    $TOTAL_RUNS"
echo "║  Completed:     $COMPLETED"
echo "║  Failed:        $FAILED"
echo "║  Skipped:       $SKIPPED"
echo "║  Total Duration: $(printf '%dh %dm %ds' $((SUITE_DURATION/3600)) $((SUITE_DURATION%3600/60)) $((SUITE_DURATION%60)))"
echo "║  Master CSV:    $MASTER_LOG"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Print best results per dataset
echo "Best Results by Dataset (RockNet):"
echo "─────────────────────────────────────────────────────────────"
for DATASET in "${DATASETS[@]}"; do
    BEST=$(grep "$DATASET" "$MASTER_LOG" 2>/dev/null | sort -t',' -k5 -rn | head -1 || echo "")
    if [[ -n "$BEST" ]]; then
        ACC=$(echo "$BEST" | cut -d',' -f5)
        DEV=$(echo "$BEST" | cut -d',' -f9)
        echo "  $DATASET: Best Test Acc = ${ACC}% (N=${DEV})"
    fi
done

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - Suite finished: completed=$COMPLETED failed=$FAILED skipped=$SKIPPED duration=${SUITE_DURATION}s" >> "$MASTER_SUMMARY"
