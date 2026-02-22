#!/bin/bash
set -euo pipefail
# =============================================================================
#  MASTER EXPERIMENT SUITE — Run ALL FirmWare CNN + RockNet Experiments
# =============================================================================
#
#  This is the single entry point to run the complete experiment suite:
#
#  ┌────────────────────────────────────────────────────────────────────────┐
#  │  FirmWare CNN (Distributed Layer-wise Pipeline)                       │
#  │    • 4 datasets × 5 N-values × 2 architectures = 40 experiments      │
#  │    • 2-layer CNN: N = {7, 9, 11, 13, 15}                            │
#  │    • 3-layer CNN: N = {7, 9, 11, 13, 15}                            │
#  ├────────────────────────────────────────────────────────────────────────┤
#  │  RockNet Distributed (Kernel-split MiniRocket)                        │
#  │    • 4 datasets × 5 N-values = 20 experiments                        │
#  │    • N = {7, 9, 11, 13, 15} nodes, 84 kernels split evenly          │
#  └────────────────────────────────────────────────────────────────────────┘
#  Total: 60 experiments
#
#  Architecture Summary:
#  ─────────────────────
#  FirmWare CNN 2-Layer:
#    N=7:  H+2+3+T  → 32ch/48ch  → FC:96
#    N=9:  H+3+4+T  → 48ch/64ch  → FC:128
#    N=11: H+4+5+T  → 64ch/80ch  → FC:160
#    N=13: H+5+6+T  → 80ch/96ch  → FC:192
#    N=15: H+6+7+T  → 96ch/112ch → FC:224
#
#  FirmWare CNN 3-Layer:
#    N=7:  H+1+2+2+T → 16ch/32ch/32ch → FC:64
#    N=9:  H+2+2+3+T → 32ch/32ch/48ch → FC:96
#    N=11: H+2+3+4+T → 32ch/48ch/64ch → FC:128
#    N=13: H+2+4+5+T → 32ch/64ch/80ch → FC:160
#    N=15: H+3+4+6+T → 48ch/64ch/96ch → FC:192
#
#  RockNet MiniRocket:
#    N=7:  12 kernels/node → ~1368 features/device
#    N=9:  ~9 kernels/node → ~1026 features/device
#    N=11: ~8 kernels/node → ~912 features/device
#    N=13: ~6 kernels/node → ~684 features/device
#    N=15: ~6 kernels/node → ~684 features/device
#
#  Usage:
#    ./run_all_experiments.sh [UCR_DATA_ROOT]
#
#  Default UCR data root: /mnt/c/Users/GANESH KUMAR/Downloads/Pilot
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UCR_ROOT="${1:-/mnt/c/Users/GANESH KUMAR/Downloads/Pilot}"
FIRMWARE_TIMEOUT=14400    # 4h per FirmWare run
ROCKNET_TIMEOUT=7200      # 2h per RockNet run

FIRMWARE_DIR="$SCRIPT_DIR/FirmWare"
ROCKNET_DIR="$SCRIPT_DIR/RockNet-main"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║   ███╗   ███╗ █████╗ ███████╗████████╗███████╗██████╗                      ║"
echo "║   ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗                     ║"
echo "║   ██╔████╔██║███████║███████╗   ██║   █████╗  ██████╔╝                     ║"
echo "║   ██║╚██╔╝██║██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗                     ║"
echo "║   ██║ ╚═╝ ██║██║  ██║███████║   ██║   ███████╗██║  ██║                     ║"
echo "║   ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝                     ║"
echo "║                                                                            ║"
echo "║   Distributed On-Device Learning — Full Experiment Suite                   ║"
echo "║   60 experiments: 40 FirmWare CNN + 20 RockNet Distributed                 ║"
echo "║                                                                            ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  UCR Data Root: $UCR_ROOT"
echo "║  FirmWare CNN:  40 runs (4 datasets × 5 N × 2 architectures)"
echo "║  RockNet:       20 runs (4 datasets × 5 N)"
echo "║  FW Timeout:    ${FIRMWARE_TIMEOUT}s per run"
echo "║  RN Timeout:    ${ROCKNET_TIMEOUT}s per run"
echo "║  Started:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

TOTAL_START=$(date +%s)

# =============================================================================
#  Part 1: FirmWare CNN Experiments
# =============================================================================
echo ""
echo "█████████████████████████████████████████████████████████████████████████████"
echo "██  PART 1/2: FirmWare CNN Experiments (40 runs)                          ██"
echo "█████████████████████████████████████████████████████████████████████████████"
echo ""

if [[ -f "$FIRMWARE_DIR/scripts/run_all_firmware_experiments.sh" ]]; then
    bash "$FIRMWARE_DIR/scripts/run_all_firmware_experiments.sh" "$UCR_ROOT" "$FIRMWARE_TIMEOUT"
else
    echo "ERROR: FirmWare script not found: $FIRMWARE_DIR/scripts/run_all_firmware_experiments.sh"
fi

# =============================================================================
#  Part 2: RockNet Distributed Experiments
# =============================================================================
echo ""
echo "█████████████████████████████████████████████████████████████████████████████"
echo "██  PART 2/2: RockNet Distributed Experiments (20 runs)                   ██"
echo "█████████████████████████████████████████████████████████████████████████████"
echo ""

if [[ -f "$ROCKNET_DIR/scripts/run_all_rocknet_experiments.sh" ]]; then
    bash "$ROCKNET_DIR/scripts/run_all_rocknet_experiments.sh" "$UCR_ROOT" "$ROCKNET_TIMEOUT"
else
    echo "ERROR: RockNet script not found: $ROCKNET_DIR/scripts/run_all_rocknet_experiments.sh"
fi

# =============================================================================
#  Final Summary
# =============================================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETED                                                 ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Total Duration: $(printf '%dh %dm %ds' $((TOTAL_DURATION/3600)) $((TOTAL_DURATION%3600/60)) $((TOTAL_DURATION%60)))"
echo "║                                                                            ║"
echo "║  Results:                                                                  ║"
echo "║    FirmWare CNN: $FIRMWARE_DIR/results/master_results.csv"
echo "║    RockNet:      $ROCKNET_DIR/results/master_rocknet_results.csv"
echo "║                                                                            ║"
echo "║  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
