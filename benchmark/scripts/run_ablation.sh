#!/bin/bash
# ===========================================================================
# AD vs LBD Benchmark — 单分支消融实验
# 前置条件: 已按下方 Step A/B 完成代码/配置准备
# ===========================================================================

set -euo pipefail

SEED="${1:-42}"
GPU="${2:-0}"

# ——— Auto-detect benchmark root (parent of scripts/) ———
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${BENCHMARK_DIR}" || { echo "ERROR: cannot cd to ${BENCHMARK_DIR}"; exit 1; }
echo "[ablation] benchmark root: ${BENCHMARK_DIR}"

# --- helper: wrap any method config over the base anil config ---------------
run_ablation() {
    local note="$1"
    local method_cfg="$2"
    local extra_sets="${3:-}"

    # Temporarily swap in the method config
    cp configs/methods/anil.yaml configs/methods/anil.yaml.bak
    cp "${method_cfg}" configs/methods/anil.yaml
    trap 'cp configs/methods/anil.yaml.bak configs/methods/anil.yaml' EXIT

    local cmd="python scripts/run_single.py --method anil --seed ${SEED} --gpu ${GPU} --note \"${note}\""
    if [ -n "${extra_sets}" ]; then
        cmd="${cmd} ${extra_sets}"
    fi
    echo ">>> ${cmd}"
    eval "${cmd}"

    cp configs/methods/anil.yaml.bak configs/methods/anil.yaml
    trap - EXIT
}

echo "============================================"
echo "  Single-Branch Ablation  (Seed=${SEED}, GPU=${GPU})"
echo "============================================"
echo ""

# ——— Mamba-only ———————————————————————————————————————————————————————————
echo "--- Mamba-only ---"
run_ablation "mamba_only_v5" "configs/methods/anil_mamba_only.yaml"

# ——— Conv-only ————————————————————————————————————————————————————————————
echo "--- Conv-only ---"
run_ablation "conv_only_v5" "configs/methods/anil_conv_only.yaml"

echo ""
echo "Done. Results in results/anil/seed_${SEED}/"
