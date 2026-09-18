#!/bin/bash
#SBATCH --partition=gpu-a800
### 指定队列为gpu

#SBATCH --nodes=1
### 指定该作业需要1个节点数

#SBATCH --ntasks-per-node=2
### 每个节点所运行的进程数为52

#SBATCH --gres=gpu:1 
###（声明需要的GPU数量）【单节点最大申请4个GPU】

### 程序的执行命令
# ---------------------------------------------------------
# 3DINO 专用修复脚本：解决 libcublasLt.so.12 缺失问题
# 方案：不卸载 PyTorch，而是安装 NVIDIA CUDA 12 用户态运行时库
# ---------------------------------------------------------
nvidia-smi
which gpu
# 1. 设置临时目录，防止 /tmp 空间不足
export USER_TMP_DIR=$HOME/tmp_build_cu12
mkdir -p "$USER_TMP_DIR"
export TMPDIR="$USER_TMP_DIR"
export TEMP="$USER_TMP_DIR"
export TMP="$USER_TMP_DIR"
export PIP_CACHE_DIR="$USER_TMP_DIR/pip_cache"

echo "Using TMPDIR=$TMPDIR"

# 2. 激活环境 (请根据实际情况调整环境名称)
TARGET_CONDA_ENV=${TARGET_CONDA_ENV:-dino_fix}
source $HOME/miniconda3/bin/activate "$TARGET_CONDA_ENV"

set -euo pipefail

echo "=== [0] GPU 诊断 ==="
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-}"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
  nvidia-smi || true
else
  echo "nvidia-smi 不在 PATH"
fi

ls -l /dev/nvidia* || true
command -v lspci >/dev/null 2>&1 && lspci | grep -i nvidia || true
command -v lsmod >/dev/null 2>&1 && lsmod | grep -i nvidia || true

python - <<'PY'
import os
import torch
print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("device name:", torch.cuda.get_device_name(0))
else:
    raise SystemExit(
        "GPU 不可用：请检查作业是否真的调度到 GPU 节点，或容器/作业环境是否正确注入驱动"
    )
PY

if [ -n "${SLURM_JOB_ID:-}" ]; then
  MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_PORT=${MASTER_PORT:-29500}
  RANK=${SLURM_PROCID:-0}
  WORLD_SIZE=${SLURM_NTASKS:-1}
  LOCAL_RANK=${SLURM_LOCALID:-0}
  LOCAL_WORLD_SIZE=${SLURM_NTASKS_PER_NODE:-1}
else
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  MASTER_PORT=${MASTER_PORT:-29500}
  RANK=${RANK:-0}
  WORLD_SIZE=${WORLD_SIZE:-1}
  LOCAL_RANK=${LOCAL_RANK:-0}
  LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-1}
fi
export MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE

echo "=== [1] 运行环境自检 ==="
python - <<'PY'
import sys
major, minor = sys.version_info[:2]
print(f"Python version: {major}.{minor}")
if (major, minor) >= (3, 11):
    raise SystemExit("Python 版本过高，monai/依赖轮子不匹配，请切换到 3.9/3.10")
PY

echo "=== [2] 补充安装 CUDA 12 运行时库与缺失 Python 库 ==="
# 报错缺 libcublasLt.so.12，说明 PyTorch 是 cu12 版本
# 我们需要安装 nvidia-cublas-cu12 等包来提供这些 .so 文件

python - <<'PY'
import subprocess, sys, os
import time

env = os.environ.copy()

pkgs = [
    "numpy<2.0.0",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-curand-cu12",
    "monai==1.3.0",
    "nibabel==5.1.0",
    "scikit-learn",
    "torchio"
]

def install_packages(packages):
    """Install required packages with binary-only wheels."""
    mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
    cmd = [
        sys.executable, "-m", "pip", "install", 
        "--no-cache-dir", 
        "-i", mirror,
        "--only-binary=:all:" 
    ] + packages
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)

try:
    print("Installing NVIDIA cu12 runtime libraries...")
    install_packages(pkgs)
    print("Installation successful.")
except subprocess.CalledProcessError as e:
    print(f"Installation failed with code {e.returncode}")
    # 不退出，尝试继续，也许之前装过
PY

echo "=== [3] 动态链接库路径配置 ==="
# 这一步至关重要：将 pip 安装的 nvidia 库路径加入 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=""

# 自动寻找 site-packages 下的 nvidia 目录
PYTHON_SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
NVIDIA_DIR="$PYTHON_SITE_PACKAGES/nvidia"

if [ -d "$NVIDIA_DIR" ]; then
    echo "Found nvidia dir at $NVIDIA_DIR"
    # 遍历 nvidia 下的所有子目录，将其中的 lib 目录加入路径
    for lib_dir in "$NVIDIA_DIR"/*/lib; do
        if [ -d "$lib_dir" ]; then
            export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
        fi
    done
else
    echo "WARNING: $NVIDIA_DIR does not exist!"
fi

# 加上 PyTorch 自带的 lib 路径（如果有）
TORCH_LIB_DIR="$PYTHON_SITE_PACKAGES/torch/lib"
if [ -d "$TORCH_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"
fi

# 加上系统默认路径（可选，视情况而定）
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

echo "Final LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "=== [4] 验证库文件 ==="
python - <<'PY'
import ctypes
import os

print("Checking for libcublasLt.so.12...")
try:
    # 尝试加载，如果 LD_LIBRARY_PATH 设置正确，应该能成功
    ctypes.CDLL("libcublasLt.so.12")
    print("SUCCESS: libcublasLt.so.12 loaded successfully!")
except OSError as e:
    print(f"ERROR: Could not load libcublasLt.so.12: {e}")
    # 打印一下当前搜索路径
    print(f"LD_LIBRARY_PATH is: {os.environ.get('LD_LIBRARY_PATH')}")
PY

echo "=== [5] 启动训练 ==="
# 这里使用你提到的命令
# 注意：请确认脚本路径是否正确
SCRIPT_PATH="~/MRI/code/dinov2/train/train3d.py"
CONFIG_FILE="~/MRI/code/dinov2/configs/finetune/my_dataset_cls.yaml"
OUTPUT_DIR="~/MRI/code/out/3dino_finetune_cls"
CACHE_DIR="~/MRI/code/out/cache"
# 如果是相对路径，可以使用：
# SCRIPT_PATH="./train3d.py" 

DINO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)
DINO_PARENT=$(cd "$DINO_ROOT/.." && pwd)
export PYTHONPATH="$DINO_PARENT:$DINO_ROOT:${PYTHONPATH:-}"

echo "PYTHONPATH=$PYTHONPATH"
python - <<'PY'
import sys
print("sys.path[0:5]:", sys.path[:5])
import dinov2
print("dinov2 loaded from:", dinov2.__file__)
PY

echo "Running: python $SCRIPT_PATH"
# 传递所有脚本参数
python "$SCRIPT_PATH" \
  --config-file "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  "$@"
