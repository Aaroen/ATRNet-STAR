#!/bin/bash
# ===================================================================
# ATRNet-STAR 项目一键环境配置脚本
# 该脚本将自动创建 Conda 环境并安装所有依赖。
# ===================================================================

# 如果任何命令失败，立即退出脚本
set -e

# 定义新环境的名称
ENV_NAME="ATRBench"

# --- 步骤 1: 创建全新的 Conda 环境 ---
echo ">>> 步骤 1/3: 正在检查并创建 Conda 环境 '$ENV_NAME'..."
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "--- 环境 '$ENV_NAME' 已存在，跳过创建。 ---"
else
conda create -n $ENV_NAME python=3.11 -y
    echo "--- 环境 '$ENV_NAME' 创建成功。 ---"
fi
echo

# --- 步骤 2: 安装 PyTorch ---
echo ">>> 步骤 2/3: 正在 '$ENV_NAME' 环境中安装 PyTorch..."
# 使用 conda run 在指定环境中执行命令，避免激活问题
conda run -n $ENV_NAME pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "--- PyTorch 安装完毕。 ---"
echo

# --- 步骤 3: 安装项目其余所有依赖 ---
echo ">>> 步骤 3/3: 正在 '$ENV_NAME' 环境中安装其余依赖..."
conda run -n $ENV_NAME pip install numpy scipy Pillow opencv-python-headless matplotlib tqdm captum timm
echo "--- 其余所有依赖安装成功。 ---"
echo

# --- 结束语 ---
echo "==================================================================="
echo "✅ 成功！您的开发环境 '$ENV_NAME' 已配置完毕！"
echo "   请通过 'conda activate $ENV_NAME' 命令激活环境后使用。"
echo "==================================================================="
