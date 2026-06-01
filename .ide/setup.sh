#!/bin/bash

# ==============================================================================
#  AnimaLoraStudio 容器环境初始化脚本（Docker）
# ==============================================================================

set -e

echo "============================================"
echo " AnimaLoraStudio 环境初始化"
echo "============================================"
echo ""

# --- 1. Git LFS 拉取 ---
echo "[1/5] 拉取 Git LFS 文件..."
if command -v git-lfs &> /dev/null; then
    git lfs pull
    echo "  Git LFS pull 完成。"
else
    echo "  [警告] git-lfs 未安装,跳过 LFS 拉取。"
fi
echo ""

# --- 2. 模型文件检查 ---
echo "[2/5] 检查模型文件..."
MISSING_MODELS=0

check_file() {
    local desc="$1"
    local path="$2"
    if [ -f "$path" ]; then
        local size=$(du -h "$path" | cut -f1)
        echo "  [OK] $desc ($size)"
    else
        echo "  [缺失] $desc -> $path"
        MISSING_MODELS=1
    fi
}

check_dir() {
    local desc="$1"
    local path="$2"
    if [ -d "$path" ] && [ -n "$(ls -A "$path" 2>/dev/null)" ]; then
        echo "  [OK] $desc (目录存在且非空)"
    else
        echo "  [缺失] $desc -> $path"
        MISSING_MODELS=1
    fi
}

check_file "Transformer"  "models/diffusion_models/anima-preview3-base.safetensors"
check_file "VAE"          "models/vae/qwen_image_vae.safetensors"
check_dir  "Text Encoder" "models/text_encoders"
check_dir  "T5 Tokenizer" "models/t5_tokenizer"

if [ "$MISSING_MODELS" -eq 1 ]; then
    echo ""
    echo "  [提示] 部分模型文件缺失,请手动放入对应目录。"
fi
echo ""

# --- 3. 必要目录 ---
echo "[3/5] 创建必要目录..."
mkdir -p output logs
echo "  output/ logs/ 就绪。"
echo ""

# --- 4. 前端 node_modules ---
echo "[4/5] 前端依赖..."
if [ ! -d "studio/web/node_modules" ]; then
    if [ -d "/opt/studio-web/node_modules" ]; then
        echo "  从镜像缓存复制 node_modules..."
        cp -r /opt/studio-web/node_modules studio/web/
        echo "  完成。"
    else
        echo "  [警告] 镜像中无缓存,首次运行 studio.sh 时会自动安装。"
    fi
else
    echo "  [OK] node_modules 已存在。"
fi
echo ""

# --- 5. 额外 Python 依赖 ---
echo "[5/5] 安装额外 Python 依赖..."

FLASH_ATTN_WHL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl"
TENCENT_MIRROR="https://mirrors.tencentyun.com/pypi/simple/"

# flash-attn (直接拉 GitHub release)
if python -c "import flash_attn" 2>/dev/null; then
    echo "  [OK] flash_attn 已安装,跳过。"
else
    echo "  安装 flash_attn..."
    pip install --no-cache-dir "$FLASH_ATTN_WHL"
    echo "  flash_attn 安装完成。"
fi

# onnxruntime-gpu (走腾讯云内网镜像)
if python -c "import onnxruntime; assert onnxruntime.__version__ == '1.26.0'" 2>/dev/null; then
    echo "  [OK] onnxruntime-gpu 1.26.0 已安装,跳过。"
else
    echo "  安装 onnxruntime-gpu==1.26.0..."
    pip install --no-cache-dir -i "$TENCENT_MIRROR" onnxruntime-gpu==1.26.0
    echo "  onnxruntime-gpu 安装完成。"
fi
echo ""

# --- GPU 检测 ---
echo "============================================"
echo " GPU 状态"
echo "============================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 运行失败。"
else
    echo "  [警告] nvidia-smi 不可用。"
fi
echo ""

echo "============================================"
echo " 初始化完成! 运行: bash studio.sh"
echo "============================================"