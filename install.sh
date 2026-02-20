#!/bin/bash
# MiniCPMO45 Service 一键环境安装脚本
#
# 用法：
#   cd minicpmo45_service
#   bash install.sh
#
# 功能：
#   1. 创建 Python 3.10 虚拟环境
#   2. 安装 PyTorch + 核心依赖
#   3. 尝试安装 Flash Attention 2（失败自动跳过，降级 SDPA）
#   4. 验证安装结果
#
# 环境变量（可选）：
#   PYTHON=python3.11        指定 Python 解释器（默认 python3.10）
#   SKIP_FLASH_ATTN=1        跳过 Flash Attention 安装
#   MAX_JOBS=8               Flash Attention 编译并行数（默认 nproc）

set -e  # 遇到错误退出（flash-attn 部分单独处理）

# ============ 配置 ============

VENV_DIR=".venv/base"
PIP="${VENV_DIR}/bin/pip"
PYTHON_BIN="${VENV_DIR}/bin/python"
PYTHON="${PYTHON:-python3.10}"
MAX_JOBS="${MAX_JOBS:-$(nproc 2>/dev/null || echo 8)}"
FLASH_ATTN_VERSION=">=2.7.1,<=2.8.2"  # 模型官方推荐范围

# ============ 颜色输出 ============

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============ Step 1: 创建虚拟环境 ============

info "Step 1/4: 创建虚拟环境 (${VENV_DIR})"

if [ -d "${VENV_DIR}" ]; then
    warn "虚拟环境已存在: ${VENV_DIR}，跳过创建"
else
    if ! command -v "${PYTHON}" &> /dev/null; then
        error "${PYTHON} 未找到。请安装 Python 3.10+ 或通过 PYTHON=python3.x 指定路径"
        exit 1
    fi

    PYTHON_VERSION=$("${PYTHON}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    info "使用 Python ${PYTHON_VERSION} (${PYTHON})"

    "${PYTHON}" -m venv "${VENV_DIR}"
    info "虚拟环境创建完成"
fi

${PIP} install --upgrade pip -q

# ============ Step 2: 安装 PyTorch ============

info "Step 2/4: 安装 PyTorch + torchaudio"

# 检查是否已安装（跳过重复安装）
if ${PYTHON_BIN} -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.8"; then
    TORCH_VER=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)")
    CUDA_VER=$(${PYTHON_BIN} -c "import torch; print(torch.version.cuda)")
    info "PyTorch 已安装: ${TORCH_VER} (CUDA ${CUDA_VER})，跳过"
else
    ${PIP} install "torch==2.8.0" "torchaudio==2.8.0"
    TORCH_VER=$(${PYTHON_BIN} -c "import torch; print(torch.__version__)")
    CUDA_VER=$(${PYTHON_BIN} -c "import torch; print(torch.version.cuda)")
    info "PyTorch 安装完成: ${TORCH_VER} (CUDA ${CUDA_VER})"
fi

# ============ Step 3: 安装核心依赖 ============

info "Step 3/4: 安装核心依赖 (requirements.txt)"
${PIP} install -r requirements.txt
info "核心依赖安装完成"

# ============ Step 4: 安装 Flash Attention 2（可选） ============

info "Step 4/4: 安装 Flash Attention 2（可选，失败自动跳过）"

if [ "${SKIP_FLASH_ATTN}" = "1" ]; then
    warn "SKIP_FLASH_ATTN=1，跳过 Flash Attention 安装"
    warn "推理将使用 PyTorch SDPA（性能略低 5-15%）"
else
    # 检查是否已安装
    if ${PYTHON_BIN} -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null; then
        FA_VER=$(${PYTHON_BIN} -c "import flash_attn; print(flash_attn.__version__)")
        info "Flash Attention 已安装: ${FA_VER}，跳过"
    else
        info "尝试安装 flash-attn${FLASH_ATTN_VERSION}（MAX_JOBS=${MAX_JOBS}）..."
        info "这可能需要几分钟（编译 CUDA kernel）..."

        set +e  # 临时关闭 errexit，允许失败
        MAX_JOBS=${MAX_JOBS} ${PIP} install "flash-attn${FLASH_ATTN_VERSION}" --no-build-isolation 2>&1
        FLASH_EXIT_CODE=$?
        set -e  # 恢复 errexit

        if [ ${FLASH_EXIT_CODE} -eq 0 ]; then
            FA_VER=$(${PYTHON_BIN} -c "import flash_attn; print(flash_attn.__version__)")
            info "Flash Attention 安装成功: ${FA_VER}"
        else
            warn "=========================================="
            warn "Flash Attention 安装失败（exit code: ${FLASH_EXIT_CODE}）"
            warn "这不影响服务运行——推理将自动使用 PyTorch SDPA"
            warn "性能差异：SDPA 比 Flash Attention 慢约 5-15%"
            warn ""
            warn "常见原因："
            warn "  - CUDA toolkit 版本与 PyTorch 不匹配"
            warn "  - GPU 架构不支持（需要 SM80+，即 A100/H100 等）"
            warn "  - 编译工具链缺失（gcc/g++/nvcc）"
            warn ""
            warn "如需手动重试："
            warn "  MAX_JOBS=${MAX_JOBS} ${PIP} install \"flash-attn${FLASH_ATTN_VERSION}\" --no-build-isolation"
            warn "=========================================="
        fi
    fi
fi

# ============ 安装结果汇总 ============

echo ""
echo "============================================"
info "安装完成！环境汇总："
echo "============================================"

${PYTHON_BIN} -c "
import torch
print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')

try:
    import flash_attn
    print(f'  Flash Attn:   {flash_attn.__version__} ✓')
    attn_backend = 'flash_attention_2'
except ImportError:
    print(f'  Flash Attn:   未安装（将使用 SDPA）')
    attn_backend = 'sdpa'

import transformers
print(f'  Transformers: {transformers.__version__}')
print()
print(f'  Attention Backend: {attn_backend}')
"

echo ""
info "下一步："
echo "  1. 配置模型路径："
echo "     cp config.example.json config.json"
echo "     # 编辑 config.json，设置 model.model_path"
echo ""
echo "  2. 启动服务："
echo "     bash start_all.sh"
echo "============================================"
