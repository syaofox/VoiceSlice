#!/bin/bash
# VoiceSlice 启动脚本

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "错误: 未找到 uv，请先安装 uv"
    echo "安装命令: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 检查依赖是否安装
if [ ! -d ".venv" ] && [ ! -f "uv.lock" ]; then
    echo "首次运行，正在安装依赖..."
    uv sync
fi

# 启动 WebUI
echo "启动 VoiceSlice WebUI..."
uv run python webui/app.py
