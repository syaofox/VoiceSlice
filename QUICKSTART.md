# 快速开始指南

## 前置要求

- Python >= 3.9
- FFmpeg（用于音频处理）

## 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或者使用 pip：

```bash
pip install uv
```

## 2. 安装项目依赖

```bash
cd VoiceSlice
uv sync
```

如果需要 GPU 支持（CUDA 12.8，Linux/Windows）：

```bash
uv sync --extra gpu
```

注意：在 Linux/Windows 上，即使不使用 `--extra gpu`，也会自动安装 GPU 版本的 PyTorch（CUDA 12.8）。如果需要 CPU 版本，请手动安装。

## 3. 启动 WebUI

### 方式1：使用启动脚本（推荐）

```bash
./run.sh
```

### 方式2：使用 uv run

```bash
uv run python webui/app.py
```

### 方式3：使用 Python

```bash
# 激活虚拟环境（如果需要）
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows

python webui/app.py
```

## 4. 访问 WebUI

打开浏览器访问：`http://localhost:7860`

## 5. 使用流程

### 音频切片

1. 切换到"音频切片"标签页
2. 输入音频文件或文件夹路径
3. 设置输出目录
4. 调整切片参数（可选）
5. 点击"开始切片"

### 文本识别

1. 切换到"文本识别"标签页
2. 输入切片后的音频文件夹路径
3. 选择 ASR 模型
4. 设置语言和其他参数
5. 点击"开始识别"

### 完整流程

1. 切换到"完整流程"标签页
2. 输入音频文件或文件夹路径
3. 设置输出目录
4. 配置切片和识别参数
5. 点击"开始处理"

## 常见问题

### Q: 模型下载很慢怎么办？

A: 
- Faster Whisper 模型会自动下载，首次使用需要等待
- FunASR 模型需要手动下载（见 README.md）
- 可以使用镜像源加速下载

### Q: 如何修改端口？

A: 编辑 `config.yaml` 文件中的 `webui.port` 配置项。

### Q: 支持哪些音频格式？

A: 支持所有 FFmpeg 支持的格式，包括 WAV、MP3、M4A、FLAC、OGG 等。

## 下一步

- 查看 [README.md](README.md) 了解详细功能
- 查看 [config.yaml](config.yaml) 了解配置选项
- 查看源代码了解实现细节
