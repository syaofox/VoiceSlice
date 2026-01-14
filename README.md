# VoiceSlice - 音频切片和文本识别工具

从 GPT-SoVITS 项目中提取的音频切片和文本识别功能，提供完整的 WebUI 界面，支持批量处理和实时预览。

## 功能特性

- 🎵 **音频自动切片**：基于静音检测的智能音频分割
- 🗣️ **多模型 ASR**：支持 Faster Whisper（多语种）和 FunASR（中文/粤语）
- 🌐 **WebUI 界面**：基于 Gradio 的现代化 Web 界面
- 📦 **批量处理**：支持文件夹批量处理
- ⚡ **实时预览**：处理进度和结果实时显示
- 🔧 **参数可调**：所有切片和识别参数可自定义

## 项目结构

```
VoiceSlice/
├── src/
│   ├── slicer/          # 音频切片模块
│   ├── asr/            # ASR 文本识别模块
│   └── utils/          # 工具函数
├── webui/              # WebUI 界面
├── output/             # 输出目录
├── models/             # 模型存储目录
├── pyproject.toml      # 项目配置（uv 管理）
├── config.yaml         # 应用配置
└── README.md           # 项目文档
```

## 安装

### 前置要求

- Python >= 3.9
- FFmpeg（用于音频处理）
- [uv](https://github.com/astral-sh/uv)（包管理器）

### 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或使用 pip
pip install uv
```

### 安装项目

```bash
# 克隆项目
git clone <repository-url>
cd VoiceSlice

# 使用 uv 安装依赖
uv sync

# 如果需要 GPU 支持，需要手动安装 GPU 版本的 PyTorch
# 方法1：使用 pip（推荐）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# 方法2：使用 conda
# conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 使用方法

### WebUI 方式（推荐）

```bash
# 启动 WebUI
uv run python webui/app.py

# 或使用 uv 直接运行
uv run webui/app.py
```

然后在浏览器中打开 `http://localhost:7860` 访问 WebUI。

### 命令行方式

#### 音频切片

```python
from src.slicer import slice_audio

slice_audio(
    inp="path/to/audio.wav",
    opt_root="output/sliced",
    threshold=-34,
    min_length=4000,
    min_interval=300,
    hop_size=10,
    max_sil_kept=500,
    _max=0.9,
    alpha=0.25,
)
```

#### 文本识别

```python
from src.asr import fasterwhisper_asr, funasr_asr

# 使用 Faster Whisper
fasterwhisper_asr(
    input_folder="output/sliced",
    output_folder="output/asr",
    model_size="large-v3",
    language="auto",
    precision="float16",
)

# 使用 FunASR（中文）
funasr_asr(
    input_folder="output/sliced",
    output_folder="output/asr",
    model_size="large",
    language="zh",
)
```

## WebUI 功能说明

### 1. 音频切片标签页

- **输入路径**：选择要切片的音频文件或文件夹
- **输出目录**：设置切片后的输出目录
- **切片参数**：
  - 音量阈值：静音检测阈值（dB）
  - 最小长度：每段音频的最小长度（毫秒）
  - 最小间隔：切割点的最小间隔（毫秒）
  - 帧长度：用于计算音量曲线的帧长度（毫秒）
  - 最大静音保留：切完后静音最多保留的长度（毫秒）
  - 归一化最大值：音频归一化的最大值
  - 混音比例：音频混音的比例

### 2. 文本识别标签页

- **输入文件夹**：选择切片后的音频文件夹
- **ASR 模型**：
  - Faster Whisper（多语种）：支持多种语言，自动语言检测
  - 达摩 ASR（中文）：专门针对中文和粤语优化
- **语言设置**：选择识别语言（auto 表示自动检测）
- **模型尺寸**：Faster Whisper 的模型大小（仅 Faster Whisper）
- **精度**：计算精度（float32/float16/int8，仅 Faster Whisper）

### 3. 完整流程标签页

一键执行：上传 → 切片 → 识别，自动完成整个流程。

## 配置说明

编辑 `config.yaml` 可以修改默认配置：

```yaml
# WebUI 配置
webui:
  host: "0.0.0.0"
  port: 7860
  share: false

# 音频切片默认参数
slicer:
  threshold: -34
  min_length: 4000
  min_interval: 300
  hop_size: 10
  max_sil_kept: 500
  max: 0.9
  alpha: 0.25

# ASR 默认配置
asr:
  default_model: "Faster Whisper (多语种)"
  default_language: "auto"
  default_precision: "float16"
  default_model_size: "large-v3"
```

## 模型下载

### Faster Whisper

模型会在首次使用时自动下载到 `models/asr/` 目录。

### FunASR

FunASR 模型需要手动下载：

1. **中文 ASR 模型**：
   - [Paraformer Large](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
   - [VAD 模型](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch)
   - [标点模型](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)

2. **粤语 ASR 模型**：
   - [UniASR Cantonese](https://modelscope.cn/models/iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online)

下载后放置到 `models/asr/` 目录下对应的文件夹中。

## 输出格式

### 切片输出

切片后的音频文件命名格式：`原文件名_起始帧_结束帧.wav`

### ASR 输出

识别结果保存在 `.list` 文件中，格式为：
```
文件路径|文件夹名|语言|识别文本
```

示例：
```
/path/to/audio_0000000000_0000005000.wav|sliced|ZH|这是识别出的文本内容
```

## 常见问题

### Q: 如何提高识别准确率？

A: 
- 对于中文，推荐使用 FunASR（达摩 ASR）
- 调整切片参数，确保每个音频片段清晰完整
- 使用更大的模型（如 large-v3）

### Q: 支持哪些音频格式？

A: 支持所有 FFmpeg 支持的格式，包括 WAV、MP3、M4A、FLAC 等。

### Q: GPU 加速如何启用？

A: 
1. 安装 CUDA 环境（推荐 CUDA 12.8）
2. 使用 `uv sync --extra gpu` 安装 GPU 版本的 PyTorch：
   ```bash
   uv sync --extra gpu
   ```
   这会自动从 PyTorch CUDA 索引安装 GPU 版本的 torch 和 torchaudio
3. 系统会自动检测并使用 GPU

注意：默认安装的是 CPU 版本，只有使用 `--extra gpu` 时才会安装 GPU 版本。

### Q: 模型文件很大，可以自定义存储位置吗？

A: 可以修改代码中的 `models/asr/` 路径，或使用符号链接。

## 开发

### 项目依赖

主要依赖见 `pyproject.toml`：

- numpy<2.0
- scipy
- librosa==0.10.2
- faster-whisper>=1.1.1
- funasr==1.0.27
- gradio<5
- torch, torchaudio

### 代码结构

- `src/slicer/`：音频切片核心算法
- `src/asr/`：ASR 识别实现
- `src/utils/`：工具函数
- `webui/`：Gradio WebUI 界面

## 许可证

本项目基于 MIT 许可证开源。

## 致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)：原始项目
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)：多语种 ASR
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)：中文 ASR
- [Gradio](https://github.com/gradio-app/gradio)：WebUI 框架

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v0.1.0 (2024)

- 初始版本
- 音频切片功能
- Faster Whisper 和 FunASR 支持
- WebUI 界面
- 批量处理支持
