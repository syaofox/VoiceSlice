"""Faster Whisper ASR 实现"""

import os
import time
import traceback
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from tqdm import tqdm

# 兼容不同版本的 huggingface_hub
try:
    from huggingface_hub.errors import LocalEntryNotFoundError
except ImportError:
    # 旧版本可能没有这个错误类，使用通用异常
    LocalEntryNotFoundError = Exception

from .config import get_models
from .funasr_asr import only_asr

# fmt: off
language_code_list = [
    "af", "am", "ar", "as", "az", 
    "ba", "be", "bg", "bn", "bo", 
    "br", "bs", "ca", "cs", "cy", 
    "da", "de", "el", "en", "es", 
    "et", "eu", "fa", "fi", "fo", 
    "fr", "gl", "gu", "ha", "haw", 
    "he", "hi", "hr", "ht", "hu", 
    "hy", "id", "is", "it", "ja", 
    "jw", "ka", "kk", "km", "kn", 
    "ko", "la", "lb", "ln", "lo", 
    "lt", "lv", "mg", "mi", "mk", 
    "ml", "mn", "mr", "ms", "mt", 
    "my", "ne", "nl", "nn", "no", 
    "oc", "pa", "pl", "ps", "pt", 
    "ro", "ru", "sa", "sd", "si", 
    "sk", "sl", "sn", "so", "sq", 
    "sr", "su", "sv", "sw", "ta", 
    "te", "tg", "th", "tk", "tl", 
    "tr", "tt", "uk", "ur", "uz", 
    "vi", "yi", "yo", "zh", "yue",
    "auto"] 
# fmt: on


def download_model(model_size: str, base_path: str = None):
    """
    下载 Faster Whisper 模型
    
    Args:
        model_size: 模型尺寸
        base_path: 模型存储基础路径
        
    Returns:
        模型本地路径
    """
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "asr")
    os.makedirs(base_path, exist_ok=True)
    
    if "distil" in model_size:
        repo_id = "Systran/faster-{}-whisper-{}".format(*model_size.split("-", maxsplit=1))
    else:
        repo_id = f"Systran/faster-whisper-{model_size}"
    model_path = os.path.join(base_path, repo_id.strip("Systran/"))

    files: list[str] = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
    ]
    if model_size == "large-v3" or "distil" in model_size:
        files.append("preprocessor_config.json")
        files.append("vocabulary.json")
        files.remove("vocabulary.txt")

    for attempt in range(2):
        try:
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=files,
                local_dir=model_path,
            )
            break
        except LocalEntryNotFoundError:
            if attempt < 1:
                time.sleep(2)
            else:
                print("[ERROR] LocalEntryNotFoundError and no fallback.")
                traceback.print_exc()
                raise
        except Exception as e:
            print(f"[ERROR] Unexpected error on attempt {attempt + 1}: {e}")
            traceback.print_exc()
            if attempt == 1:
                raise

    return model_path


def execute_asr(input_folder, output_folder, model_size="large-v3", language="auto", precision="float16", output_mode=None):
    """
    执行 Faster Whisper ASR 识别
    
    Args:
        input_folder: 输入音频文件夹
        output_folder: 输出文件夹
        model_size: 模型尺寸
        language: 语言代码，"auto" 表示自动检测
        precision: 计算精度（float16, float32, int8）
        output_mode: 输出方式列表，可选值：["list"], ["txt"], ["list", "txt"]，默认为 ["list"]
        
    Returns:
        输出文件路径（如果output_mode包含"list"则返回list文件路径，否则返回None）
    """
    if output_mode is None:
        output_mode = ["list"]
    if language == "auto":
        language = None  # 不设置语种由模型自动输出概率最高的语种
    
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "asr")
    model_path = download_model(model_size, base_path)
    
    print(f"Loading faster whisper model: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_path, device=device, compute_type=precision)

    input_file_names = [f for f in os.listdir(input_folder) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    input_file_names.sort()

    output = []
    output_file_name = os.path.basename(input_folder)

    for file_name in tqdm(input_file_names, desc="Transcribing"):
        try:
            file_path = os.path.join(input_folder, file_name)
            segments, info = model.transcribe(
                audio=file_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language,
            )
            text = ""

            if info.language == "zh":
                print(f"检测为中文文本, 转 FunASR 处理: {file_name}")
                text = only_asr(file_path, language=info.language.lower())

            if text == "":
                for segment in segments:
                    text += segment.text
            
            # 如果选择了list输出方式，添加到输出列表
            if "list" in output_mode:
                output.append(f"{file_path}|{output_file_name}|{info.language.upper()}|{text}")
            
            # 如果选择了txt输出方式，在音频文件同目录生成同名txt文件
            if "txt" in output_mode:
                try:
                    audio_dir = os.path.dirname(file_path)
                    audio_name_without_ext = os.path.splitext(file_name)[0]
                    txt_file_path = os.path.join(audio_dir, f"{audio_name_without_ext}.txt")
                    
                    with open(txt_file_path, "w", encoding="utf-8") as txt_f:
                        txt_f.write(text)
                except Exception as e:
                    print(f"Error writing txt file for {file_name}: {e}")
                    traceback.print_exc()
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            traceback.print_exc()

    # 如果选择了list输出方式，生成list文件
    output_file_path = None
    if "list" in output_mode:
        output_folder = output_folder or "output/asr_opt"
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.abspath(os.path.join(output_folder, f"{output_file_name}.list"))

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output))
            print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
    else:
        print(f"ASR 任务完成（已生成txt文件）\n")
    
    return output_file_path
