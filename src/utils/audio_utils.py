"""音频处理工具函数"""

import os
import numpy as np
import ffmpeg


def clean_path(path_str: str) -> str:
    """
    清理路径字符串，移除多余的空格、引号和换行符
    
    Args:
        path_str: 原始路径字符串
        
    Returns:
        清理后的路径字符串
    """
    if path_str.endswith(("\\", "/")):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace("/", os.sep).replace("\\", os.sep)
    return path_str.strip(" '\n\"\u202a")


def load_audio(file: str, sr: int) -> np.ndarray:
    """
    加载音频文件并重采样到指定采样率
    
    Args:
        file: 音频文件路径
        sr: 目标采样率
        
    Returns:
        音频波形数据（numpy array，float32，单声道）
        
    Raises:
        RuntimeError: 音频加载失败
    """
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = clean_path(file)  # 防止小白拷路径头尾带了空格和"和回车
        if os.path.exists(file) is False:
            raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True)
        )  # Expose the Error
        raise RuntimeError("音频加载失败")

    return np.frombuffer(out, np.float32).flatten()
