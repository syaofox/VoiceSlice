"""ASR 文本识别模块"""

from .config import asr_dict, get_models
from .fasterwhisper_asr import execute_asr as fasterwhisper_asr
from .funasr_asr import execute_asr as funasr_asr

__all__ = ["asr_dict", "get_models", "fasterwhisper_asr", "funasr_asr"]
