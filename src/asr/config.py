"""ASR 配置"""

import os


def check_fw_local_models():
    """
    启动时检查本地是否有 Faster Whisper 模型.
    """
    model_size_list = [
        "medium",
        "medium.en",
        "distil-large-v2",
        "distil-large-v3",
        "large-v1",
        "large-v2",
        "large-v3",
    ]
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "asr")
    for i, size in enumerate(model_size_list):
        if os.path.exists(os.path.join(base_path, f"faster-whisper-{size}")):
            model_size_list[i] = size + "-local"
    return model_size_list


def get_models():
    """获取可用的 Faster Whisper 模型列表"""
    model_size_list = [
        "medium",
        "medium.en",
        "distil-large-v2",
        "distil-large-v3",
        "large-v1",
        "large-v2",
        "large-v3",
    ]
    return model_size_list


asr_dict = {
    "达摩 ASR (中文)": {
        "lang": ["zh", "yue"],
        "size": ["large"],
        "path": "funasr_asr",
        "precision": ["float32"]
    },
    "Faster Whisper (多语种)": {
        "lang": ["auto", "zh", "en", "ja", "ko", "yue"],
        "size": get_models(),
        "path": "fasterwhisper_asr",
        "precision": ["float32", "float16", "int8"],
    },
}
