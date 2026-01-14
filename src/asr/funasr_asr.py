"""FunASR ASR 实现（中文/粤语）"""

import os
import traceback

from funasr import AutoModel
from tqdm import tqdm

funasr_models = {}  # 存储模型避免重复加载


def only_asr(input_file, language="zh"):
    """
    使用 FunASR 进行 ASR 识别（仅识别，不包含批量处理）
    
    Args:
        input_file: 输入音频文件路径
        language: 语言代码（zh 或 yue）
        
    Returns:
        识别文本
    """
    try:
        model = create_model(language)
        text = model.generate(input=input_file)[0]["text"]
    except Exception as e:
        text = ""
        print(f"Error in only_asr: {traceback.format_exc()}")
    return text


def create_model(language="zh"):
    """
    创建或获取 FunASR 模型
    
    Args:
        language: 语言代码（zh 或 yue）
        
    Returns:
        FunASR 模型实例
    """
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "asr")
    
    path_vad = os.path.join(base_path, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
    path_punc = os.path.join(base_path, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
    
    # 如果本地不存在，使用 modelscope 的模型 ID
    if not os.path.exists(path_vad):
        path_vad = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    if not os.path.exists(path_punc):
        path_punc = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    
    vad_model_revision = punc_model_revision = "v2.0.4"

    if language == "zh":
        path_asr = os.path.join(base_path, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        if not os.path.exists(path_asr):
            path_asr = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        model_revision = "v2.0.4"
    elif language == "yue":
        path_asr = os.path.join(base_path, "speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online")
        if not os.path.exists(path_asr):
            path_asr = "iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"
        model_revision = "master"
        path_vad = path_punc = None
        vad_model_revision = punc_model_revision = None
        # 友情提示：粤语带VAD识别可能会有少量shape不对报错的，但是不带VAD可以.不带vad只能分阶段单独加标点。不过标点模型对粤语效果真的不行…
    else:
        raise ValueError(f"FunASR 不支持该语言: {language}")

    if language in funasr_models:
        return funasr_models[language]
    else:
        model = AutoModel(
            model=path_asr,
            model_revision=model_revision,
            vad_model=path_vad,
            vad_model_revision=vad_model_revision,
            punc_model=path_punc,
            punc_model_revision=punc_model_revision,
        )
        print(f"FunASR 模型加载完成: {language.upper()}")

        funasr_models[language] = model
        return model


def execute_asr(input_folder, output_folder, model_size="large", language="zh"):
    """
    执行 FunASR ASR 识别
    
    Args:
        input_folder: 输入音频文件夹
        output_folder: 输出文件夹
        model_size: 模型尺寸（FunASR 固定为 large）
        language: 语言代码（zh 或 yue）
        
    Returns:
        输出文件路径
    """
    input_file_names = [f for f in os.listdir(input_folder) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    input_file_names.sort()

    output = []
    output_file_name = os.path.basename(input_folder)

    model = create_model(language)

    for file_name in tqdm(input_file_names, desc="Transcribing"):
        try:
            print(f"\n{file_name}")
            file_path = os.path.join(input_folder, file_name)
            text = model.generate(input=file_path)[0]["text"]
            output.append(f"{file_path}|{output_file_name}|{language.upper()}|{text}")
        except Exception as e:
            print(f"Error processing {file_name}: {traceback.format_exc()}")

    output_folder = output_folder or "output/asr_opt"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.abspath(os.path.join(output_folder, f"{output_file_name}.list"))

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
        print(f"ASR 任务完成->标注文件路径: {output_file_path}\n")
    return output_file_path
