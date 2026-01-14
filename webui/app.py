"""VoiceSlice WebUI 主程序"""

import os
import sys
import yaml
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
import numpy as np
from tqdm import tqdm

from src.asr import asr_dict, fasterwhisper_asr, funasr_asr
from src.slicer import slice_audio


# 加载配置
def load_config():
    """加载配置文件"""
    config_path = project_root / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


config = load_config()

# 默认配置
DEFAULT_SLICE_PARAMS = config.get("slicer", {
    "threshold": -34,
    "min_length": 4000,
    "min_interval": 300,
    "hop_size": 10,
    "max_sil_kept": 500,
    "max": 0.9,
    "alpha": 0.25,
})

DEFAULT_ASR_CONFIG = config.get("asr", {
    "default_model": "Faster Whisper (多语种)",
    "default_language": "auto",
    "default_precision": "float16",
    "default_model_size": "large-v3",
})

OUTPUT_DIR = config.get("paths", {}).get("output_dir", "output")
SLICE_OUTPUT = config.get("paths", {}).get("slice_output", "output/slicer_opt")
ASR_OUTPUT = config.get("paths", {}).get("asr_output", "output/asr_opt")


def process_slice(
    input_path,
    output_dir,
    threshold,
    min_length,
    min_interval,
    hop_size,
    max_sil_kept,
    max_val,
    alpha,
    progress=gr.Progress(),
):
    """处理音频切片"""
    try:
        if not input_path:
            return "错误：请选择输入文件或文件夹", None
        
        if not output_dir:
            output_dir = SLICE_OUTPUT
        
        os.makedirs(output_dir, exist_ok=True)
        
        progress(0, desc="开始切片...")
        
        result = slice_audio(
            inp=input_path,
            opt_root=output_dir,
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept,
            _max=max_val,
            alpha=alpha,
            i_part=0,
            all_part=1,
        )
        
        progress(1.0, desc="切片完成")
        
        # 统计切片文件数量
        slice_count = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
        
        return f"切片完成！共生成 {slice_count} 个音频片段\n输出目录：{output_dir}", output_dir
    except Exception as e:
        return f"切片失败：{str(e)}", None


def process_asr(
    input_folder,
    output_dir,
    asr_model,
    language,
    model_size,
    precision,
    progress=gr.Progress(),
):
    """处理 ASR 识别"""
    try:
        if not input_folder:
            return "错误：请选择输入文件夹", None
        
        if not output_dir:
            output_dir = ASR_OUTPUT
        
        os.makedirs(output_dir, exist_ok=True)
        
        progress(0, desc="开始识别...")
        
        if asr_model == "达摩 ASR (中文)":
            result_path = funasr_asr(
                input_folder=input_folder,
                output_folder=output_dir,
                model_size="large",
                language=language,
            )
        else:  # Faster Whisper
            result_path = fasterwhisper_asr(
                input_folder=input_folder,
                output_folder=output_dir,
                model_size=model_size,
                language=language,
                precision=precision,
            )
        
        progress(1.0, desc="识别完成")
        
        # 读取结果文件
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = f.read().strip().split("\n")
            
            result_text = f"识别完成！共识别 {len(results)} 个文件\n\n结果预览（前10条）：\n\n"
            for i, line in enumerate(results[:10]):
                parts = line.split("|")
                if len(parts) >= 4:
                    result_text += f"{i+1}. {parts[3]}\n"
            if len(results) > 10:
                result_text += f"\n... 还有 {len(results) - 10} 条结果\n"
            
            return result_text, result_path
        else:
            return "识别完成，但结果文件未找到", None
            
    except Exception as e:
        import traceback
        return f"识别失败：{str(e)}\n{traceback.format_exc()}", None


def process_full_pipeline(
    input_path,
    slice_output_dir,
    asr_output_dir,
    asr_model,
    language,
    model_size,
    precision,
    threshold,
    min_length,
    min_interval,
    hop_size,
    max_sil_kept,
    max_val,
    alpha,
    progress=gr.Progress(),
):
    """完整流程：切片 + 识别"""
    try:
        # 步骤1：切片
        progress(0.1, desc="步骤 1/2: 音频切片...")
        
        # 创建一个简单的进度回调
        class SimpleProgress:
            def __init__(self, base_progress, start, end):
                self.base_progress = base_progress
                self.start = start
                self.end = end
            
            def __call__(self, value, desc=None):
                if desc:
                    self.base_progress(self.start + (self.end - self.start) * value, desc=desc)
                else:
                    self.base_progress(self.start + (self.end - self.start) * value)
        
        slice_progress = SimpleProgress(progress, 0.1, 0.5)
        slice_result, slice_output = process_slice(
            input_path=input_path,
            output_dir=slice_output_dir,
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept,
            max_val=max_val,
            alpha=alpha,
            progress=slice_progress,
        )
        
        if "失败" in slice_result or slice_output is None:
            return slice_result, None, None
        
        # 步骤2：识别
        progress(0.6, desc="步骤 2/2: 文本识别...")
        asr_progress = SimpleProgress(progress, 0.6, 0.95)
        asr_result, asr_output = process_asr(
            input_folder=slice_output,
            output_dir=asr_output_dir,
            asr_model=asr_model,
            language=language,
            model_size=model_size,
            precision=precision,
            progress=asr_progress,
        )
        
        progress(1.0, desc="全部完成！")
        
        return f"完整流程完成！\n\n{slice_result}\n\n{asr_result}", slice_output, asr_output
        
    except Exception as e:
        import traceback
        return f"流程失败：{str(e)}\n{traceback.format_exc()}", None, None


# 创建 Gradio 界面
def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="VoiceSlice - 音频切片和文本识别", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # VoiceSlice - 音频切片和文本识别工具
            
            从 GPT-SoVITS 项目提取的音频切片和文本识别功能，支持批量处理和实时预览。
            """
        )
        
        with gr.Tabs():
            # 标签页1：音频切片
            with gr.Tab("音频切片"):
                gr.Markdown("### 音频自动切片工具")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        slice_input = gr.Textbox(
                            label="输入路径（文件或文件夹）",
                            placeholder="请输入音频文件或文件夹路径",
                        )
                        slice_output_dir = gr.Textbox(
                            label="输出目录",
                            value=SLICE_OUTPUT,
                        )
                        
                        with gr.Accordion("切片参数", open=False):
                            slice_threshold = gr.Slider(
                                label="音量阈值 (dB)",
                                minimum=-60,
                                maximum=-10,
                                value=DEFAULT_SLICE_PARAMS["threshold"],
                                step=1,
                            )
                            slice_min_length = gr.Number(
                                label="最小长度 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["min_length"],
                            )
                            slice_min_interval = gr.Number(
                                label="最小间隔 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["min_interval"],
                            )
                            slice_hop_size = gr.Number(
                                label="帧长度 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["hop_size"],
                            )
                            slice_max_sil_kept = gr.Number(
                                label="最大静音保留 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["max_sil_kept"],
                            )
                            slice_max = gr.Slider(
                                label="归一化最大值",
                                minimum=0.1,
                                maximum=1.0,
                                value=DEFAULT_SLICE_PARAMS["max"],
                                step=0.05,
                            )
                            slice_alpha = gr.Slider(
                                label="混音比例",
                                minimum=0.0,
                                maximum=1.0,
                                value=DEFAULT_SLICE_PARAMS["alpha"],
                                step=0.05,
                            )
                        
                        slice_button = gr.Button("开始切片", variant="primary")
                    
                    with gr.Column(scale=1):
                        slice_result = gr.Textbox(
                            label="处理结果",
                            lines=10,
                            interactive=False,
                        )
                        slice_output_path = gr.Textbox(
                            label="输出路径",
                            visible=False,
                        )
            
            # 标签页2：文本识别
            with gr.Tab("文本识别"):
                gr.Markdown("### ASR 文本识别工具")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        asr_input_folder = gr.Textbox(
                            label="输入文件夹（切片后的音频文件夹）",
                            placeholder="请输入音频文件夹路径",
                        )
                        asr_output_dir = gr.Textbox(
                            label="输出目录",
                            value=ASR_OUTPUT,
                        )
                        
                        asr_model = gr.Dropdown(
                            label="ASR 模型",
                            choices=list(asr_dict.keys()),
                            value=DEFAULT_ASR_CONFIG["default_model"],
                        )
                        
                        asr_language = gr.Dropdown(
                            label="语言",
                            choices=["auto", "zh", "en", "ja", "ko", "yue"],
                            value=DEFAULT_ASR_CONFIG["default_language"],
                        )
                        
                        asr_model_size = gr.Dropdown(
                            label="模型尺寸（仅 Faster Whisper）",
                            choices=asr_dict["Faster Whisper (多语种)"]["size"],
                            value=DEFAULT_ASR_CONFIG["default_model_size"],
                            visible=True,
                        )
                        
                        asr_precision = gr.Dropdown(
                            label="精度（仅 Faster Whisper）",
                            choices=["float32", "float16", "int8"],
                            value=DEFAULT_ASR_CONFIG["default_precision"],
                            visible=True,
                        )
                        
                        asr_button = gr.Button("开始识别", variant="primary")
                    
                    with gr.Column(scale=1):
                        asr_result = gr.Textbox(
                            label="识别结果",
                            lines=15,
                            interactive=False,
                        )
                        asr_output_path = gr.Textbox(
                            label="结果文件路径",
                            visible=False,
                        )
            
            # 标签页3：完整流程
            with gr.Tab("完整流程"):
                gr.Markdown("### 一键处理：切片 + 识别")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        pipeline_input = gr.Textbox(
                            label="输入路径（文件或文件夹）",
                            placeholder="请输入音频文件或文件夹路径",
                        )
                        
                        with gr.Row():
                            pipeline_slice_output = gr.Textbox(
                                label="切片输出目录",
                                value=SLICE_OUTPUT,
                            )
                            pipeline_asr_output = gr.Textbox(
                                label="识别输出目录",
                                value=ASR_OUTPUT,
                            )
                        
                        with gr.Accordion("切片参数", open=False):
                            pipeline_threshold = gr.Slider(
                                label="音量阈值 (dB)",
                                minimum=-60,
                                maximum=-10,
                                value=DEFAULT_SLICE_PARAMS["threshold"],
                                step=1,
                            )
                            pipeline_min_length = gr.Number(
                                label="最小长度 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["min_length"],
                            )
                            pipeline_min_interval = gr.Number(
                                label="最小间隔 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["min_interval"],
                            )
                            pipeline_hop_size = gr.Number(
                                label="帧长度 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["hop_size"],
                            )
                            pipeline_max_sil_kept = gr.Number(
                                label="最大静音保留 (毫秒)",
                                value=DEFAULT_SLICE_PARAMS["max_sil_kept"],
                            )
                            pipeline_max = gr.Slider(
                                label="归一化最大值",
                                minimum=0.1,
                                maximum=1.0,
                                value=DEFAULT_SLICE_PARAMS["max"],
                                step=0.05,
                            )
                            pipeline_alpha = gr.Slider(
                                label="混音比例",
                                minimum=0.0,
                                maximum=1.0,
                                value=DEFAULT_SLICE_PARAMS["alpha"],
                                step=0.05,
                            )
                        
                        with gr.Accordion("识别参数", open=False):
                            pipeline_asr_model = gr.Dropdown(
                                label="ASR 模型",
                                choices=list(asr_dict.keys()),
                                value=DEFAULT_ASR_CONFIG["default_model"],
                            )
                            pipeline_language = gr.Dropdown(
                                label="语言",
                                choices=["auto", "zh", "en", "ja", "ko", "yue"],
                                value=DEFAULT_ASR_CONFIG["default_language"],
                            )
                            pipeline_model_size = gr.Dropdown(
                                label="模型尺寸（仅 Faster Whisper）",
                                choices=asr_dict["Faster Whisper (多语种)"]["size"],
                                value=DEFAULT_ASR_CONFIG["default_model_size"],
                            )
                            pipeline_precision = gr.Dropdown(
                                label="精度（仅 Faster Whisper）",
                                choices=["float32", "float16", "int8"],
                                value=DEFAULT_ASR_CONFIG["default_precision"],
                            )
                        
                        pipeline_button = gr.Button("开始处理", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        pipeline_result = gr.Textbox(
                            label="处理结果",
                            lines=20,
                            interactive=False,
                        )
                        pipeline_slice_path = gr.Textbox(
                            label="切片输出路径",
                            visible=False,
                        )
                        pipeline_asr_path = gr.Textbox(
                            label="识别结果路径",
                            visible=False,
                        )
        
        # 绑定事件
        slice_button.click(
            fn=process_slice,
            inputs=[
                slice_input,
                slice_output_dir,
                slice_threshold,
                slice_min_length,
                slice_min_interval,
                slice_hop_size,
                slice_max_sil_kept,
                slice_max,
                slice_alpha,
            ],
            outputs=[slice_result, slice_output_path],
        )
        
        asr_button.click(
            fn=process_asr,
            inputs=[
                asr_input_folder,
                asr_output_dir,
                asr_model,
                asr_language,
                asr_model_size,
                asr_precision,
            ],
            outputs=[asr_result, asr_output_path],
        )
        
        pipeline_button.click(
            fn=process_full_pipeline,
            inputs=[
                pipeline_input,
                pipeline_slice_output,
                pipeline_asr_output,
                pipeline_asr_model,
                pipeline_language,
                pipeline_model_size,
                pipeline_precision,
                pipeline_threshold,
                pipeline_min_length,
                pipeline_min_interval,
                pipeline_hop_size,
                pipeline_max_sil_kept,
                pipeline_max,
                pipeline_alpha,
            ],
            outputs=[pipeline_result, pipeline_slice_path, pipeline_asr_path],
        )
        
        # 根据模型选择更新语言选项
        def update_language_options(model_name):
            if model_name in asr_dict:
                return {
                    "choices": asr_dict[model_name]["lang"],
                    "value": asr_dict[model_name]["lang"][0] if asr_dict[model_name]["lang"] else "auto"
                }
            return {"choices": [], "value": None}
        
        asr_model.change(
            fn=update_language_options,
            inputs=[asr_model],
            outputs=[asr_language],
        )
        
        pipeline_asr_model.change(
            fn=update_language_options,
            inputs=[pipeline_asr_model],
            outputs=[pipeline_language],
        )
    
    return app


def main():
    """主函数"""
    webui_config = config.get("webui", {})
    host = webui_config.get("host", "0.0.0.0")
    port = webui_config.get("port", 7860)
    share = webui_config.get("share", False)
    
    app = create_interface()
    app.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
