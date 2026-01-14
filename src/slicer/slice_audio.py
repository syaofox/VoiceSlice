"""音频切片处理脚本"""

import os
import numpy as np
import traceback
from scipy.io import wavfile

from ..utils.audio_utils import load_audio
from .slicer import Slicer


def slice_audio(
    inp,
    opt_root,
    threshold=-34,
    min_length=4000,
    min_interval=300,
    hop_size=10,
    max_sil_kept=500,
    _max=0.9,
    alpha=0.25,
    i_part=0,
    all_part=1,
):
    """
    对音频文件或文件夹进行切片处理
    
    Args:
        inp: 输入文件或文件夹路径
        opt_root: 输出根目录
        threshold: 音量阈值（dB），小于此值视为静音
        min_length: 每段最小长度（毫秒）
        min_interval: 最短切割间隔（毫秒）
        hop_size: 帧长度（毫秒）
        max_sil_kept: 切完后静音最多保留长度（毫秒）
        _max: 归一化后最大值
        alpha: 混音比例
        i_part: 当前处理的批次索引（用于多进程）
        all_part: 总批次数（用于多进程）
        
    Returns:
        str: 处理结果消息
    """
    os.makedirs(opt_root, exist_ok=True)
    if os.path.isfile(inp):
        input_files = [inp]
    elif os.path.isdir(inp):
        input_files = [os.path.join(inp, name) for name in sorted(list(os.listdir(inp)))]
    else:
        return "输入路径存在但既不是文件也不是文件夹"
    
    slicer = Slicer(
        sr=32000,  # 长音频采样率
        threshold=int(threshold),  # 音量小于这个值视作静音的备选切割点
        min_length=int(min_length),  # 每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值
        min_interval=int(min_interval),  # 最短切割间隔
        hop_size=int(hop_size),  # 怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）
        max_sil_kept=int(max_sil_kept),  # 切完后静音最多留多长
    )
    _max = float(_max)
    alpha = float(alpha)
    
    # 处理指定批次的文件
    for inp_path in input_files[int(i_part) :: int(all_part)]:
        try:
            name = os.path.basename(inp_path)
            audio = load_audio(inp_path, 32000)
            for chunk, start, end in slicer.slice(audio):  # start和end是帧数
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk /= tmp_max
                if tmp_max > 0:
                    chunk = (chunk / tmp_max * (_max * alpha)) + (1 - alpha) * chunk
                else:
                    chunk = chunk * _max
                wavfile.write(
                    "%s/%s_%010d_%010d.wav" % (opt_root, name, start, end),
                    32000,
                    (chunk * 32767).astype(np.int16),
                )
        except Exception as e:
            print(f"{inp_path} ->fail-> {traceback.format_exc()}")
    
    return "执行完毕，请检查输出文件"
