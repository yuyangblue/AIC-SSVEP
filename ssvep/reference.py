# -*- coding: utf-8 -*-
"""参考信号与刺激频率配置。

SSVEP 的 CCA 类方法原理：对每个候选刺激频率 f，构造一组正余弦参考信号
    Y_f(t) = [sin(2πft), cos(2πft), sin(2π·2ft), cos(2π·2ft), ...]
若被试注视频率 f 的目标，EEG 与 Y_f 的典型相关系数最大，由此判定类别。
"""
import numpy as np

# 清华 SSVEP Benchmark（Wang et al., 2017）40 个刺激频率（Hz）。
# 排列为 8 行 5 列：第 r 行第 c 列频率 = 8 + r + 0.2*c，范围 8.0~15.8Hz。
# 正式比赛若给出官方频率表，以官方为准，覆盖此变量即可。
BENCHMARK_FREQS = np.array(
    [8.0 + (k // 5) + 0.2 * (k % 5) for k in range(40)]
)

# 演示/自测用的 8 类频率（Hz），间隔 1Hz。正式比赛以官方频率表替换。
DEMO_FREQS_8 = np.arange(8.0, 16.0, 1.0)


def reference_signal(freq, fs, n_samples, n_harmonics=3, delay=0.0):
    """生成单个刺激频率的正余弦参考信号矩阵。

    参数：
        freq        刺激频率（Hz）
        fs          采样率（Hz）
        n_samples   信号片段长度（采样点数）
        n_harmonics 谐波个数（含基频），常用 3~5
        delay       视觉延迟（秒），默认 0；分段时通常已统一扣除 0.14s
    返回：
        Y，形状 (2*n_harmonics, n_samples)，行依次为
        sin(2πft),cos(2πft),sin(4πft),cos(4πft),...
    """
    # 采样时刻从 0 开始；所有候选频率共用同一时间基准。
    t = np.arange(n_samples) / fs + delay
    rows = []
    for h in range(1, n_harmonics + 1):
        rows.append(np.sin(2 * np.pi * h * freq * t))
        rows.append(np.cos(2 * np.pi * h * freq * t))
    return np.vstack(rows)
