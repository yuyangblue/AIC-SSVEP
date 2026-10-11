# -*- coding: utf-8 -*-
"""预处理：带通滤波、工频陷波、滤波器组、分段、归一化。

全部基于 scipy.signal 自主实现，零阶相位失真滤波（filtfilt）。
"""
import numpy as np
from scipy import signal


def _butter_bandpass_coeff(fs, low, high, order=4):
    """设计 Butterworth 带通滤波器系数，high 自动限制在奈奎斯特频率以内。"""
    nyq = fs / 2.0
    low = max(low, 0.5)
    high = min(high, nyq - 1.0)
    if low >= high:
        raise ValueError(f"无效通带 [{low}, {high}] Hz（奈奎斯特频率 {nyq} Hz）")
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return b, a


def bandpass_filter(x, fs, low, high, order=4, axis=-1):
    """零相位 Butterworth 带通滤波。

    参数：
        x     一维/多维 EEG，时间维由 axis 指定
        fs    采样率
        low   通带下限（Hz）
        high  通带上限（Hz）
    """
    b, a = _butter_bandpass_coeff(fs, low, high, order)
    # padlen 保护：数据太短时减小 pad
    padlen = min(3 * max(len(a), len(b)), x.shape[axis] - 1)
    return signal.filtfilt(b, a, x, axis=axis, padlen=padlen)


def notch_filter(x, fs, f0=50.0, q=30.0, axis=-1):
    """工频陷波（默认 50Hz）。"""
    w0 = f0 / (fs / 2.0)
    if w0 >= 1.0:
        return x
    b, a = signal.iirnotch(w0, q)
    padlen = min(3 * max(len(a), len(b)), x.shape[axis] - 1)
    return signal.filtfilt(b, a, x, axis=axis, padlen=padlen)


def filterbank(x, fs, bands=None, order=4, axis=-1):
    """滤波器组：对 x 分别做多个子带带通滤波。

    参数：
        bands 子带通带列表，默认采用 Chen et al.(2015) FBCCA 经典设置：
              第 n 个子带为 [8n, 90] Hz，n=1..5，即
              (8,90),(16,90),(24,90),(32,90),(40,90)
              分别覆盖基频、2 次谐波、3 次谐波……
    返回：
        list[np.ndarray]，每个元素形状与 x 相同
    """
    if bands is None:
        bands = [(8.0 * n, 90.0) for n in range(1, 6)]
    out = []
    for low, high in bands:
        try:
            out.append(bandpass_filter(x, fs, low, high, order=order, axis=axis))
        except ValueError:
            break  # 通带超出奈奎斯特频率则停止增加子带
    return out


def zscore(x, axis=-1, eps=1e-12):
    """逐通道（逐行）z-score 标准化。"""
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def epoch(continuous, triggers, fs, offset=0.14, duration=2.0, ch_axis=0):
    """从连续 EEG 记录中按触发点切出每次实验的信号片段。

    比赛若给的是连续记录（而非已分段 trial），用本函数分段。
    参数：
        continuous 连续信号，形状 (通道, 采样点)（ch_axis=0）
        triggers   触发时刻列表（秒），即每次刺激开始时刻
        fs         采样率
        offset     刺激开始后跳过的视觉延迟（秒），SSVEP 通常取 0.14
        duration   每个片段时长（秒）
        ch_axis    通道维位置
    返回：
        epochs，形状 (n_trial, 通道, 采样点)
    """
    if ch_axis != 0:
        continuous = np.moveaxis(continuous, ch_axis, 0)
    n_ch, n_total = continuous.shape
    s0 = int(round(offset * fs))
    win = int(round(duration * fs))
    epochs = []
    for t in triggers:
        start = int(round(t * fs)) + s0
        if start < 0 or start + win > n_total:
            continue
        epochs.append(continuous[:, start:start + win])
    return np.stack(epochs, axis=0)
