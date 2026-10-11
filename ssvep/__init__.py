# -*- coding: utf-8 -*-
"""ssvep：自主实现的 SSVEP 特征提取与分类算法包（仅依赖 numpy/scipy/pandas）。

模块：
    reference   正余弦参考信号、刺激频率配置
    preprocess  带通/陷波滤波、滤波器组、分段、归一化
    cca         CCA 与 FBCCA 分类器（无监督，无需训练数据）
    trca        TRCA 与 ensemble TRCA 分类器（监督，需要带标签训练数据）
    submit      标准结果 Excel 生成
"""
from .reference import reference_signal, BENCHMARK_FREQS
from .preprocess import bandpass_filter, notch_filter, filterbank, zscore, epoch
from .cca import CCAClassifier, FBCCAClassifier
from .trca import TRCAClassifier

__all__ = [
    "reference_signal", "BENCHMARK_FREQS",
    "bandpass_filter", "notch_filter", "filterbank", "zscore", "epoch",
    "CCAClassifier", "FBCCAClassifier", "TRCAClassifier",
]
