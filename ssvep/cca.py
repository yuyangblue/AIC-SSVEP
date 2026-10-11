# -*- coding: utf-8 -*-
"""CCA 与 FBCCA 分类器（无监督，无需训练数据）。

数学原理（典型相关分析，Lin et al. 2006 用于 SSVEP）：
    对 EEG 片段 X ∈ R^{C×T} 与频率 f 的参考信号 Y_f ∈ R^{2Nh×T}，
    设联合协方差
        Sxx = XX'/T,  Syy = Y_f Y_f'/T,  Sxy = X Y_f'/T
    求解广义特征问题
        Sxx^{-1} Sxy Syy^{-1} Syx w = ρ² w
    最大典型相关系数 ρ_max = sqrt(最大特征值)。
    取 ρ_max 最大的候选频率作为预测类别。

FBCCA（Chen et al., 2015）：先把信号分成多个子带分别做 CCA，
再按  w_n = n^{-1.25} + 0.25  加权融合各子带相关系数的平方。
"""
import numpy as np

from .reference import reference_signal
from .preprocess import filterbank

_EPS = 1e-8


def cca_max_corr(x, y, n_components=1):
    """计算两组多通道信号 x(C×T) 与 y(Q×T) 的前 n_components 个典型相关系数。

    返回：np.ndarray，按从大到小排列的相关系数（取实部并裁剪到 [0,1]）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)

    nx = x.shape[0]
    z = np.vstack([x, y])
    cov = np.cov(z)
    sxx = cov[:nx, :nx] + _EPS * np.eye(nx)
    syy = cov[nx:, nx:] + _EPS * np.eye(cov.shape[0] - nx)
    sxy = cov[:nx, nx:]

    # M 与对称矩阵相似，特征值为非负实数；用 eig 求广义特征值更稳
    m = np.linalg.inv(sxx) @ sxy @ np.linalg.inv(syy) @ sxy.T
    eigvals = np.linalg.eigvals(m)
    corr = np.sqrt(np.clip(np.real(eigvals), 0.0, 1.0))
    return np.sort(corr)[::-1][:n_components]


class CCAClassifier:
    """标准 CCA 分类器：构造时给定候选频率，预测时直接比对，无需 fit。"""

    def __init__(self, freqs, fs, n_harmonics=3, n_components=1,
                 lowcut=6.0, highcut=None):
        """
        参数：
            freqs        候选刺激频率数组（长度 = 类别数，如 8 类）
            fs           采样率
            n_harmonics  参考信号谐波个数（含基频）
            n_components 每个频率取前几个典型相关（默认 1）
            lowcut/highcut 预测前对 EEG 做一次宽频带通的范围（Hz），
                           highcut 默认 min(90, fs/2-1)
        """
        self.freqs = np.asarray(freqs, dtype=float)
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.n_components = n_components
        self.lowcut = lowcut
        self.highcut = highcut

    def _prefilter(self, x):
        if self.lowcut is None:
            return x
        from .preprocess import bandpass_filter
        high = self.highcut if self.highcut else min(90.0, self.fs / 2 - 1.0)
        return bandpass_filter(x, self.fs, self.lowcut, high)

    def decision_scores(self, x):
        """返回每个候选频率的相关得分，形状 (n_freqs,)。"""
        x = self._prefilter(np.asarray(x, dtype=float))
        n_samples = x.shape[-1]
        scores = np.zeros(len(self.freqs))
        for k, f in enumerate(self.freqs):
            y = reference_signal(f, self.fs, n_samples, self.n_harmonics)
            rhos = cca_max_corr(x, y, self.n_components)
            scores[k] = np.sum(rhos ** 2)
        return scores

    def predict(self, x):
        """对单个片段 x(C×T) 预测类别编号（0~K-1）。"""
        return int(np.argmax(self.decision_scores(x)))

    def predict_batch(self, epochs):
        """对多个片段 (n_trial, C, T) 批量预测。"""
        return np.array([self.predict(e) for e in epochs])


class FBCCAClassifier:
    """滤波器组 CCA：多子带分别 CCA 后加权融合（通常显著优于单带 CCA）。"""

    def __init__(self, freqs, fs, n_harmonics=3, n_subbands=5,
                 bands=None, weight_a=1.25, weight_b=0.25):
        """
        参数：
            n_subbands 子带个数（默认 5）
            bands      自定义子带通带列表；默认第 n 带 [8n, 90]Hz
            weight_a/b 融合权重 w_n = n^{-a} + b（论文默认 a=1.25, b=0.25）
        """
        self.freqs = np.asarray(freqs, dtype=float)
        self.fs = fs
        self.n_harmonics = n_harmonics
        if bands is None:
            bands = [(8.0 * n, 90.0) for n in range(1, n_subbands + 1)]
        self.bands = bands
        self.weights = np.array(
            [n ** (-weight_a) + weight_b for n in range(1, len(bands) + 1)]
        )

    def decision_scores(self, x):
        x = np.asarray(x, dtype=float)
        n_samples = x.shape[-1]
        sub_signals = filterbank(x, self.fs, self.bands)
        n_bands = len(sub_signals)
        scores = np.zeros(len(self.freqs))
        for k, f in enumerate(self.freqs):
            y = reference_signal(f, self.fs, n_samples, self.n_harmonics)
            rho2 = np.zeros(n_bands)
            for nb, xb in enumerate(sub_signals):
                rho = cca_max_corr(xb, y, 1)[0]
                rho2[nb] = rho ** 2
            scores[k] = np.sum(self.weights[:n_bands] * rho2)
        return scores

    def predict(self, x):
        return int(np.argmax(self.decision_scores(x)))

    def predict_batch(self, epochs):
        return np.array([self.predict(e) for e in epochs])
