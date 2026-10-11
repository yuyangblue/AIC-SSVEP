# -*- coding: utf-8 -*-
"""TRCA 与 ensemble TRCA 分类器（监督，需要带标签训练数据）。

任务相关成分分析（Task-Related Component Analysis, Tanaka 2013；
Nakanishi et al. 2017 将 ensemble TRCA 用于 SSVEP，是传统方法中的强基线）。

原理：
    对类别 k 的 J 个训练片段 X_j ∈ R^{C×T}（去均值），学习空间滤波器 w_k，
    使不同 trial 滤波后信号的 trial 间协方差最大化：
        S_k = Σ_{j1≠j2} X_j1 X_j2' / T      （trial 间协方差）
        Q_k = Σ_j X_j X_j' / T              （trial 内协方差之和）
        Q_k^{-1} S_k w = λ w，取最大特征值对应特征向量即 w_k。
    测试片段 X 与类模板 T_k = mean_j X_j 经全部滤波器 W=[w_1..w_K] 投影后
    做相关（ensemble 策略），相关最大的类即为预测结果。
"""
import numpy as np

_EPS = 1e-8


def _trca_spatial_filter(trials):
    """对单类训练片段求 TRCA 空间滤波器。

    参数：trials，形状 (n_trial, n_channel, n_sample)
    返回：w，形状 (n_channel,)
    """
    trials = np.asarray(trials, dtype=float)
    n_trial, n_ch, n_samp = trials.shape
    trials = trials - trials.mean(axis=2, keepdims=True)

    s = np.zeros((n_ch, n_ch))
    q = np.zeros((n_ch, n_ch))
    for j in range(n_trial):
        q += trials[j] @ trials[j].T / n_samp
        for j2 in range(n_trial):
            if j2 != j:
                s += trials[j] @ trials[j2].T / n_samp
    q += _EPS * np.eye(n_ch)
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(q) @ s)
    w = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    w = w / (np.linalg.norm(w) + _EPS)
    return w


class TRCAClassifier:
    """ensemble TRCA 分类器。"""

    def __init__(self, n_classes=8, ensemble=True):
        """
        参数：
            n_classes 类别数
            ensemble  True 用 ensemble TRCA（所有类滤波器拼接投影后相关，
                      Nakanishi 2017，准确率更高）；False 用单类滤波器相关
        """
        self.n_classes = n_classes
        self.ensemble = ensemble
        self.filters_ = None     # (K, C)
        self.templates_ = None   # (K, C, T)

    def fit(self, trials, labels):
        """
        参数：
            trials 形状 (n_trial, C, T)
            labels 形状 (n_trial,)，取值 0~K-1
        """
        trials = np.asarray(trials, dtype=float)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        self.n_classes = len(classes)
        filters, templates = [], []
        for k in classes:
            xk = trials[labels == k]
            filters.append(_trca_spatial_filter(xk))
            templates.append(xk.mean(axis=0))
        self.filters_ = np.stack(filters, axis=0)          # (K, C)
        self.templates_ = np.stack(templates, axis=0)      # (K, C, T)
        return self

    @staticmethod
    def _corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b) + _EPS
        return float(a @ b / denom)

    def decision_scores(self, x):
        x = np.asarray(x, dtype=float)
        scores = np.zeros(self.n_classes)
        if self.ensemble:
            # 用全部类滤波器拼接投影：(T, K)
            w_all = self.filters_.T  # (C, K)
            rx = x.T @ w_all        # (T, K)
            for k in range(self.n_classes):
                rt = self.templates_[k].T @ w_all
                scores[k] = self._corr(rx, rt)
        else:
            for k in range(self.n_classes):
                w = self.filters_[k]
                scores[k] = self._corr(x.T @ w, self.templates_[k].T @ w)
        return scores

    def predict(self, x):
        return int(np.argmax(self.decision_scores(x)))

    def predict_batch(self, epochs):
        return np.array([self.predict(e) for e in epochs])
