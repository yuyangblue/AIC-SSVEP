# -*- coding: utf-8 -*-
"""模拟 SSVEP 数据自测：不依赖任何下载，验证 CCA / FBCCA / TRCA 实现是否正确。

运行：
    cd ~/workspace/aic
    python examples/demo_synthetic.py

数据设定（模拟）：
    8 个刺激频率（8~15Hz），9 个枕区通道，采样率 250Hz，每片段 3s；
    信号 = 基频 + 2/3 次谐波经固定空间混合 + 高斯噪声。
预期：CCA/FBCCA 接近满分，TRCA（4 trial 训练）也应很高。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssvep import CCAClassifier, FBCCAClassifier, TRCAClassifier
from ssvep.reference import DEMO_FREQS_8
from ssvep.preprocess import bandpass_filter


def make_synthetic(n_classes=8, n_ch=9, fs=250, duration=3.0,
                   n_trials=8, noise=0.6, seed=0):
    """生成模拟 SSVEP 数据。

    返回：
        trials (n_classes*n_trials, n_ch, n_sample)
        labels (...)
        freqs  (n_classes,)
    """
    rng = np.random.default_rng(seed)
    freqs = DEMO_FREQS_8[:n_classes]
    n_samp = int(fs * duration)
    t = np.arange(n_samp) / fs

    # 固定空间混合矩阵（谐波源 -> 通道），类别间共用，区分信息只在频率
    mixing = rng.normal(size=(n_ch, 3))
    phases = rng.uniform(0, 2 * np.pi, size=(n_classes, 3))

    trials, labels = [], []
    for k, f in enumerate(freqs):
        sources = np.vstack([
            np.sin(2 * np.pi * h * f * t + phases[k, h - 1])
            for h in (1, 2, 3)
        ])  # (3, T)
        clean = mixing @ sources  # (C, T)
        for _ in range(n_trials):
            x = clean + noise * rng.normal(size=(n_ch, n_samp))
            trials.append(x)
            labels.append(k)
    return np.array(trials), np.array(labels), freqs


def accuracy(pred, true):
    return float(np.mean(np.asarray(pred) == np.asarray(true))) * 100


def main():
    fs = 250
    trials, labels, freqs = make_synthetic(fs=fs)
    print(f"模拟数据：{len(freqs)} 类 × 8 trial，9 通道，250Hz，3s/片段")
    print(f"刺激频率：{list(freqs.astype(int))} Hz\n")

    # 预处理：统一宽频带通（6~80Hz）
    trials_bp = np.array([bandpass_filter(x, fs, 6.0, 80.0) for x in trials])

    # 1) CCA / FBCCA：无监督，全部片段测试
    cca = CCAClassifier(freqs, fs, n_harmonics=3)
    fbcca = FBCCAClassifier(freqs, fs, n_harmonics=3, n_subbands=5)
    pred_cca = cca.predict_batch(trials_bp)
    pred_fb = fbcca.predict_batch(trials_bp)
    print(f"CCA    正确率：{accuracy(pred_cca, labels):6.2f}%")
    print(f"FBCCA  正确率：{accuracy(pred_fb, labels):6.2f}%")

    # 2) TRCA：每类前 4 trial 训练，后 4 trial 测试
    n_train = 4
    train_x, train_y, test_x, test_y = [], [], [], []
    for k in range(len(freqs)):
        idx = np.where(labels == k)[0]
        # 每一类独立按 trial 序号切分，保证训练集和测试集都覆盖全部类别。
        train_x.extend(trials_bp[idx[:n_train]]); train_y.extend([k] * n_train)
        test_x.extend(trials_bp[idx[n_train:]]); test_y.extend([k] * (len(idx) - n_train))
    trca = TRCAClassifier(n_classes=len(freqs), ensemble=True)
    trca.fit(np.array(train_x), np.array(train_y))
    pred_trca = trca.predict_batch(np.array(test_x))
    print(f"TRCA   正确率：{accuracy(pred_trca, test_y):6.2f}%  "
          f"（每类 {n_train} trial 训练）")

    # 3) 结果文件生成演示
    from ssvep.submit import write_submission
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", "demo_result.xlsx")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_submission(pred_fb, out)

    print("\n自测完成。若 CCA/FBCCA 未接近满分，先检查采样率、频率表与分段长度。")


if __name__ == "__main__":
    main()
