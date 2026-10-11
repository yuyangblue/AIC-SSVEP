# -*- coding: utf-8 -*-
"""在清华 SSVEP Benchmark 数据集上评估 CCA / FBCCA / TRCA。

数据获取（免费，科研用途）：
    1. 打开 http://bci.med.tsinghua.edu.cn/download.html
    2. 下载 "Benchmark dataset"（35 个 S01.mat~S35.mat，约 3.3GB）
    3. 把 .mat 文件放到本工程 data/benchmark/ 目录

数据格式（Wang et al., 2017）：
    data 形状 [64, 1500, 40, 6] = 通道 × 采样点 × 目标(40类) × block(6)
    采样率 250Hz，每 trial 6s；视觉延迟 0.14s（35 个采样点）。

运行：
    python examples/run_benchmark.py --subject 1 --tw 1.5
    （--tw 为分析窗口长度秒，越短越接近实战 ITR 场景）
"""
import argparse
import os
import sys
import numpy as np
from scipy import io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssvep import CCAClassifier, FBCCAClassifier, TRCAClassifier
from ssvep.reference import BENCHMARK_FREQS

# 9 个枕顶区通道在 64 导中的 0-based 索引：
# Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
OCCIPITAL_9 = [47, 53, 54, 55, 56, 57, 60, 61, 62]
FS = 250
LATENCY = 0.14  # 视觉延迟（秒）


def load_subject(path):
    mat = sio.loadmat(path, squeeze_me=True)
    data = mat["data"]  # [64, 1500, 40, 6]
    freqs = np.asarray(mat.get("freqs", BENCHMARK_FREQS), dtype=float).ravel()
    return data, freqs


def build_epochs(data, freqs, tw, channels=OCCIPITAL_9):
    """切出 (n_trial, C, T) 与标签，窗口 [0.14, 0.14+tw]。"""
    s0 = int(round(LATENCY * FS))
    win = int(round(tw * FS))
    data = data[channels, s0:s0 + win, :, :]  # [C, T, 40, 6]
    n_ch, n_samp, n_target, n_block = data.shape
    # 调整为 block 在前、目标在后，再展平成 trial 维，便于后续分折。
    x = np.transpose(data, (3, 2, 0, 1)).reshape(n_block * n_target, n_ch, n_samp)
    y = np.tile(np.arange(n_target), n_block)
    return x, y, n_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, help="被试编号 1~35")
    ap.add_argument("--tw", type=float, default=1.5, help="分析窗口长度（秒）")
    ap.add_argument("--n-classes", type=int, default=40,
                    help="评估类别数，默认 40；可设 8 模拟比赛（取前 8 个频率）")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "data", "benchmark", f"S{args.subject:02d}.mat")
    if not os.path.exists(path):
        print(f"未找到 {path}\n请先从 http://bci.med.tsinghua.edu.cn/download.html "
              f"下载 Benchmark dataset 并放到 data/benchmark/")
        return

    data, freqs = load_subject(path)
    x, y, n_target = build_epochs(data, freqs, args.tw)
    if args.n_classes < n_target:
        keep = y < args.n_classes
        x, y = x[keep], y[keep]
        freqs = freqs[:args.n_classes]
    print(f"被试 S{args.subject:02d} | 窗口 {args.tw}s | {len(freqs)} 类 | "
          f"{x.shape[1]} 通道 | {len(y)} 个 trial")

    cca = CCAClassifier(freqs, FS, n_harmonics=3)
    fbcca = FBCCAClassifier(freqs, FS, n_harmonics=3, n_subbands=5)
    acc_cca = (cca.predict_batch(x) == y).mean() * 100
    acc_fb = (fbcca.predict_batch(x) == y).mean() * 100
    print(f"CCA    正确率：{acc_cca:6.2f}%")
    print(f"FBCCA  正确率：{acc_fb:6.2f}%")

    # TRCA：留一 block 交叉验证（每轮 5 个 block 训练，1 个 block 测试）
    n_block = 6
    n_target = len(freqs)
    # 还原 block 维度：x 排列为 block 外循环、target 内循环
    x_blocks = x.reshape(n_block, n_target, *x.shape[1:])
    y_blocks = y.reshape(n_block, n_target)
    accs = []
    for b in range(n_block):
        # 留出一个完整 block 测试，避免同一 block 的试次同时进入训练和测试。
        test_idx = [b]
        train_idx = [i for i in range(n_block) if i != b]
        tr_x = x_blocks[train_idx].reshape(-1, *x.shape[1:])
        tr_y = np.tile(np.arange(n_target), len(train_idx))
        te_x = x_blocks[test_idx].reshape(-1, *x.shape[1:])
        te_y = y_blocks[test_idx].ravel()
        trca = TRCAClassifier(n_classes=n_target, ensemble=True).fit(tr_x, tr_y)
        accs.append((trca.predict_batch(te_x) == te_y).mean() * 100)
    print(f"TRCA   正确率：{np.mean(accs):6.2f}%  （6 折留一 block 交叉验证）")


if __name__ == "__main__":
    main()
