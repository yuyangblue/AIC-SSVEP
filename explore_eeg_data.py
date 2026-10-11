# -*- coding: utf-8 -*-
"""
赛题数据通用探查脚本（纯 numpy / scipy / pandas 实现，不含任何现成分类算法）

用途：
    拿到测试/正式数据后，先跑这个脚本摸清数据长什么样：
      1. 文件格式（.mat / .npy / .npz / .csv / .txt / .zip）
      2. 每个数组的形状、数据类型、数值范围（判断 通道x时间点 的排列）
      3. 采样率等元信息（找 fs / srate / sample 之类字段）
      4. 可选：对候选的 [通道, 时间点] 数组做 FFT，输出频谱 PNG，直接看刺激频率峰

用法：
    python explore_eeg_data.py <文件或目录路径>          # 探查目录下所有数据文件
    python explore_eeg_data.py xxx.zip -p                # 解压后探查，并画频谱预览
    python explore_eeg_data.py data.mat -p -fs 250       # 手动指定采样率（Hz）
"""
import os
import sys
import glob
import zipfile
import tempfile

import numpy as np
from scipy import io as sio

EEG_CH_HINT = ["O1", "O2", "Oz", "POz", "PO3", "PO4", "PO5", "PO6",
               "PO7", "PO8", "Pz", "P3", "P4", "Cz", "FP1", "FP2"]
FS_FIELDS = ["fs", "srate", "sample_rate", "sampling_rate", "Fs", "freq", "frequency"]


def print_title(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


def probe_ndarray(name, arr, fs_hint=None, do_plot=False):
    a = np.asarray(arr)
    print(f"\n  [{name}]  shape={a.shape}  dtype={a.dtype}")
    if a.size == 0:
        print("    空数组，跳过")
        return None
    if a.dtype.kind in "USO":
        show = [str(x) for x in a.ravel()[:12]]
        print(f"    内容(前12): {show}")
        hit = [x for x in show if x in EEG_CH_HINT]
        if hit:
            print(f"    ★ 含电极名: {hit} —— 很可能是通道列表")
        return None
    try:
        print(f"    min={np.nanmin(a):.3f}  max={np.nanmax(a):.3f}  "
              f"mean={np.nanmean(a):.3f}  std={np.nanstd(a):.3f}")
    except Exception:
        pass
    if a.ndim == 0 or a.size == 1:
        print(f"    标量: {np.asarray(a).reshape(-1)[0]}")
        return float(np.asarray(a).reshape(-1)[0])
    if a.ndim == 2:
        print("    二维数组：可能是 [通道, 时间点] 或 [时间点, 通道]")
    if a.ndim == 3:
        print("    三维数组：可能是 [trial, 通道, 时间点] 等排列")
    if a.ndim == 4:
        print("    四维数组：可能是 [通道, 时间点, 目标, block]（清华 benchmark 惯例）")

    if do_plot and a.ndim == 2 and fs_hint:
        row = a[0] if min(a.shape) == a.shape[0] and a.shape[0] <= 64 else a[:, 0]
        row = np.asarray(row, dtype=float).ravel()
        row = row - row.mean()
        n = len(row)
        if n >= 64:
            spec = np.abs(np.fft.rfft(row * np.hanning(n))) ** 2
            freqs = np.fft.rfftfreq(n, 1.0 / fs_hint)
            idx = (freqs >= 1) & (freqs <= 60)
            top = np.argsort(spec[idx])[-5:][::-1]
            print("    频谱峰(1-60Hz) Top5: " +
                  ", ".join(f"{freqs[idx][i]:.2f}Hz" for i in top))
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.figure(figsize=(7, 3.5))
                plt.plot(freqs[idx], spec[idx])
                plt.xlabel("Hz"); plt.ylabel("PSD")
                out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "explore_spec.png")
                plt.title(f"{name} channel-0 spectrum (fs={fs_hint}Hz)")
                plt.tight_layout(); plt.savefig(out, dpi=110)
                print(f"    频谱图已保存: {out}")
            except Exception as e:
                print(f"    画图失败(不影响探查): {e}")
    return None


def probe_file(path, fs_hint=None, do_plot=False):
    ext = os.path.splitext(path)[1].lower()
    base = os.path.basename(path)
    if ext == ".mat":
        print_title(f"MATLAB 文件: {base}")
        try:
            d = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
        except Exception as e:
            print(f"  读取失败: {e}"); return
        for k, v in d.items():
            if k.startswith("__") and k.endswith("__"):
                continue
            found = probe_ndarray(k, v, fs_hint, do_plot)
            if found is not None and any(fk in k.lower() for fk in FS_FIELDS):
                fs_hint = found
        if fs_hint:
            print(f"\n  → 采样率(从文件推断): {fs_hint} Hz")
    elif ext == ".npz":
        print_title(f"NumPy 压缩包: {base}")
        try:
            with np.load(path, allow_pickle=True) as d:
                for k in d.files:
                    found = probe_ndarray(k, d[k], fs_hint, do_plot)
                    if found is not None and any(fk in k.lower() for fk in FS_FIELDS):
                        fs_hint = found
        except Exception as e:
            print(f"  读取失败: {e}"); return
        if fs_hint:
            print(f"\n  → 采样率(从文件推断): {fs_hint} Hz")
    elif ext == ".npy":
        print_title(f"NumPy 数组: {base}")
        try:
            probe_ndarray(base, np.load(path, allow_pickle=True), fs_hint, do_plot)
        except Exception as e:
            print(f"  读取失败: {e}")
    elif ext in (".csv", ".txt", ".dat"):
        print_title(f"文本表格: {base}")
        try:
            import pandas as pd
            df = pd.read_csv(path, header=None)
            head = [str(x) for x in df.iloc[0].tolist()]
            hit = [x for x in head if x in EEG_CH_HINT]
            if hit:
                df = pd.read_csv(path)
                print(f"  ★ 首行是通道名 {hit}，已按 header=0 重读")
            print(f"  shape={df.shape}  前3行预览:")
            print(df.head(3).to_string(max_colwidth=12))
            if df.shape[0] > 1000 and df.shape[1] <= 128:
                print(f"  提示：{df.shape[1]} 列很可能是通道数，行数为时间点")
        except Exception as e:
            print(f"  读取失败: {e}")
    elif ext == ".zip":
        print_title(f"ZIP 压缩包: {base}")
        tmp = tempfile.mkdtemp(prefix="aic_probe_")
        try:
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
            print(f"  已解压到 {tmp}，继续探查：")
            walk(tmp, fs_hint, do_plot)
        except Exception as e:
            print(f"  解压失败: {e}")
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"  未知扩展名 {ext}，前 64 字节：")
        try:
            with open(path, "rb") as f:
                print("   ", f.read(64)[:32].hex(" "))
        except Exception as e:
            print(f"    读取失败: {e}")


def walk(root, fs_hint=None, do_plot=False):
    if os.path.isfile(root):
        probe_file(root, fs_hint, do_plot); return
    exts = (".mat", ".npy", ".npz", ".csv", ".txt", ".dat", ".zip")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, "**", "*" + e), recursive=True))
    if not files:
        print(f"目录 {root} 下未找到 {exts} 文件"); return
    for f in sorted(files):
        probe_file(f, fs_hint, do_plot)


def main():
    argv = sys.argv[1:]
    fs_hint, args = None, []
    i = 0
    while i < len(argv):
        if argv[i] == "-fs" and i + 1 < len(argv):
            try:
                fs_hint = float(argv[i + 1]); i += 2; continue
            except ValueError:
                pass
        args.append(argv[i]); i += 1
    do_plot = any(x in ("-p", "--plot") for x in args)
    args = [a for a in args if not a.startswith("-")]
    if not args:
        print(__doc__); sys.exit(1)
    for a in args:
        if not os.path.exists(a):
            print(f"路径不存在: {a}"); continue
        print(f"手动指定采样率: {fs_hint} Hz" if fs_hint else "")
        walk(a, fs_hint, do_plot)
    print("\n完成。")


if __name__ == "__main__":
    main()
