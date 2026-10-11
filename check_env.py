# -*- coding: utf-8 -*-
"""环境检查：确认运行本工程所需的科学计算包是否齐全。

用法：
    python check_env.py
"""
import sys


def main():
    print("Python:", sys.version.split()[0], "|", sys.executable)
    ok = True
    for m in ["numpy", "scipy", "pandas", "matplotlib", "openpyxl"]:
        try:
            mod = __import__(m)
            print(f"{m:12s} OK   {getattr(mod, '__version__', '?')}")
        except ImportError:
            ok = False
            print(f"{m:12s} MISSING  <-- pip install {m}")
    try:
        import torch  # 仅深度学习方案需要，非必需
        print(f"{'torch':12s} OK   {torch.__version__} (GPU: {torch.cuda.is_available()})")
    except ImportError:
        print(f"{'torch':12s} 未安装（仅深度学习方案需要，CCA/FBCCA/TRCA 不需要）")
    print("\n环境检查通过。" if ok else "\n缺少依赖，请先 pip install。")


if __name__ == "__main__":
    main()
