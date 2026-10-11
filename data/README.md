# data 目录

数据文件不放 git、不外传。按来源分子目录：

```
data/
├── benchmark/   # 清华 SSVEP Benchmark（公开练习数据，S01.mat ~ S35.mat）
└── official/    # 赛题官方测试数据集 / 正式数据集（自行放入，勿外传）
```

## 清华 SSVEP Benchmark（公开，用于练算法）

- 下载页：http://bci.med.tsinghua.edu.cn/download.html
- 内容：35 名被试，每人一个 `Sxx.mat`，约 3.3GB
- 格式：`data` 形状 `[64, 1500, 40, 6]` = 通道 × 采样点 × 40 个目标 × 6 个 block
- 采样率 250Hz，每 trial 6s，刺激频率 8.0~15.8Hz
- 下载后把 `S01.mat` 等文件直接放到 `data/benchmark/`

## 官方数据

- 备赛测试数据集：组委会在 QQ 群（100628255）发布，格式与正式赛完全一致
- 正式数据集：比赛 T0-4h 发布加密包，T0 公布解压密码
- 拿到后先运行：`python explore_eeg_data.py data/official -p -fs <采样率>`
