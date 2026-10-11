# AIC 脑电信号特征提取与识别（SSVEP 8 分类）备赛工程

全球校园人工智能算法精英大赛 · 算法应用赛（赛马制）· 赛题 13。
任务：对真人稳态视觉诱发电位（SSVEP）脑电数据做 **8 类分类（类别 0~7）**，
把每个信号片段的识别结果填入赛题标准 Excel。

成绩 = 算法任务分（识别正确率 ACC × 60，满分 60）+ 技术报告分（满分 40）。

---

## 一、目录结构

```
aic/
├── README.md                 # 本文件：总说明、算法原理、比赛流程
├── requirements.txt          # 依赖清单（d2l 环境已自带）
├── check_env.py              # 环境检查
├── explore_eeg_data.py       # 数据通用探查（mat/npy/npz/csv/zip，含频谱预览）
├── notebooks/                # 5 篇循序渐进的 Jupyter 教学笔记（已运行，含图）
│   ├── 01_SSVEP信号与预处理.ipynb
│   ├── 02_CCA典型相关分析.ipynb
│   ├── 03_FBCCA滤波器组CCA.ipynb
│   ├── 04_TRCA任务相关成分分析.ipynb
│   └── 05_综合对比与参数调优.ipynb
├── ssvep/                    # 自主实现的算法包（仅依赖 numpy/scipy/pandas）
│   ├── reference.py          #   正余弦参考信号、刺激频率表
│   ├── preprocess.py         #   带通/陷波滤波、滤波器组、分段、归一化
│   ├── cca.py                #   CCA 与 FBCCA（无监督基线，无需训练）
│   ├── trca.py               #   TRCA / ensemble TRCA（监督，传统方法强基线）
│   └── submit.py             #   标准结果 Excel 生成
├── examples/
│   ├── demo_synthetic.py     # 模拟数据自测（无需下载，验证算法正确性）
│   └── run_benchmark.py      # 清华 SSVEP Benchmark 评估脚本
├── data/
│   ├── benchmark/            # 公开练习数据（自行下载，见 data/README.md）
│   └── official/             # 官方测试/正式数据（自行放入，勿外传）
├── papers/                   # 论文索引；本地 PDF 不提交（先看 papers/README.md）
│   ├── README.md             #   论文分类、推荐阅读顺序、付费文献官方链接、撤稿声明
│   └── verify_pdfs.py        #   PDF 完整性校验脚本（python verify_pdfs.py）
└── output/                   # 预测结果输出
```

---

## 二、环境配置

推荐直接使用 WSL 中已有的 conda `d2l` 环境（Python 3.11，已含
numpy / scipy / pandas / matplotlib / openpyxl / torch GPU）：

```bash
conda activate d2l
cd ~/workspace/aic
python check_env.py
```

新建环境时：

```bash
pip install -r requirements.txt
```

---

## 三、快速开始

```bash
# 0) 推荐先看教学笔记（5 篇，从零讲清原理与实现）
jupyter lab notebooks/          # 或用 VSCode 直接打开 notebooks/ 下的 .ipynb

# 0.5) 配合论文深入：papers/README.md 给了分类索引与阅读顺序
#      综述入门 → CCA/FBCCA/TRCA 原文 → 数据集论文 → Transformer 最新进展
ls papers/

# 1) 模拟数据自测：验证 CCA/FBCCA/TRCA（约 10 秒，无需任何数据）
python examples/demo_synthetic.py

# 2) 下载清华 Benchmark 放到 data/benchmark/ 后，评估公开数据正确率
python examples/run_benchmark.py --subject 1 --tw 1.5
#    只评 8 类（模拟比赛）：--n-classes 8

# 3) 拿到官方数据后，先探查结构（格式、维度、采样率、频谱峰）
python explore_eeg_data.py data/official -p -fs 250
```

---

## 四、教学笔记（Jupyter）

`notebooks/` 下 5 篇笔记建议按顺序学习，每篇都含原理推导、**从零实现**、
可视化和小结，且已在 d2l 环境实际运行（输出与图片已保存，打开即看）：

1. **01_SSVEP信号与预处理**：基频/谐波、FFT 频谱、多通道混合、带通/陷波/
   分段（0.14 s 延迟）/z-score。
2. **02_CCA典型相关分析**：从皮尔逊相关到广义特征值推导，手写 CCA，
   参考信号、打分图、混淆矩阵、谐波数与谐波混叠。
3. **03_FBCCA滤波器组CCA**：子带设计、各子带相关、权重融合、噪声鲁棒性，
   并说明模拟数据与真实数据的差异。
4. **04_TRCA任务相关成分分析**：最大化 trial 间一致性、空间滤波器推导、
   类模板匹配、ensemble TRCA、留一 block 交叉验证、训练量影响。
5. **05_综合对比与参数调优**：窗口长度、谐波数、ITR 信息传输率、混淆矩阵、
   噪声鲁棒性、比赛调参清单与深度学习方向。

---

## 五、SSVEP 与算法原理

被试注视以固定频率 f 闪烁的目标时，枕区 EEG 会出现频率 f 及其谐波
（2f、3f…）的周期响应。8 个目标对应 8 个频率，分类即"找信号里的主导频率"。

### 1. CCA（典型相关分析，无监督基线）

对 EEG 片段 X ∈ R^{C×T} 与频率 f 的参考信号

$$Y_f = [\sin(2\pi ft), \cos(2\pi ft), \sin(4\pi ft), \cos(4\pi ft), \ldots]$$

求最大典型相关系数 ρ：解广义特征问题

$$S_{xx}^{-1} S_{xy} S_{yy}^{-1} S_{yx} w = \rho^2 w$$

取 ρ 最大的候选频率为预测类别。**无需任何训练数据，秒级出结果，是保底方案。**

### 2. FBCCA（滤波器组 CCA，无监督，显著更强）

先把信号分成多个子带（默认第 n 带 [8n, 90] Hz，覆盖基频与各次谐波），
每个子带分别做 CCA，再按论文权重融合：

$$\hat{\rho}_k = \sum_{n=1}^{N} w_n \rho_{n,k}^2, \qquad w_n = n^{-1.25} + 0.25$$

### 3. TRCA / ensemble TRCA（监督，传统方法最强基线）

用每类的多个带标签训练片段学习空间滤波器 w，使 trial 间任务相关成分
最大化（Q⁻¹S 的最大特征向量）；测试片段与类模板经滤波器组投影后比相关。
**需要带标签训练数据**，准确率通常高于 FBCCA。

### 4. 深度学习（报告创新点，可选）

EEGNet / Conv-CA / EEG-Transformer 等方向，环境里已有 GPU 版 PyTorch。
网络结构必须自己写；4 小时比赛窗口内调参风险高，建议备赛期在 Benchmark
上调好并固化，比赛当天作为补充方案，不要替代保底基线。`papers/` 中
SSVEPformer（07）、IncepFormerNet（08）、Guney DNN（09）是直接参照。

### 方法选择策略

| 方法 | 是否需要训练数据 | 速度 | 定位 |
| --- | --- | --- | --- |
| CCA | 否 | 秒级 | 保底基线，先提交 |
| FBCCA | 否 | 秒级 | 无训练数据时的主力 |
| TRCA/eTRCA | 是 | 快 | 有训练数据时的主力 |
| CNN/Transformer | 是（较多） | 慢 | 报告创新点、冲刺提分 |

---

## 六、数据说明

- **官方数据**：N 组真人 SSVEP 脑电，8 类（0~7）；测试数据集备赛期在
  QQ 群 100628255 发布，格式与正式赛完全一致；正式数据 T0-4h 才发加密包。
- **公开练习数据**：清华 SSVEP Benchmark（35 被试、64 导、250Hz、40 目标、
  6 block），下载方式见 `data/README.md`。取其中 8 个频率即可模拟比赛。
  另有 9 通道 12 目标的 MP 数据集（与本赛题通道/类别数最接近），见
  `papers/14_Zhao2024_MP_dataset.pdf`。
- **分段注意**：规则要求"获取每次实验的信号片段"。若官方给的是连续记录，
  用 `ssvep/preprocess.py` 的 `epoch()` 按触发点分段，统一扣除
  **0.14s 视觉延迟**，并把分段逻辑写进技术报告。
- 拿到数据第一步永远是 `explore_eeg_data.py`：确认格式、数组维度排列、
  通道数、采样率、0~7 与刺激频率的对应关系。

---

## 七、比赛当天 T0 流程（4 小时操作清单）

1. **T0-4h ~ T0**：提前下载加密包、解压密码公布后立即解压；确认电量/网络。
2. **T0+0~15min**：跑 `explore_eeg_data.py` 摸清数据（维度、采样率、频率表）。
3. **T0+15~40min**：适配数据读取（分段、通道、延迟），跑通 CCA，
   **立刻生成 Excel 提交第一版保底结果**（可多次提交，以最后一次为准）。
4. **T0+40~120min**：上 FBCCA；若有带标签训练数据，训练 TRCA/eTRCA，
   交叉验证后提交第二版。
5. **T0+120~210min**：尝试深度学习/参数调优（窗口长度、谐波数、子带数），
   每有提升就提交一版。
6. **T0+210~240min**：冻结结果，核对 Excel 文件名、行数、类别取值（0~7），
   按要求上传报名系统 + 网盘链接发邮箱，邮件主题与网盘目录一致。

关键纪律：**先保底再迭代**；正确率相同时提交时间早者排名靠前。

---

## 八、合规红线（务必遵守）

- 所有算法必须**自主编程实现**，严禁抄袭、代写，严禁用第三方成品算法或
  现成软件模块替代自主编程过程。
- numpy / scipy / pandas 是通用数学库，可以使用；本工程的 CCA、FBCCA、TRCA、
  滤波、分段全部为自主实现。**不要**直接调用 MNE、scikit-learn 中的现成
  分类/解码函数。
- 留存完整源代码、运行日志与结果版本，组委会将进行代码复核，总决赛一等奖
  候选还须参加线下复核赛。

---

## 九、技术报告要点（40 分）

按"完整性、创新性、可行性"评分，建议章节：

1. 数据采集与传输过程（SSVEP 范式、8 类刺激、电极与采样、触发同步）
2. 总体算法思路与技术对比（CCA→FBCCA→TRCA→深度方法的选型理由）
3. 信号预处理（带通/陷波、分段与 0.14s 延迟、坏道处理、归一化）
4. 特征提取方法（参考信号构造、滤波器组、空间滤波的数学原理）
5. 分类模型设计与优化策略（窗口长度、谐波数、子带权重、交叉验证）
6. 成果分析（ACC、混淆矩阵、不同窗口/方法对比、错误案例分析）
7. 关键代码实现说明（与提交源代码对应）

`papers/17_WRC2022_recognition_overview.pdf` 是同类竞赛（世界机器人大赛
SSVEP 本科组）的方法综述与冠军方案分析，技术报告的方法对比与指标部分
可直接参照其写作框架。

---

## 十、备赛 Checklist

- [ ] 官网报名赛题 13（截止 2026-11-05 20:00），队长加 QQ 群 100628255
- [ ] `python check_env.py` 环境就绪
- [ ] 通读并运行 `notebooks/` 5 篇教学笔记
- [ ] 按 `papers/README.md` 阅读顺序精读核心论文（02/03/17/07/08）
- [ ] `demo_synthetic.py` 三种算法自测通过（CCA/FBCCA 接近满分）
- [ ] 下载清华 Benchmark（或 MP 小数据集），`run_benchmark.py` 复现合理正确率
- [ ] 拿到官方测试数据集，跑通"探查→分段→分类→Excel"全流程
- [ ] 做一次 4 小时限时模拟
- [ ] 技术报告初稿完成
- [ ] 准备好提交用的网盘目录命名与邮件模板

---

## 十一、参考资料

完整论文库（15 篇本地 PDF + 3 篇付费奠基文献的官方链接）见
**`papers/README.md`**，含分类、推荐阅读顺序与数据集下载地址。核心文献：

- Lin Z. et al. Frequency recognition based on canonical correlation analysis
  for SSVEP-based BCIs. IEEE TBME, 2006.（CCA 开山之作，付费，见 papers/README）
- Chen X. et al. Filter bank canonical correlation analysis for implementing
  a high-speed SSVEP-based brain-computer interface. J. Neural Eng., 2015.
  （FBCCA，本地 `papers/02_Chen2015_FBCCA.pdf`）
- Tanaka H. et al. Task-related component analysis. NeuroImage, 2013.
  （TRCA 理论源头，本地 `papers/03_Tanaka2013_TRCA_original.pdf`；SSVEP 上的
  ensemble TRCA, Nakanishi 2018 IEEE TNSRE 为付费，见 papers/README）
- Wang Y. et al. A benchmark dataset for SSVEP-based BCIs. IEEE TNSRE, 2017.
  （清华 Benchmark，付费；数据集免费 http://bci.med.tsinghua.edu.cn/download.html）
- Chen J. et al. SSVEPformer. Neural Networks, 2023；Huang Y. et al.
  IncepFormerNet, 2025（Transformer 路线，本地 papers/07、08）
