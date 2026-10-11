# SSVEP 脑电解码论文库

为「脑电信号特征提取与识别」赛题（SSVEP 8 分类）整理的权威文献索引，覆盖综述、经典方法（CCA/FBCCA/TRCA 系）、公开数据集、深度学习最新进展。论文 PDF 仅供本地使用，不纳入 GitHub 仓库；请从文中列出的官方或开放获取来源获取。

- 收录日期：2026-09-22
- 获取原则：**只收录开放获取（Open Access）版本**——arXiv、作者机构主页、PMC/出版社 OA 期刊；未使用 sci-hub 等灰色渠道。
- 3 篇奠基性论文（CCA 原文、Benchmark、ensemble TRCA）出版商仅付费开放，本地未收录，文末给出官方链接，可通过学校图书馆权限获取。
- 全部 PDF 已通过完整性校验（`%PDF` 头、`%%EOF` 尾、页数/大小正常），校验脚本：`python verify_pdfs.py`。

## 推荐阅读顺序

1. 先读 **11 中文综述**（入门最快）→ **10 Vialatte 综述**（领域圣经）→ **12 刺激参数综述**（理解实验设计）。
2. 经典方法主线：**18 CCA 方法对比**（PLoS，免费讲清 CCA/IT-CCA）→ **02 FBCCA** → **06 MsetCCA** → **03 TRCA 原始理论** → **16 gTRCA**。
3. 数据集与评测：**05 BETA** → **14 MP**（9 通道 12 目标，与本赛题 8 类最接近）→ **17 世界机器人大赛方法综述**（与本赛题形式几乎一致，强烈建议精读）。
4. 深度学习：**09 Guney DNN**（Benchmark+BETA 基线，附代码）→ **07 SSVEPformer**（首个 Transformer）→ **08 IncepFormerNet**（2025 最新，附代码）。

## 一、综述

| 编号 | 文件 | 题名 / 作者 / 出处 | 与比赛的关系 |
| --- | --- | --- | --- |
| 10 | `10_Vialatte2010_SSVEP_review.pdf` | Steady-state visually evoked potentials: Focus on essential paradigms and future perspectives. Vialatte F-B, Maurice M, Dauwels J, Cichocki A. *Progress in Neurobiology* 90(4):413–438, 2010 | SSVEP 领域最经典综述，机制、范式、信号处理的理论根基 |
| 11 | `11_Liu2022_SSVEP_review_CN.pdf` | Review of brain–computer interface based on steady-state visual evoked potential. Liu S, Zhang D, Liu Z, et al. *Brain Science Advances*, 2022（北理工，英文正文） | 系统梳理 CCA 及各改进解码算法，入门首选 |
| 12 | `12_Reitelbach2024_stimulus_review.pdf` | Influence of Stimulus Parameters on SSVEP-BCI Performance: A Scoping Review. Reitelbach C, et al. *Multimodal Technologies and Interaction* 8(2):6, 2024（MDPI OA） | 刺激频率/颜色/布局如何影响识别，帮你理解数据是怎么采的 |
| 17 | `17_WRC2022_recognition_overview.pdf` | Overview of recognition methods for SSVEP-based BCIs in World Robot Contest 2022: MATLAB undergraduate group. Yi C, Wu Y, Ye F, Zhang X, Chen J. *Brain Science Advances*, 2023 | **与本赛题最像**：竞赛数据上免校准方法横评，冠军 ITR 198.94 bits/min，技术报告必引 |

## 二、经典解码方法（免训练 / 少训练）

| 编号 | 文件 | 题名 / 作者 / 出处 | 与比赛的关系 |
| --- | --- | --- | --- |
| 02 | `02_Chen2015_FBCCA.pdf` | Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based BCI. Chen X, Wang Y, Nakanishi M, Gao X, Jung T-P, Gao S. *J. Neural Engineering* 12(4):046008, 2015（hybrid OA） | **FBCCA 原文**，你工程中 `ssvep/cca.py` 的 FBCCAClassifier 即按此文权重 w(n)=n^-1.25+0.25 实现 |
| 03 | `03_Tanaka2013_TRCA_original.pdf` | Task-related component analysis for functional neuroimaging and application to near-infrared spectroscopy data. Tanaka H, Katura T, Sato H. *NeuroImage* 83:308–320, 2013（作者机构 JAIST 主页开放） | **TRCA 理论源头**：最大化试次间可重复性，广义瑞利商推导，对应 `ssvep/trca.py` |
| 06 | `06_Zhang2013_MsetCCA.pdf` | Frequency recognition in SSVEP-based BCI using multiset canonical correlation analysis. Zhang Y, Zhou G, Jin J, Wang X, Cichocki A. *Int. J. Neural Systems* 24(4):1450013, 2014（arXiv:1308.5609） | MsetCCA，用多试次学习参考模板，CCA→数据驱动的过渡 |
| 16 | `16_Tanaka2020_gTRCA.pdf` | Group task-related component analysis (gTRCA): inter-trial reproducibility and inter-subject similarity maximization for EEG. Tanaka H. *Scientific Reports* 10:440, 2020（Nature OA） | TRCA 的跨被试扩展，理解被试间泛化（正式数据无训练样本时的思路） |
| 18 | `18_Nakanishi2015_CCA_comparison.pdf` | A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting SSVEPs. Nakanishi M, Wang Y, Wang Y-T, Jung T-P. *PLoS ONE* 10(10):e0140703, 2015（OA） | 标准 CCA / IT-CCA 在 Benchmark 上的系统对比，CCA 调参（谐波数、窗口）的实证依据 |
| 15 | `15_Wai2020_CSTA_subsecond_WITHDRAWN-v1.pdf` | Towards a Fast SSVEP-BCI (CSTA: training-free spatial-filtering + temporal alignment). Wai A A P, Zhang Y, Guo H, et al. arXiv:2002.01171**v1**, 2020 | ⚠️ **撤稿稿，见下方声明**。仅亚秒级免训练融合思路可参考 |

## 三、公开数据集

| 编号 | 文件 | 题名 / 作者 / 出处 | 与比赛的关系 |
| --- | --- | --- | --- |
| 05 | `05_Liu2020_BETA_dataset.pdf` | BETA: A Large Benchmark Database Toward SSVEP-BCI Application. Liu B, Chen X, Shi N, Wang Y, Gao S, Gao X. *Frontiers in Neuroscience* 14:627, 2020（arXiv:1911.13045） | BETA 数据集论文，70 被试 40 目标，跨被试/跨天场景，比 Benchmark 更接近真实比赛 |
| 14 | `14_Zhao2024_MP_dataset.pdf` | MP: A steady-state visual evoked potential dataset based on multiple paradigms. Zhao X, Xu S, Geng K, Wang Z, Feng S, Hu H. *iScience* 27(11):111030, 2024（gold OA） | **9 枕区通道、12 目标、24 被试**，通道数和类别数与本赛题（8 类）最接近，强烈建议下载练习 |

## 四、深度学习

| 编号 | 文件 | 题名 / 作者 / 出处 | 与比赛的关系 |
| --- | --- | --- | --- |
| 09 | `09_Guney2020_DeepDNN.pdf` | A Deep Neural Network for SSVEP-based Brain Computer Interfaces. Guney O B, Oblokulov M, Ozkan H. arXiv:2011.08562, 2020 | Benchmark+BETA 上的 CNN 强基线（0.4 s 达 ITR 265/197 bits/min），官方代码 github.com/osmanberke/Deep-SSVEP-BCI |
| 07 | `07_Chen2023_SSVEPformer.pdf` | SSVEPformer: A Transformer-based model for SSVEP classification. Chen J, et al. *Neural Networks* 165:170–181, 2023（arXiv:2210.04172） | **首个把 Transformer 用于 SSVEP**，赛题背景点名 Transformer，技术报告创新点首选参照 |
| 08 | `08_Huang2025_IncepFormerNet.pdf` | IncepFormerNet: A multi-scale multi-head attention network for SSVEP classification. Huang Y, Chen Y, Cao L, et al. arXiv:2502.13972, 2025 | 2025 年最新：Inception 多尺度卷积 + Transformer，1.0 s 窗口 87.4%/72.0%，代码 github.com/CECNL/SSVEP-DAN |

## 五、未收录的付费奠基论文（请通过武大图书馆/IEEE Xplore 获取）

| 编号 | 文献 | 官方链接 |
| --- | --- | --- |
| 01 | Lin Z, Zhang C, Wu W, Gao X. Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs. *IEEE TBME* 53(12):2610–2614, 2006（**CCA 应用于 SSVEP 的开山之作**） | https://doi.org/10.1109/TBME.2006.886577 |
| 04 | Wang Y, Chen X, Gao X, Gao S. A Benchmark Dataset for SSVEP-Based BCIs. *IEEE TNSRE* 25(10):1746–1752, 2017（**Benchmark 数据集论文**；数据集本身免费） | https://doi.org/10.1109/TNSRE.2016.2627556 |
| 13 | Nakanishi M, Wang Y, Chen X, Wang Y-T, Gao X, Jung T-P. Enhancing Detection of SSVEP for BCIs Using Task-Related Component Analysis. *IEEE TNSRE* 26(4):871–879, 2018（**ensemble TRCA，SSVEP 上的 TRCA 论文**） | https://doi.org/10.1109/TNSRE.2018.2814064 |

> Benchmark 数据集（35 被试、64 导、40 目标、250 Hz、约 3.3 GB）免费下载页：http://bci.med.tsinghua.edu.cn/download.html
> BETA 数据集：http://bci.med.tsinghua.edu.cn/download.html （同页）

## 撤稿声明（文件 15）

`15_Wai2020_CSTA_subsecond_WITHDRAWN-v1.pdf` 是 arXiv:2002.01171 的 **v1 历史快照**。该稿 v2（2020-05）已被作者撤稿，撤稿说明为 *"Further improvements or modifications required to algorithm design"*（算法设计尚需改进），arXiv 当前页面已不提供 PDF。**不要把文中的精度数字当作已验证结论，也不要在技术报告中把它作为权威方法引用**；仅可作为「亚秒级免训练、空间滤波+时间对齐互补融合」这一思路的启发材料。

## 数据集下载与使用提示

- **Benchmark**：35 名被试 `S01-S35.mat`，64 导 250 Hz，40 目标（8.0–15.8 Hz，间隔 0.2 Hz），每目标 6 个 block，每 trial 6 s；数据形状 `[64, 1500, 40, 6]`，视觉延迟 0.14 s（35 个采样点）。工程已在 `examples/run_benchmark.py` 配好 9 个枕区通道索引与留一 block 交叉验证。
- **MP**：9 枕区通道、12 目标、24 被试、5 种刺激范式，规模小、下载快，建议先用它跑通全流程。
- 官方比赛数据待 QQ 群（100628255）在 T0-4h 发布加密包，T0 给密码，T0+4h 截止；本库论文仅用于算法准备。

## 合规提醒

赛题明确要求算法自主编程，**严禁调用第三方成品解码算法/现成软件模块**。这些论文用于理解原理；numpy/scipy/pandas 等通用数学库可以用，但 MNE、scikit-learn 中的现成 CCA/TRCA/分类解码函数不要直接用于提交作品（学习理解可以）。工程 `ssvep/` 包中的 CCA、FBCCA、TRCA、正余弦参考信号均为手写实现，可直接用于参赛。
