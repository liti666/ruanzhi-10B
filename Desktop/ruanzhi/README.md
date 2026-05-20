# 基于 BERT 的对抗攻击与鲁棒性分析

**课程项目 · 5人分工 · Python + PyTorch**

---

## 目录结构

```
ruanzhi-10B/
├── configs/config.py                    # 全局参数（模型名、路径、超参数）
├── data/download_data.py                # 预下载数据集与 NLTK 资源
├── train/finetune_bert.py               # [成员1] BERT 微调脚本（演示用）
├── attack/
│   ├── baseline_attack.py               # [成员2+3] TextFooler / BERT-Attack / HotFlip
│   └── improved_attack.py               # [成员4] AWIR 改进攻击
├── defense/adversarial_training.py      # [成员5] 对抗训练防御
├── evaluate/
│   ├── evaluate.py                      # [成员5] 汇总结果，输出对比表
│   └── visualize.py                     # [成员5] 生成图表
├── results/                             # 实验输出（自动创建）
├── checkpoints/                         # 模型权重（自动创建）
├── install.py                           # 一键安装脚本（自动判断 GPU/CPU）
├── requirements.txt
└── main.py                              # 环境验证
```

---

## 一、环境搭建（所有人都要做，只做一次）

### 前提条件

- Python 3.8 ~ 3.10（推荐 3.9）
- 网络：**需要能访问 HuggingFace**（国内用户需开 VPN，或使用校园网代理）
- 有 NVIDIA GPU（推荐），无 GPU 也能跑，速度更慢

### 步骤

**第1步：克隆代码**

```bash
git clone https://github.com/faye370/ruanzhi-10B.git
cd ruanzhi-10B
```

**第2步：安装依赖**

```bash
python install.py
```

> `install.py` 会自动检测有无 NVIDIA GPU：
> - **有 GPU** → 安装 CUDA 版 PyTorch（保留你现有的 GPU 环境）
> - **无 GPU** → 安装 CPU 版 PyTorch
>
> ⚠️ **不要**直接 `pip install -r requirements.txt`，否则会把 CUDA 版 PyTorch 替换成 CPU 版。

**第3步：下载数据集和 NLTK 资源**

```bash
python data/download_data.py
```

**第4步：验证环境**

```bash
python main.py
```

输出 "所有环境依赖已全部就位" 即表示成功。

---

## 二、受害模型说明

本项目使用 **TextAttack 官方已微调好的模型**，无需自行训练：

| 参数 | 值 |
|------|-----|
| 模型 | `textattack/bert-base-uncased-imdb` |
| 基础架构 | BERT-base-uncased |
| 微调数据集 | IMDB 情感分类（二分类：正面/负面） |
| 干净准确率 | 93.7% |
| 来源 | HuggingFace Hub（TextAttack 官方） |

首次运行任意攻击脚本时，模型会**自动下载**到本地缓存（~418 MB，需要 VPN）。

成员1的 `train/finetune_bert.py` 是完整的微调流程演示（用于展示训练过程、供报告撰写），攻击阶段**不依赖**其输出。

---

## 三、五人分工与执行命令

### 成员1 — 环境搭建 & 微调流程演示

**职责**：负责让整个项目跑通，完成上方"环境搭建"部分，并运行微调脚本演示训练流程。

**文件**：`train/finetune_bert.py`

```bash
python train/finetune_bert.py
```

**产出**：
- `checkpoints/bert-imdb/`（本地微调的模型权重，演示用）
- 训练曲线截图（loss 下降、eval accuracy 上升），供报告使用
- 确认其他成员能正常 clone 代码、安装环境

**说明**：此步骤需要 GPU，预计 30~40 分钟（3 epoch）。如果没有 GPU 可以截取前几步的 loss 日志即可。

---

### 成员2 — 基线攻击：TextFooler & BERT-Attack

**职责**：复现两篇论文的黑盒攻击方法，记录攻击成功率。

**文件**：`attack/baseline_attack.py`

```bash
python attack/baseline_attack.py
```

> 脚本会依次跑 TextFooler → BERT-Attack → HotFlip，全部完成约需 1~2 小时（GPU）。
> 如果只想跑自己负责的两个，可以临时注释掉 HotFlip 那段（约第 102~108 行）。

**产出**：
- `results/baseline/textfooler_results.csv`
- `results/baseline/bertattack_results.csv`
- `results/baseline/baseline_summary.txt`（自动打印对比表）

**对应论文**：
- TextFooler：*Is BERT Really Robust?* Jin et al., AAAI 2020
- BERT-Attack：*BERT-ATTACK: Adversarial Attack Against BERT Using BERT* Li et al., EMNLP 2020

---

### 成员3 — 基线攻击：HotFlip（白盒）

**职责**：复现白盒攻击方法 HotFlip，与黑盒方法做对比。

**文件**：`attack/baseline_attack.py`（同上，同一个脚本）

```bash
python attack/baseline_attack.py
```

> 与成员2使用同一脚本，建议协商由一人统一跑完三个攻击，避免重复跑。
> HotFlip 是白盒攻击（需要梯度），只能在 GPU 上运行，CPU 上极慢。

**产出**：
- `results/baseline/hotflip_results.csv`

**对应论文**：HotFlip: *White-Box Adversarial Examples for Text Classification* Ebrahimi et al., ACL 2018

---

### 成员4（组长）— 改进攻击：AWIR

**职责**：实现并验证 AWIR（注意力加权重要性排序）改进方法，与标准 WIR 做 A/B 对比。

**文件**：`attack/improved_attack.py`

```bash
python attack/improved_attack.py
```

**产出**：
- `results/improved/improved_results.json`（包含 WIR 对照组和 AWIR 实验组的 ASR、查询次数、扰动率）

详见下方"改进方法详解"。

---

### 成员5 — 对抗训练防御 & 汇总评估

**职责**：训练鲁棒模型，汇总全部实验结果，生成报告图表。

**文件**：`defense/adversarial_training.py`、`evaluate/evaluate.py`、`evaluate/visualize.py`

**步骤（按顺序执行）**：

```bash
# 第1步：对抗训练（约1~2小时，需要等成员2跑完攻击后才能进行）
python defense/adversarial_training.py

# 第2步：用 TextFooler 再攻击一次鲁棒模型，看防御效果
python attack/baseline_attack.py --model_dir checkpoints/bert-imdb-adv \
    --results_dir results/defense

# 第3步：汇总所有实验结果（等所有人都跑完后）
python evaluate/evaluate.py

# 第4步：生成图表
python evaluate/visualize.py
```

**产出**：
- `checkpoints/bert-imdb-adv/`（对抗训练后的鲁棒模型）
- `results/defense/textfooler_robust_results.csv`（鲁棒模型被攻击的结果）
- `results/final_comparison.csv`（最终对比表，直接贴进报告）
- `results/figures/*.png`（图表，直接贴进报告）

---

## 四、改进方法详解：AWIR

### 背景：标准 WIR 的局限性

TextFooler 使用**词重要性排序（WIR, Word Importance Ranking）**决定先替换哪个词：

```
importance(w_i) = conf(x) - conf(x_删除w_i)
```

即：把第 i 个词删掉之后，模型置信度下降越多，说明这个词越重要，优先被替换。

**问题**：这种方法只考虑了"删除影响"，但没有利用模型内部的结构信息——BERT 的注意力权重明确记录了模型在分类时"关注"哪些词。如果一个词同时满足：
1. 删掉之后置信度下降很多（WIR 分数高）
2. 在注意力机制里被大量关注（attention 权重高）

那么这个词才是真正的关键词，应该最优先被攻击。

### AWIR 改进公式

```
importance(w_i) = [conf(x) - conf(x_删除w_i)] × (1 + attention_weight(w_i))
```

其中 `attention_weight(w_i)` 通过以下方式计算：
1. 对输入文本做一次前向传播，提取全部 12 层、12 个注意力头的权重矩阵
2. 对所有层和所有头取平均，得到每个 token 的综合注意力权重
3. 将 subword token 的权重合并为完整单词的权重（取均值）
4. 归一化到 [0, 1] 范围

### 理论预期

| 维度 | 标准 WIR | AWIR（本方法）|
|------|----------|--------------|
| 词排序依据 | 仅删除影响 | 删除影响 × 注意力权重 |
| 利用 BERT 内部信息 | 否 | 是 |
| 预期 ASR | 基准 | 更高（±3~8%） |
| 预期查询次数 | 基准 | 更少（排序更准，少走弯路） |

### 实验设计（A/B 对比）

脚本自动跑两组：
- **控制组**（WIR baseline）：`use_attention=False`，即标准 TextFooler 的词排序
- **实验组**（AWIR）：`use_attention=True`，加入注意力权重

两组完全相同的数据、相同的模型、相同的替换词库，唯一区别是词排序方式。这样可以排除其他变量，直接衡量注意力权重的贡献。

### 局限性（诚实分析，报告里要写）

1. **计算开销增加**：每次攻击多一次带 `output_attentions=True` 的前向传播
2. **注意力权重不等于因果重要性**：注意力高的词不一定是分类决策的直接原因（注意力机制关注的是词义关联，不完全对应预测置信度）
3. **WIR 分数本身有噪声**：短文本删词影响不稳定，乘以注意力权重可能放大噪声

---

## 五、实验结果汇总结构（供报告参考）

全部跑完后，`evaluate/evaluate.py` 会输出如下格式的对比表：

```
Method                    Total   Successful   ASR (%)   Avg Words Changed   Avg Queries
TextFooler (Jin 2020)       200          ...       ...                 ...           ...
BERT-Attack (Li 2020)       200          ...       ...                 ...           ...
HotFlip (Ebrahimi 2018)     200          ...       ...                 ...           ...
WIR (control, ours)         100          ...       ...                 ...           ...
AWIR (ours, improved)       100          ...       ...                 ...           ...
```

---

## 六、评估指标说明

| 指标 | 含义 | 越高代表 |
|------|------|---------|
| ASR（攻击成功率） | 成功让模型分类出错的比例 | 模型越脆弱 / 攻击越强 |
| Avg Queries | 平均每次攻击调用模型的次数 | 查询越少 = 攻击越高效 |
| Perturbation Rate | 平均修改词的比例 | 修改越少 = 攻击越隐蔽 |
| Robust Accuracy | 被攻击后仍能正确分类的比例 | 模型越鲁棒 |

---

## 七、常见问题

**Q：运行时提示"无法连接 HuggingFace"？**
> 国内网络访问 HuggingFace 需要 VPN。首次运行时会自动下载模型（~418 MB），之后会缓存到本地，不需要再次下载。

**Q：`python install.py` 运行后提示 torch 版本不对？**
> 如果你已经有 CUDA 版 PyTorch，`install.py` 会跳过 torch 安装，不会覆盖。如有问题查看脚本输出。

**Q：没有 GPU 能跑吗？**
> 可以，但速度慢很多。TextFooler 攻击 200 条样本：GPU 约 30 分钟，CPU 约 3~5 小时。建议 `--num_examples 20` 先小跑验证代码正确，再交给有 GPU 的人跑完整实验。

**Q：攻击脚本需要等成员1训练完吗？**
> **不需要**。攻击脚本默认使用 `textattack/bert-base-uncased-imdb`（已微调的现成模型），不依赖本地训练的 checkpoint。成员1的训练脚本是独立的演示流程。

**Q：baseline_attack.py 成员2和成员3都要跑吗？**
> 同一个脚本，建议一人跑完三个攻击，另一人负责分析结果写报告。不要两人同时跑，结果会重复覆盖。

---

## 八、环境要求

- Python 3.8~3.10
- PyTorch 2.0+（GPU 版或 CPU 版，由 `install.py` 自动选择）
- transformers 4.38.0
- textattack 0.3.x
- datasets 2.18.0
- CUDA 11.8（有 GPU 时）

详见 `requirements.txt`。
