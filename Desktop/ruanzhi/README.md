# 基于 BERT 的对抗攻击与鲁棒性分析

**课程项目 · 5人分工 · Python + PyTorch**


## 目录结构

```
ruanzhi-10B/
├── configs/config.py                    # 全局参数（模型名、路径、超参数）
├── data/download_data.py                # 预下载数据集与 NLTK 资源
├── train/finetune_bert.py               # BERT 微调流程（参考用，攻击不依赖此步）
├── attack/
│   ├── baseline_attack.py               # TextFooler / BERT-Attack / HotFlip（可单独运行）
│   └── improved_attack.py               # 改进A：AWIR 注意力加权攻击
├── defense/adversarial_training.py      # 改进B：对抗训练防御
├── evaluate/
│   ├── evaluate.py                      # 汇总所有结果，输出对比表
│   └── visualize.py                     # 生成图表
├── results/                             # 实验输出（自动创建）
├── checkpoints/                         # 模型权重（自动创建）
├── install.py                           # 一键安装脚本（自动判断 GPU/CPU）
├── requirements.txt
└── main.py                              # 环境验证
```


## 一、环境搭建（每位成员都要在自己电脑上做，只做一次）

### 前提条件

- Python 3.8 ~ 3.10（推荐 3.9）

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
> - **有 GPU** → 安装 CUDA 版 PyTorch
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

## 二、受害模型说明（无需自行训练）

本项目直接使用 **TextAttack 官方已微调好的模型**，所有成员无需运行训练脚本：

| 参数 | 值 |
|------|-----|
| 模型 | `textattack/bert-base-uncased-imdb` |
| 基础架构 | BERT-base-uncased |
| 微调数据集 | IMDB 情感分类（正面/负面二分类） |
| 干净准确率 | **93.7%** |

**首次运行攻击脚本时，模型自动下载到本地缓存（~418 MB）。**

---

## 三、五人分工

> 每人独立运行自己的脚本，结果文件互不覆盖。
> 成员5需要等成员1~4全部跑完后再执行汇总。

### 成员1 — TextFooler 复现（黑盒攻击）

**文件**：`attack/baseline_attack.py`

```bash
python attack/baseline_attack.py --attack textfooler
```

**产出**：`results/baseline/textfooler_results.csv`

**需要记录并与论文对比的指标**（对应论文 Table，BERT victim on IMDB）：

| 指标 | 论文报告值 | 你的复现值（填入） |
|------|-----------|-----------------|
| ASR（攻击成功率） | ~87% | |
| Avg Words Changed | ~6 词 | |
| Avg Queries | — | |

> 复现值与论文值存在差异是正常的（论文用的 BERT 版本、样本数可能不同），需在报告中说明原因。

**对应论文**：*Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment* Jin et al., AAAI 2020

---

### 成员2 — BERT-Attack 复现（黑盒攻击）

**文件**：`attack/baseline_attack.py`

```bash
python attack/baseline_attack.py --attack bertattack
```

**产出**：`results/baseline/bertattack_results.csv`

**需要记录并与论文对比的指标**（对应论文 Table 2，BERT on IMDB）：

| 指标 | 论文报告值 | 你的复现值（填入） |
|------|-----------|-----------------|
| ASR（攻击成功率） | ~90%+ | |
| Avg Words Changed | — | |
| Avg Queries | — | |

**对应论文**：*BERT-ATTACK: Adversarial Attack Against BERT Using BERT* Li et al., EMNLP 2020

---

### 成员3 — HotFlip 复现（白盒攻击）

**文件**：`attack/baseline_attack.py`

```bash
python attack/baseline_attack.py --attack hotflip
```

> HotFlip 是白盒攻击（需要模型梯度），CPU 上极慢，建议用 GPU。

**产出**：`results/baseline/hotflip_results.csv`

**需要记录并与论文对比的指标**：

| 指标 | 论文报告值 | 你的复现值（填入） |
|------|-----------|-----------------|
| ASR（攻击成功率） | 接近100%（白盒梯度优势） | |
| Avg Words Changed | 较多（白盒不受扰动约束） | |
| Avg Queries | 极少（梯度直接指引） | |

> HotFlip 原论文主要针对字符级扰动，TextAttack 实现的是词级版本，因此与原论文指标对比意义有限，重点是与黑盒方法横向比较（ASR 更高、查询更少，代价是需要白盒访问）。

**对应论文**：*HotFlip: White-Box Adversarial Examples for Text Classification* Ebrahimi et al., ACL 2018

---

### 成员4 — 改进A：AWIR 改进攻击

**文件**：`attack/improved_attack.py`

```bash
python attack/improved_attack.py
```

**产出**：`results/improved/improved_results.json`（含 WIR 对照组和 AWIR 实验组的 ASR、查询次数、扰动率）

详见下方"改进A详解"。

---

### 成员5 — 鲁棒性分析：对抗训练防御 & 汇总评估

**文件**：`defense/adversarial_training.py`、`evaluate/evaluate.py`、`evaluate/visualize.py`

**步骤（等成员1~4跑完后执行）**：

```bash
# 第1步：对抗训练
python defense/adversarial_training.py
# 产出：checkpoints/bert-imdb-adv/

# 第2步：用 TextFooler 攻击鲁棒模型，测量防御效果
python attack/baseline_attack.py \
    --attack textfooler \
    --model_dir checkpoints/bert-imdb-adv \
    --results_dir results/defense
# 产出：results/defense/textfooler_results.csv

# 第3步：汇总所有实验结果
python evaluate/evaluate.py
# 产出：results/final_comparison.csv

# 第4步：生成图表
python evaluate/visualize.py
# 产出：results/figures/*.png
```

详见下方"改进B详解"。

---

## 四、改进A详解：AWIR（注意力加权重要性排序）

### 标准 WIR 的局限性

TextFooler 使用词重要性排序（WIR）决定先替换哪个词：

```
importance(w_i) = conf(x) - conf(x_删除w_i)
```

只考虑了"删除影响"，没有利用 BERT 内部的注意力信息。

### AWIR 改进公式

```
importance(w_i) = [conf(x) - conf(x_删除w_i)] × (1 + attention_weight(w_i))
```

**attention_weight 计算方式**：
1. 带 `output_attentions=True` 做一次前向传播
2. 提取全部 12 层 × 12 头的注意力矩阵，对所有层和头取均值
3. 对 subword token 合并为完整单词（取均值）
4. 归一化到 [0, 1]

**直觉**：注意力权重高的词是 BERT 在分类时"关注"的词。同时删除影响大、注意力又高的词才是真正的关键词，优先攻击它，用更少查询达到更高 ASR。

### 实验设计（A/B 对比）

脚本自动跑两组，唯一区别是词排序方式：

| 组别 | 排序方式 | 参数 |
|------|----------|------|
| 控制组（WIR baseline） | 仅删除影响 | `use_attention=False` |
| 实验组（AWIR） | 删除影响 × 注意力权重 | `use_attention=True` |

**预期结果**：AWIR 的 ASR 高于 WIR，平均查询次数低于 WIR。

### 局限性

1. 额外计算开销：每次攻击多一次带注意力输出的前向传播
2. 注意力权重 ≠ 因果重要性：高注意力不完全等于高预测贡献
3. 短文本下 WIR 分数噪声较大，乘以注意力可能放大误差

---

## 五、鲁棒性分析：对抗训练防御

> **注意**：对抗训练是一种**防御方法**，不是对现有攻击工具的算法改进。改进A（AWIR）修改了攻击的核心评分公式，属于方法层面的改进；对抗训练是从模型训练侧提升鲁棒性，对应题目"鲁棒性分析"部分。两者研究方向不同，互为补充。

### 核心思想

在训练时将对抗样本混入训练集，让模型见过被攻击的文本，从而提高鲁棒性。这是对抗鲁棒性研究的标准基线方法，直接对应题目中"鲁棒性分析"部分。

### 流程

```
① 用 TextFooler 攻击训练集，生成对抗样本
② 将对抗样本与干净样本 1:1 混合，构建混合训练集
③ 在混合数据集上重新微调 BERT（2 epoch）
④ 用 TextFooler 再次攻击，对比干净模型 vs 鲁棒模型的 ASR 变化
```

### 预期结果

| 模型 | 干净准确率 | 鲁棒准确率（被 TextFooler 攻击后） |
|------|-----------|----------------------------------|
| 原始模型 | | |
| 对抗训练后 ||  |


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

**Q：需要先运行训练脚本吗？**
> 不需要。所有攻击和防御脚本默认使用现成预训练模型，与 `train/finetune_bert.py` 完全独立。

---

## 八、环境要求

- Python 3.8~3.10
- PyTorch 2.0+（由 `install.py` 自动选择 GPU/CPU 版）
- transformers 4.38.0
- textattack 0.3.x
- datasets 2.18.0
- CUDA 11.8

详见 `requirements.txt`。
