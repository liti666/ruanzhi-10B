# 基于 BERT 的时序感知模型对抗攻击与鲁棒性分析

Adversarial Attack and Robustness Analysis of BERT-based Text Classification Models.

## 项目结构

```
ruanzhi-10B/
├── configs/config.py           # 全局参数配置（模型名、数据集、路径等）
├── data/download_data.py       # 一键预下载数据集与 NLTK 资源
├── train/finetune_bert.py      #  Fine-tune BERT，保存受害模型
├── attack/
│   ├── baseline_attack.py      #  复现 TextFooler / BERT-Attack
│   └── improved_attack.py      # 改进方法：注意力加权重要性排序(AWIR)
├── defense/adversarial_training.py  #  对抗训练防御
├── evaluate/
│   ├── evaluate.py             #  汇总所有实验结果，输出对比表
│   └── visualize.py            #  生成报告用图表
├── results/                    # 实验输出
├── checkpoints/                # 模型权重
├── requirements.txt
└── main.py                     # 环境验证脚本
```

## 快速开始（按此顺序执行）

### 第一步：克隆仓库，安装依赖

```bash
git clone https://github.com/faye370/ruanzhi-10B.git
cd ruanzhi-10B
python install.py
```

> `install.py` 会自动检测是否有 NVIDIA GPU：有则安装 CUDA 版 PyTorch，没有则安装 CPU 版。
> 不要直接运行 `pip install -r requirements.txt`，否则会把 CUDA 版 PyTorch 覆盖成 CPU 版。

### 第二步：预下载数据集（只需执行一次）

```bash
python data/download_data.py
```

### 第三步：验证环境

```bash
python main.py
```

输出 "所有环境依赖已全部就位" 即表示环境正常。

---

## 执行命令

> 注意：`train/finetune_bert.py` 必须最先执行，其他攻击脚本依赖其输出的模型。

### 成员1 — 模型训练

```bash
python train/finetune_bert.py
# 完成后模型保存至 ./checkpoints/bert-imdb/
# 通知其他成员可以开始
```

### 2— 基线攻击复现（等成员1完成后）

```bash
python attack/baseline_attack.py
# 结果：./results/baseline/textfooler_results.csv
#        ./results/baseline/bertattack_results.csv
```

### 3 — 改进攻击（等成员1完成后）

```bash
python attack/improved_attack.py
# 结果：./results/improved/improved_results.json
```

### 4 — 对抗训练（等成员1完成后）

```bash
python defense/adversarial_training.py
# 结果：./checkpoints/bert-imdb-adv/（鲁棒模型）
```

### 5 — 汇总评估与可视化（等以上全部完成后）

```bash
python evaluate/evaluate.py
python evaluate/visualize.py
# 结果：./results/final_comparison.csv
#        ./results/figures/*.png
```

---

## 实验设计

### 基线攻击（复现论文结果）

| 方法 | 威胁模型 | 对应论文 | 会议 |
|------|----------|----------|------|
| TextFooler | 黑盒 | *Is BERT Really Robust? A Strong Baseline for Natural Language Attack...* Jin et al. | AAAI 2020 |
| BERT-Attack | 黑盒 | *BERT-ATTACK: Adversarial Attack Against BERT Using BERT* Li et al. | EMNLP 2020 |
| HotFlip | 白盒 | *HotFlip: White-Box Adversarial Examples for Text Classification* Ebrahimi et al. | ACL 2018 |

### 改进方法：AWIR（注意力加权重要性排序）

标准 TextFooler 的词重要性：`importance(w_i) = conf(x) - conf(x_删除w_i)`

本项目改进（AWIR）：`importance(w_i) = [conf(x) - conf(x_删除w_i)] × (1 + attention_weight(w_i))`

直觉：BERT 的注意力权重反映了模型在分类时"关注"哪些词。同时具有高删除分数和高注意力权重的词对预测贡献最大——优先扰动这些词，在更少的查询次数内达到更高的攻击成功率。

### 防御：对抗训练

在训练集中混入 TextFooler 生成的对抗样本，重新微调模型，提升鲁棒准确率。

---

## 评估指标

| 指标 | 含义 |
|------|------|
| ASR (Attack Success Rate) | 攻击成功率，越高说明模型越脆弱 |
| Avg Queries | 平均模型查询次数，越少说明攻击越高效 |
| Perturbation Rate | 平均被修改的词比例，越低说明攻击越隐蔽 |
| Robust Accuracy | 被攻击后模型仍能正确分类的比例 |

---

## 环境要求

- Python 3.9
- CUDA GPU（推荐，训练时间从 10h 降至 2h）
- 详见 [环境搭建README.md](环境搭建README.md)
