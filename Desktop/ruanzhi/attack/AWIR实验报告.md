# AWIR（注意力加权词重要性排序）攻击实验报告

> **负责人**：徐梦茹  
> **产出文件**：`attack/improved_attack.py`、`results/improved/improved_results.json`

---

## 一、改进思路

### 1.1 TextFooler 的 WIR 方法

TextFooler（Jin et al., AAAI 2020）使用词重要性排序（Word Importance Ranking, WIR）决定先替换哪个词：

```
importance(w_i) = confidence(x) - confidence(x_without_w_i)
```

该方法只考虑了词的"删除影响"，没有利用 BERT 内部的注意力信息。

### 1.2 AWIR 改进

在 WIR 基础上乘以注意力权重因子：

```
importance(w_i) = [confidence(x) - confidence(x_without_w_i)] × (1 + attention_weight(w_i))
```

**直觉**：注意力权重高的词是 BERT 分类时"关注"的关键词。删除影响大且注意力又高的词，优先攻击，预期用更少的查询次数达到更高的攻击成功率。

### 1.3 注意力权重提取方式

1. 带 `output_attentions=True` 做一次前向传播，获取所有 12 层 × 12 头的注意力矩阵
2. 对所有层和头取均值，得到每个 token 的注意力分数
3. 将 subword token 合并为完整单词（均值聚合）
4. 归一化到 [0, 1]，作为 `attention_weight`

---

## 二、代码实现

### 2.1 文件及函数说明

| 函数 | 作用 |
|------|------|
| `predict(model, tokenizer, text, device)` | 单条文本预测，返回 (标签, 置信度) |
| `predict_batch(model, tokenizer, texts, device)` | 批量文本预测，一次前向传播处理多条 |
| `get_word_attention_weights(model, tokenizer, text, words, device)` | 提取 BERT 注意力权重，映射到单词级别 |
| `get_synonyms(word)` | 从 WordNet 获取同义词（最多 10 个候选） |
| `awir_attack(model, tokenizer, text, true_label, device, use_attention)` | 核心攻击函数，`use_attention=True` 即 AWIR |
| `compute_semantic_similarity(orig_texts, pert_texts)` | 用 sentence-transformers 计算语义相似度 |
| `run_experiment(model, tokenizer, dataset, ...)` | 批量运行实验，汇总 ASR、查询次数、扰动率、语义相似度 |

### 2.2 性能优化：批量推理

原始实现每删除一个词做一次单独前向传播，100 词的文本需 100 次调用。优化后：

```python
# 优化前：逐词循环（100 词 = 100 次前向传播）
for i in range(len(words)):
    deleted = " ".join(words[:i] + words[i+1:])
    _, conf = predict(model, tokenizer, deleted, device)  # 每次 1 条

# 优化后：批量推理（100 词 = 1 次前向传播）
deletion_texts = [" ".join(words[:i] + words[i+1:]) for i in range(n_words)]
_, confs = predict_batch(model, tokenizer, deletion_texts, device)  # 一次搞定
```

同义词替换同样批量化。整体提速 **20–40 倍**：

| | 优化前 | 优化后 |
|------|------|------|
| 100 词文本的前向传播次数 | ~200 | ~5–8 |
| CPU 预计耗时 (100 样本) | 数小时 | 15–30 分钟 |

> 注：Query 计数仍按"等效单次预测"统计（批量 N 条计 N 次），与 baseline attack 的 query 计数方式可比。

### 2.3 语义相似度计算

使用 `sentence-transformers`（`all-MiniLM-L6-v2`）计算成功攻击样本的原文与对抗文余弦相似度均值。原始论文使用 USE（Universal Sentence Encoder），数值可能略有差异，报告中需说明。

---

## 三、运行方式

### 3.1 本地运行（需 GPU，耗时 10–15 分钟）

```bash
conda activate bert_attack
cd ruanzhi
python attack/improved_attack.py --num_examples 100
```

可选参数：
- `--model_dir`：受害模型路径（默认 `textattack/bert-base-uncased-imdb`）
- `--num_examples`：攻击样本数（默认 100）
- `--results_dir`：结果输出目录（默认 `./results`）

### 3.2 Google Colab 运行（使用免费 T4 GPU）

上传 `attack/improved_attack_colab.py`，在 notebook 中执行：

```python
# 第1格：安装依赖
!pip install datasets nltk sentence-transformers -q
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 第2格：运行
!python improved_attack_colab.py
```

下载 `results/improved/improved_results.json` 到本地对应目录。

---

## 四、实验结果

| 指标 | WIR 对照组 | AWIR 实验组 |
|------|-----------|------------|
| **ASR（攻击成功率）** | 14.0% | 14.0% |
| **Avg Queries（平均查询次数）** | 495.78 | 495.68 |
| **Perturb Rate（扰动率）** | 1.01% | 1.01% |
| **Sem. Similarity（语义相似度）** | 0.9873 | 0.9873 |

> 实验条件：100 条 IMDB 测试样本、WordNet 同义词替换、textattack/bert-base-uncased-imdb 受害模型。

### 4.1 结论

AWIR 在此实验设置下**未产生显著改善**，两组结果几乎一致。可能原因：

1. **注意力均值化**：对 12 层 × 12 头全部取均值，不同层/头关注不同模式，平均后趋向均匀分布，`(1+attention_weight)` 缩放因子对排序影响微弱
2. **WordNet 同义词质量**：WordNet 同义词覆盖面有限，多数词没有合适的替代词
3. **ASR 偏低**：相比论文中 TextFooler 的 ~87% ASR（使用 counter-fitted 词嵌入），WordNet 替换的攻击力有限

### 4.2 后续改进方向

1. 仅使用特定层（如最后一层）的 `[CLS]` token 注意力，而非全部 12 层 12 头均值
2. 使用 counter-fitted 词嵌入替代 WordNet，提升同义词质量
3. 尝试基于梯度或 Shapley 值的词重要性度量

---

## 五、与 baseline attack 的对比说明

| 维度 | baseline_attack.py | improved_attack.py |
|------|-------------------|-------------------|
| 攻击方法 | TextFooler / BERT-Attack（TextAttack 框架） | 自定义 WIR/AWIR 实现 |
| 同义词来源 | counter-fitted embeddings | WordNet |
| 推理方式 | TextAttack 内置 | 自实现批量推理 |
| 语义相似度 | sentence-transformers | sentence-transformers |
| 输出格式 | CSV | JSON（含 WIR 对照组 + AWIR 实验组） |

两者语义相似度使用同一模型计算，指标可比。
