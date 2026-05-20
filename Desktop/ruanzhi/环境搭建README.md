# 📝 实验环境搭建指南 (Environment Setup)

> **项目名称**：基于 BERT 的时序感知模型对抗攻击与鲁棒性分析
> **系统要求**：Windows 11 (x64)
> **技术栈**：Python 3.9 + PyTorch + Transformers + TextAttack

本文档旨在指导开发者从零开始搭建本项目的专属虚拟环境，确保基线算法（TextFooler、HotFlip）复现及后续自适应动态调整策略实验的可复现性。

---

## 零、 前置准备 (Prerequisites)

在开始之前，请确保您的计算机已安装以下基础软件：
1. **Miniconda** (用于环境隔离与管理)：[点击下载 Miniconda3 Windows版](https://docs.conda.io/en/latest/miniconda.html)
2. **Visual Studio Code** (推荐的 IDE)：[点击下载 VS Code](https://code.visualstudio.com/)
   - *注意：请在 VS Code 的插件市场中安装官方 `Python` 插件。*

---

## 一、 构建虚拟环境

为了防止依赖冲突，本项目使用 Miniconda 隔离开发环境。

1. 打开 Windows 开始菜单，找到并运行 **Anaconda Prompt (Miniconda3)**。
2. 创建名为 `bert_attack` 的虚拟环境，并严格指定 Python 版本为 3.9.25：
   ```bash
   conda create -n bert_attack python=3.9.25 -y
   ```
3. 激活该虚拟环境：
   ```bash
   conda activate bert_attack
   ```
   *(激活成功后，命令行提示符的最左侧会显示 `(bert_attack)`)*

---

## 二、 安装核心组件与依赖库

在 `(bert_attack)` 环境下，执行以下命令一键安装本项目的核心技术栈（包括 PyTorch CPU版本、HuggingFace 生态库及 TextAttack）：

```bash
pip install torch torchvision torchaudio transformers datasets textattack
```

### 依赖项修复：NLTK 数据包
`textattack` 强依赖于 NLTK 的某些语料库。为了防止后续运行攻击脚本时报错，请在终端中继续输入 `python` 进入交互模式，并执行以下下载命令：

```python
import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
exit() # 下载完成后退出 Python
```

---

## 三、 IDE 与调试链路配置 (VS Code)

本项目要求支持通过命令行参数动态调整实验参数（如任务名称、Batch Size 等）。请按以下步骤配置 VS Code：

1. **选择解释器**：
   在 VS Code 中打开项目根目录，按下 `Ctrl + Shift + P`，输入 `Python: Select Interpreter`，在弹出的列表中选择带有 `bert_attack` 字样的 Python 解释器。

2. **配置 launch.json**：
   在项目根目录下创建一个名为 `.vscode` 的文件夹，并在其中新建 `launch.json` 文件，填入以下内容：

   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Python: 自动化校验 (main.py)",
               "type": "debugpy",
               "request": "launch",
               "program": "${workspaceFolder}/main.py",
               "console": "integratedTerminal",
               "justMyCode": true,
               "args": [
                   "--task", "imdb",
                   "--batch_size", "8"
               ]
           }
       ]
   }
   ```

---

## 四、 核心组件可用性验证

在项目根目录下创建 `main.py` 文件，复制以下校验代码：

```python
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import textattack

# 忽略即将弃用的 API 警告 (对应报告截图中 pkg_resources 的警告)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    print("\n" + "="*50)
    print(" 🚀 环境自动化校验脚本 (bert_attack) 开始...")
    print("="*50)
    
    # 1. 框架验证
    print("\n[1/3] 检查深度学习框架...")
    print(f"  [-] PyTorch 版本: {torch.__version__}")
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"  [-] GPU 状态: 不可用, 将使用 {device}")
    
    # 2. 模型验证
    print("\n[2/3] 正在从 Hugging Face 加载核心组件: 'bert-base-uncased'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        print("  [-] Hugging Face Transformers: 加载成功!")
    except Exception as e:
        print(f"  [x] 模型加载失败: {e}")
        print("  💡 提示: 如果出现网络连接错误，请参考 README 中的 [常见问题解答] 设置国内镜像。")
        return
        
    # 3. 攻击库验证
    print("\n[3/3] 检查对抗攻击库 TextAttack...")
    try:
        print(f"  [-] TextAttack 版本: {textattack.__version__}")
        print("  [-] TextAttack: 导入完成！")
    except Exception as e:
        print(f"  [x] TextAttack 导入失败: {e}")
        return
        
    print("\n✅ 所有环境依赖已全部就绪！闭环验证通过！")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
```

**运行方式**：
按下 `F5` 或点击 VS Code 左侧调试面板的绿色播放按钮运行该脚本。如果终端输出“**所有环境依赖已全部就绪！闭环验证通过！**”，则标志着实验环境搭建圆满结束，可进入下一阶段“基线攻击算法复现”。

---

## 五、 常见问题解答 (FAQ)

### Q1: 运行 `main.py` 时，Hugging Face 模型 (`bert-base-uncased`) 下载极慢或报错 `ConnectionError` 怎么办？
**原因**：由于 Hugging Face 官方服务器在海外，国内直连容易超时断开。
**解决方案**：使用官方支持的国内镜像加速。在 `main.py` 的顶部添加环境变量，或者在运行前在终端执行以下命令配置镜像：
* **Windows (CMD/Anaconda Prompt)**:
  ```cmd
  set HF_ENDPOINT=https://hf-mirror.com
  python main.py
  ```
* **PowerShell**:
  ```powershell
  $env:HF_ENDPOINT="https://hf-mirror.com"
  python main.py
  ```

### Q2: 提示 `ModuleNotFoundError: No module named 'textattack'`？
**原因**：VS Code 当前使用的解释器不是 `bert_attack` 虚拟环境。
**解决方案**：检查 VS Code 右下角的 Python 版本，确保它指向的是 `miniconda3/envs/bert_attack/python.exe`。如果不是，请点击它重新选择解释器。