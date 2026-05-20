import torch
from transformers import BertTokenizer, BertForSequenceClassification
import textattack

def final_check():
    """
    环境检查函数，展示所有核心组件都已就位。
    """
    
    print(" 环境检查开始...")
    

    # 1. 检查 PyTorch 和 GPU
    print(f" PyTorch 版本: {torch.__version__}")
    if torch.cuda.is_available():
        print("   GPU 状态:   可用")
    else:
        print("   GPU 状态:   不可用, 将使用 CPU ")

    # 2. 检查 Transformers (Hugging Face)
    try:
        model_name = "bert-base-uncased"
        print(f"\n正在从Hugging Face加载核心组件: '{model_name}'...")
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        print("   Hugging Face Transformers:  加载成功！")
    except Exception as e:
        print(f"   Hugging Face Transformers 加载失败: {e}")
        return

    # 3. 检查 TextAttack (你们项目后续的攻击库)
    print(f"\n检查对抗攻击库 TextAttack...")
    # 既然程序能运行到这里，就说明 textattack 已经导入成功了
    print("   TextAttack: 导入完成！")


   
    print("所有环境依赖已全部就位")
   

if __name__ == "__main__":
    final_check()