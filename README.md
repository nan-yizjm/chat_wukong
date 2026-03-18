# 孙悟空对话模型微调项目 (Chat Wukong)

本项目旨在基于 `Llama3-8B-Chinese-Chat` 大语言模型，利用《西游记》白话文数据构建的孙悟空对话数据集，通过 LoRA (Low-Rank Adaptation) 技术进行微调，训练出一个能够模仿孙悟空语气、性格和知识背景的对话机器人。

## 📁 项目文件说明

- **`train.ipynb`**: 核心训练脚本。包含数据加载、预处理、模型配置（4-bit 量化 + LoRA）、训练循环及推理测试的完整代码。
- **`悟空_孙悟空对话_规范.json`**: 微调训练数据集。包含 73 条精心构造的指令微调数据，格式为 `instruction` (指令/上文), `input` (输入，通常为空), `output` (孙悟空的回答)。
- **`西游记白话文.txt`**: 原始语料库。包含了《西游记》的全本白话文故事，用于提取对话逻辑和背景知识（本项目中主要作为数据构建的源头参考）。

## 🛠️ 环境依赖

请确保您的环境中安装了以下主要库：

```bash
torch
transformers
peft
datasets
modelscope
bitsandbytes
pandas
accelerate
```

建议使用 Python 3.10+ 版本，并配备 NVIDIA GPU (显存建议 16GB 以上以支持 4-bit 量化训练)。

## 🚀 快速开始

### 1. 数据准备
项目已包含处理好的数据集 `悟空_孙悟空对话_规范.json`。如果需要重新构建数据，可参考 `西游记白话文.txt` 进行提取。

数据格式示例：
```json
[
    {
        "instruction": "嫌那口大刀太轻，不好用。",
        "input": "",
        "output": "嫌它太轻。"
    },
    {
        "instruction": "师父快救我出去，我保护你到西天取经。",
        "input": "",
        "output": "老孙已经五百多年没有用过这宝贝了，今天用它弄件衣服穿穿！"
    }
]
```

### 2. 模型训练
运行 `train.ipynb` 中的训练单元格。主要步骤包括：
- **模型下载**: 自动从 ModelScope 下载 `LLM-Research/Llama3-8B-Chinese-Chat`。
- **量化配置**: 使用 `BitsAndBytesConfig` 进行 4-bit 量化以节省显存。
- **LoRA 配置**: 针对 `q_proj`, `k_proj`, `v_proj` 等模块注入 LoRA 适配器。
- **训练参数**: 
  - Epochs: 3
  - Batch Size: 1 (梯度累积步数: 2)
  - Learning Rate: 1e-4
  - Max Length: 128

```python
# 核心训练代码片段 (来自 train.ipynb)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
```

### 3. 模型推理
训练完成后，使用 `train.ipynb` 中的推理单元格加载微调后的权重进行测试。

```python
# 推理示例
prompt = "孙悟空,我要开始念咒了"
messages = [
    {"role": "system", "content": "假设你是孙悟空。"},
    {"role": "user", "content": prompt}
]
# ... (加载模型与 LoRA 权重)
# 输出示例:
# 唐僧：孙悟空,我要开始念咒了
# 孙悟空：你念咒吧！我不怕！如果你能念得动我的金箍棒，那就再好不过了！快点念吧！别等了！别等了！
```

## 📂 输出目录结构

训练过程中会生成以下目录（默认路径）：
```text
./output/llama3_1_instruct_lora/
├── checkpoint-50/
├── checkpoint-100/
└── ... (保存的 LoRA 适配器权重)
```

## ⚙️ 技术细节

- **基座模型**: Llama3-8B-Chinese-Chat
- **微调方法**: QLoRA (4-bit Quantization + LoRA)
- **显存优化**: 
  - 开启梯度检查点 (`gradient_checkpointing`)
  - 使用 `paged_adamw_8bit` 优化器
  - 禁用 KV Cache (`use_cache=False`)
- **数据处理**: 自动识别 `instruction/output` 字段并拼接为特定模板格式。

## 📝 注意事项

1. **显存要求**: 尽管使用了 4-bit 量化，训练仍需约 12-16GB 显存。如果显存不足，可尝试减小 `per_device_train_batch_size` 或 `max_length`。
2. **路径配置**: 代码中使用了绝对路径 (如 `/home/nanyi/...`)，使用时请根据实际环境修改 `model_dir` 和数据集路径。
3. **代理设置**: 脚本中包含了清除代理环境变量的代码，以确保在部分网络环境下能正常下载模型。

## 📄 许可证

本项目仅供学习和研究使用。基座模型遵循其原有的开源协议。

---
*Generated based on project files: `train.ipynb`, `悟空_孙悟空对话_规范.json`, `西游记白话文.txt`*
