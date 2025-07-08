import argparse
import os
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast, WhisperProcessor
from peft import PeftModel, PeftConfig

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, required=True, help="本地基础模型路径")
parser.add_argument("--lora_model", type=str, required=True, help="LoRA适配器路径")
parser.add_argument("--output_dir", type=str, default="merged_model", help="合并后模型保存路径")
args = parser.parse_args()

# 检查路径是否存在
assert os.path.exists(args.base_model), f"基础模型路径 {args.base_model} 不存在"
assert os.path.exists(args.lora_model), f"LoRA路径 {args.lora_model} 不存在"

# 加载基础模型（强制使用CPU）
print(f"加载基础模型从: {args.base_model}")
base_model = WhisperForConditionalGeneration.from_pretrained(
    args.base_model,
    device_map={"": "cpu"},
    local_files_only=True
)

# 加载LoRA适配器
print(f"加载LoRA适配器从: {args.lora_model}")
model = PeftModel.from_pretrained(
    base_model,
    args.lora_model,
    device_map={"": "cpu"},
    local_files_only=True
)

# 合并模型
print("开始合并模型...")
merged_model = model.merge_and_unload()
merged_model.eval()  # 设置为评估模式

# 加载其他组件
print("加载处理器组件...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model, local_files_only=True)
tokenizer = WhisperTokenizerFast.from_pretrained(args.base_model, local_files_only=True)
processor = WhisperProcessor.from_pretrained(args.base_model, local_files_only=True)

# 保存合并后的模型
os.makedirs(args.output_dir, exist_ok=True)
print(f"保存合并模型到: {args.output_dir}")
merged_model.save_pretrained(args.output_dir, max_shard_size="4GB")
feature_extractor.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
processor.save_pretrained(args.output_dir)

print("✅ 合并完成！")