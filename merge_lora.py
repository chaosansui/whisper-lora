import argparse
import os
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast, WhisperProcessor
from peft import PeftModel

def validate_path(path, is_dir=False):
    """检查路径是否存在，并验证是否为目录（如果需要）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")
    if is_dir and not os.path.isdir(path):
        raise NotADirectoryError(f"路径不是目录: {path}")
    return path

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description="合并 Whisper 基础模型和 LoRA 适配器")
    parser.add_argument("--base_model", type=str, required=True, help="本地基础模型路径")
    parser.add_argument("--lora_model", type=str, required=True, help="LoRA适配器路径")
    parser.add_argument("--output_dir", type=str, default="merged_model", help="合并后模型保存路径")
    parser.add_argument("--local_files_only", action="store_true", help="是否仅使用本地模型（不下载）")
    args = parser.parse_args()

    # 检查路径
    args.base_model = validate_path(args.base_model, is_dir=True)
    args.lora_model = validate_path(args.lora_model, is_dir=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔄 加载基础模型从: {args.base_model}")
    try:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model,
            device_map={"": "cpu"},
            local_files_only=args.local_files_only
        )
    except Exception as e:
        print(f"❌ 加载基础模型失败: {e}")
        exit(1)

    print(f"🔄 加载LoRA适配器从: {args.lora_model}")
    try:
        model = PeftModel.from_pretrained(
            base_model,
            args.lora_model,
            device_map={"": "cpu"},
            local_files_only=args.local_files_only
        )
    except Exception as e:
        print(f"❌ 加载LoRA适配器失败: {e}")
        exit(1)

    # 合并模型
    print("⏳ 开始合并模型...")
    try:
        merged_model = model.merge_and_unload()
        merged_model.eval()
    except Exception as e:
        print(f"❌ 模型合并失败: {e}")
        exit(1)

    # 加载处理器
    print("🔄 加载处理器组件...")
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(args.base_model, local_files_only=args.local_files_only)
        tokenizer = WhisperTokenizerFast.from_pretrained(args.base_model, local_files_only=args.local_files_only)
        processor = WhisperProcessor.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    except Exception as e:
        print(f"❌ 加载处理器失败: {e}")
        exit(1)

    # 保存合并后的模型
    print(f"💾 保存合并模型到: {args.output_dir}")
    try:
        merged_model.save_pretrained(args.output_dir, max_shard_size="4GB")
        feature_extractor.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        exit(1)

    print("✅ 合并完成！输出目录:", os.path.abspath(args.output_dir))

if __name__ == "__main__":
    main()