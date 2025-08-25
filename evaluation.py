import argparse
import functools
import gc
import os
import platform
import re
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test.json", help="测试集的路径")
add_arg("model_path",  type=str, default="/home/jinhui/data/internvl_deploy/models/whisper-large-v3", help="合并模型的路径，确保是微调后的路径")
add_arg("batch_size",  type=int, default=2,        help="评估的batch size")
add_arg("num_workers", type=int, default=8,         help="读取数据的线程数量")
add_arg("language",    type=str, default="Cantonese", help="设置语言，可全称也可简写，如果为None则评估的是多语言")
add_arg("remove_pun",  type=bool, default=True,     help="是否移除标点符号")
add_arg("to_simple",   type=bool, default=False,     help="是否转为简体中文")
add_arg("timestamps",  type=bool, default=False,    help="评估时是否使用时间戳数据")
add_arg("min_audio_len", type=float, default=0.5,   help="最小的音频长度，单位秒")
add_arg("max_audio_len", type=float, default=30,    help="最大的音频长度，单位秒")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
add_arg("task",        type=str, default="transcribe", choices=['transcribe', 'translate'], help="模型的任务")
add_arg("metric",      type=str, default="cer",     choices=['cer', 'wer'], help="评估方式")
args = parser.parse_args()
print_arguments(args)

# If the platform is Windows, set num_workers to 0 to avoid multiprocessing issues
if platform.system() == "Windows":
    args.num_workers = 0


def clean_text(text):
    """
    Remove timestamp-like numbers and excess whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    # Remove standalone numbers (timestamps)
    text = re.sub(r'(?<!\w)\d+(?!\w)', '', text)
    # Remove numbers at the start of a string (e.g., "958 今次...")
    text = re.sub(r'^\d+\s*', '', text)
    # Replace multiple spaces with a single one and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # Get the Whisper data processor
    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=args.language if args.language.lower() != "none" else None,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only
    )
    # Set pad_token to avoid warnings
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Get the model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        local_files_only=args.local_files_only
    )
    if args.language and args.language.lower() != "none":
        model.generation_config.language = args.language.lower()
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = [1, 2, 7, 8, 9]
    model.eval()

    # Get the test dataset
    test_dataset = CustomDataset(
        data_list_path=args.test_data,
        processor=processor,
        timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len
    )
    print(f"Test data size: {len(test_dataset)}")

    # Data collator for padding
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    eval_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=data_collator
    )

    # Get the evaluation metric
    metric = evaluate.load(f'metrics/{args.metric}.py')

    # Start evaluation
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                # Ensure attention_mask is present
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is None:
                    attention_mask = (batch["input_features"] != 0).float().cuda()
                else:
                    attention_mask = attention_mask.cuda()
                
                # Generate tokens
                generated_tokens = model.generate(
                    input_features=batch["input_features"].cuda(),
                    attention_mask=attention_mask,
                    max_new_tokens=256,  # A fixed, reasonable length
                    num_beams=5,
                    no_repeat_ngram_size=2
                ).cpu().numpy()

                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

                # Convert to text and clean
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

                # Combine and filter predictions and labels in parallel
                processed_pairs = []
                for pred, label in zip(decoded_preds, decoded_labels):
                    pred = clean_text(pred)
                    label = clean_text(label)

                    if args.remove_pun:
                        pred = remove_punctuation([pred])[0]
                        label = remove_punctuation([label])[0]
                    if args.to_simple:
                        pred = to_simple([pred])[0]
                        label = to_simple([label])[0]
                    
                    if pred.strip() and label.strip():
                        processed_pairs.append((pred, label))
                
                # Unpack filtered pairs
                if processed_pairs:
                    filtered_preds, filtered_labels = zip(*processed_pairs)
                    # Add to metric
                    metric.add_batch(predictions=list(filtered_preds), references=list(filtered_labels))

                # Debug information for the first batch
                if step == 0:
                    print(f"Sample predictions: {filtered_preds[:2]}")
                    print(f"Sample references: {filtered_labels[:2]}")
                    print(f"Batch input features shape: {batch['input_features'].shape}")
                    print(f"Batch labels shape: {batch['labels'].shape}")
                    print(f"Batch attention mask shape: {attention_mask.shape}")

        # Clean up memory
        del generated_tokens, labels, batch
        gc.collect()

    # Compute and print final result
    m = metric.compute()
    print(f"Evaluation result: {args.metric}={round(m, 5) if m == m else 'NaN'}")

if __name__ == '__main__':
    main()