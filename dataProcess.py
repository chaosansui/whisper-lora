import os
import json
import tarfile
import zipfile
import soundfile
from pydub import AudioSegment
from tqdm import tqdm
import logging
import random
import re
import argparse  # Add this import

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def unpack(filepath, target_dir):
    try:
        if filepath.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=target_dir)
        elif filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        logger.info(f"解压完成 -> {target_dir}")
    except Exception as e:
        raise RuntimeError(f"解压失败: {str(e)}")

def check_audio(audio_path, expected_sr=16000):
    try:
        samples, sr = soundfile.read(audio_path)
        if sr != expected_sr:
            logger.warning(f"{audio_path} 的采样率 {sr} 不符合预期 {expected_sr}")
        duration = round(len(samples) / sr, 2)
        return duration
    except Exception as e:
        logger.error(f"读取音频文件 {audio_path} 失败: {str(e)}")
        return None

def preprocess_text(text):
    return " ".join(text.strip().split())

def split_audio(audio_path, output_dir, segment_length_ms=30000):
    try:
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        segments = []
        for i in range(0, duration_ms, segment_length_ms):
            segment = audio[i:i + segment_length_ms]
            output_path = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_{i//1000}.wav")
            segment.export(output_path, format="wav")
            segments.append((output_path, i/1000, (i + min(segment_length_ms, duration_ms - i))/1000))
        return segments
    except Exception as e:
        logger.error(f"切分音频 {audio_path} 失败: {str(e)}")
        return []

def parse_transcript_line(line):
    match = re.match(r'\[([\d.]+),([\d.]+)\]\s+(\w+)\s+(\w+)\s+(.+)', line.strip())
    if not match:
        raise ValueError(f"无效的转录行格式: {line}")
    start, end, _, _, text = match.groups()
    return float(start), float(end), preprocess_text(text)

def main():
    parser = argparse.ArgumentParser(description="解压音频和文本数据并生成JSON数据集")
    parser.add_argument("--filepath", required=True, help="压缩包路径（.tar.gz或.zip）")
    parser.add_argument("--target_dir", default="dataset", help="解压目录")
    parser.add_argument("--train_json", default="dataset/train.json", help="训练集JSON文件名")
    parser.add_argument("--test_json", default="dataset/test.json", help="测试集JSON文件名")
    parser.add_argument("--language", default="yue", help="语言（默认：yue）")
    parser.add_argument("--expected_sr", type=int, default=16000, help="预期采样率（默认：16000）")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        raise FileNotFoundError(f"文件不存在: {args.filepath}")

    os.makedirs(args.target_dir, exist_ok=True)
    unpack(args.filepath, args.target_dir)

    wav_dir = os.path.join(args.target_dir, "WAV")
    txt_dir = os.path.join(args.target_dir, "TXT")
    if not os.path.exists(wav_dir) or not os.path.exists(txt_dir):
        raise FileNotFoundError(f"解压后的目录结构不符合预期，未找到 WAV 或 TXT 目录")

    segment_dir = os.path.join(args.target_dir, "WAV_segments")
    os.makedirs(segment_dir, exist_ok=True)

    logger.info("正在生成数据集...")
    dataset = []
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

    for wav_file in tqdm(wav_files, desc="处理音频文件"):
        txt_file = wav_file.replace(".wav", ".txt")
        txt_path = os.path.join(txt_dir, txt_file)

        if not os.path.exists(txt_path):
            logger.warning(f"未找到对应的文本文件: {txt_path}")
            continue

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            sentences = []
            full_text = ""
            for line in lines:
                if not line.strip():
                    continue
                start, end, text = parse_transcript_line(line)
                sentences.append({"start": start, "end": end, "text": text})
                full_text += text + " "
            full_text = preprocess_text(full_text)
            if len(full_text) < 1 or len(full_text) > 20000:
                logger.warning(f"文本 {txt_path} 长度 {len(full_text)} 超出范围 [1, 20000]")
                continue
        except Exception as e:
            logger.error(f"处理文本文件 {txt_path} 失败: {str(e)}")
            continue

        audio_path = os.path.join(wav_dir, wav_file)
        duration = check_audio(audio_path)
        if duration is None:
            continue

        segments = split_audio(audio_path, segment_dir)
        for seg_path, start, end in segments:
            seg_duration = round(end - start, 2)
            if seg_duration < 0.5:
                logger.warning(f"音频片段 {seg_path} 时长 {seg_duration} 无效，跳过")
                continue
            seg_sentences = [
                {"start": max(0, s["start"] - start), "end": min(seg_duration, s["end"] - start), "text": s["text"]}
                for s in sentences if s["start"] < end and s["end"] > start
            ]
            if not seg_sentences:
                logger.warning(f"音频片段 {seg_path} 没有匹配的句子，跳过")
                continue
            seg_full_text = preprocess_text(" ".join(s["text"] for s in seg_sentences))
            if len(seg_full_text) < 1 or len(seg_full_text) > 20000:
                logger.warning(f"音频片段 {seg_path} 句子长度 {len(seg_full_text)} 超出范围 [1, 20000]")
                continue
            dataset.append({
                "audio": {"path": seg_path},
                "sentence": seg_full_text,
                "language": args.language,
                "sentences": seg_sentences,
                "duration": seg_duration
            })

    random.shuffle(dataset)
    split_index = int(len(dataset) * 0.9)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]

    with open(args.train_json, "w", encoding="utf-8") as f:
        for item in train_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"训练集已生成（JSON Lines 格式） -> {args.train_json}, 共 {len(train_dataset)} 个样本")

    with open(args.test_json, "w", encoding="utf-8") as f:
        for item in test_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"测试集已生成（JSON Lines 格式） -> {args.test_json}, 共 {len(test_dataset)} 个样本")

if __name__ == "__main__":
    main()