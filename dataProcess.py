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
import argparse

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

def resample_audio_to_16k(audio_path, output_path, target_sr=16000):
    """统一将音频采样率转换为16000Hz"""
    try:
        # 根据文件扩展名选择读取方式
        if audio_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_path)
        elif audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_wav(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)
        
        # 转换采样率
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
            logger.info(f"音频 {audio_path} 采样率从 {audio.frame_rate} 转换为 {target_sr}")
        
        # 转换为单声道
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info(f"音频 {audio_path} 从 {audio.channels} 声道转换为单声道")
        
        # 导出为WAV格式
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        logger.error(f"重采样音频 {audio_path} 失败: {str(e)}")
        return False

def check_audio(audio_path, expected_sr=16000):
    """检查音频文件并返回时长"""
    try:
        samples, sr = soundfile.read(audio_path)
        duration = round(len(samples) / sr, 2)
        
        if sr != expected_sr:
            logger.warning(f"{audio_path} 的采样率 {sr} 不符合预期 {expected_sr}")
            return None
        
        return duration
    except Exception as e:
        logger.error(f"读取音频文件 {audio_path} 失败: {str(e)}")
        return None

def preprocess_text(text):
    """预处理文本，去除多余空格"""
    return " ".join(text.strip().split())

def parse_timestamp(timestamp_str):
    """解析时间戳 [HH:MM:SS] 格式"""
    try:
        # 去掉方括号
        timestamp_str = timestamp_str.strip('[]')
        parts = timestamp_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"时间戳格式错误: {timestamp_str}")
    except Exception as e:
        raise ValueError(f"解析时间戳失败: {timestamp_str} - {str(e)}")

def parse_transcript_content(content):
    """解析对话转录内容"""
    lines = content.strip().split('\n')
    parsed_segments = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 匹配格式: [HH:MM:SS] 说话人: 内容
        match = re.match(r'(\[\d{2}:\d{2}:\d{2}\])\s+([^:]+):\s*(.+)', line)
        if match:
            timestamp_str, speaker, text = match.groups()
            try:
                start_time = parse_timestamp(timestamp_str)
                text = preprocess_text(text)
                if text:  # 确保文本不为空
                    parsed_segments.append({
                        'start': start_time,
                        'speaker': speaker.strip(),
                        'text': text
                    })
            except ValueError as e:
                logger.warning(f"跳过无效行: {line} - {e}")
                continue
    
    return parsed_segments

def merge_segments_to_sentences(segments):
    """将对话片段合并为完整的句子，计算结束时间"""
    if not segments:
        return []
    
    sentences = []
    for i, segment in enumerate(segments):
        start_time = segment['start']
        text = segment['text']
        
        # 计算结束时间
        if i < len(segments) - 1:
            # 使用下一个片段的开始时间作为当前片段的结束时间
            end_time = segments[i + 1]['start']
        else:
            # 最后一个片段，假设持续3秒
            end_time = start_time + 3.0
        
        sentences.append({
            'start': float(start_time),
            'end': float(end_time),
            'text': text
        })
    
    return sentences

def split_audio_by_duration(audio_path, output_dir, max_duration=30.0):
    """按时长切分音频文件"""
    try:
        audio = AudioSegment.from_wav(audio_path)
        total_duration = len(audio) / 1000.0  # 转换为秒
        
        if total_duration <= max_duration:
            # 如果音频时长不超过最大时长，直接复制
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_0.wav")
            audio.export(output_path, format="wav")
            return [(output_path, 0.0, total_duration)]
        
        # 需要切分
        segments = []
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        segment_length_ms = int(max_duration * 1000)
        
        for i in range(0, len(audio), segment_length_ms):
            segment = audio[i:i + segment_length_ms]
            start_time = i / 1000.0
            end_time = min((i + segment_length_ms) / 1000.0, total_duration)
            
            output_path = os.path.join(output_dir, f"{base_name}_{int(start_time)}.wav")
            segment.export(output_path, format="wav")
            segments.append((output_path, start_time, end_time))
        
        return segments
    except Exception as e:
        logger.error(f"切分音频 {audio_path} 失败: {str(e)}")
        return []

def validate_data_format(data_item):
    """验证数据格式是否符合要求"""
    try:
        # 检查必需字段
        required_fields = ["audio", "sentence", "language", "sentences", "duration"]
        for field in required_fields:
            if field not in data_item:
                return False, f"缺少必需字段: {field}"
        
        # 检查audio字段
        if "path" not in data_item["audio"]:
            return False, "audio字段缺少path"
        
        # 检查sentence字段
        if not isinstance(data_item["sentence"], str) or len(data_item["sentence"]) == 0:
            return False, "sentence字段无效"
        
        # 检查language字段
        if not isinstance(data_item["language"], str) or len(data_item["language"]) == 0:
            return False, "language字段无效"
        
        # 检查sentences字段
        if not isinstance(data_item["sentences"], list) or len(data_item["sentences"]) == 0:
            return False, "sentences字段必须是非空列表"
        
        for sentence in data_item["sentences"]:
            if not isinstance(sentence, dict):
                return False, "sentences中的每个元素必须是字典"
            
            sentence_fields = ["start", "end", "text"]
            for field in sentence_fields:
                if field not in sentence:
                    return False, f"sentences中缺少字段: {field}"
            
            # 检查时间戳
            if not isinstance(sentence["start"], (int, float)) or not isinstance(sentence["end"], (int, float)):
                return False, "start和end必须是数字"
            
            if sentence["start"] < 0 or sentence["end"] < 0 or sentence["start"] > sentence["end"]:
                return False, "时间戳无效"
            
            # 检查文本
            if not isinstance(sentence["text"], str) or len(sentence["text"]) == 0:
                return False, "sentence文本无效"
        
        # 检查duration字段
        if not isinstance(data_item["duration"], (int, float)) or data_item["duration"] <= 0:
            return False, "duration字段无效"
        
        return True, "格式正确"
    
    except Exception as e:
        return False, f"验证时发生错误: {str(e)}"

def find_audio_text_dirs(target_dir):
    """自动查找音频和文本目录"""
    wav_dir = None
    txt_dir = None
    
    # 查找可能的目录名
    possible_audio_names = ['WAV', 'wav', '语音', 'audio', 'Audio']
    possible_text_names = ['TXT', 'txt', '文本', 'text', 'Text']
    
    def search_in_dir(base_dir):
        """在指定目录中查找音频和文本目录"""
        found_wav = None
        found_txt = None
        
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                if item in possible_audio_names:
                    found_wav = item_path
                elif item in possible_text_names:
                    found_txt = item_path
        
        return found_wav, found_txt
    
    # 首先在target_dir直接查找
    wav_dir, txt_dir = search_in_dir(target_dir)
    
    if wav_dir and txt_dir:
        return wav_dir, txt_dir
    
    # 如果没找到，在子目录中查找
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            sub_wav, sub_txt = search_in_dir(item_path)
            if sub_wav and sub_txt:
                wav_dir = sub_wav
                txt_dir = sub_txt
                logger.info(f"找到音频目录: {wav_dir}")
                logger.info(f"找到文本目录: {txt_dir}")
                break
    
    return wav_dir, txt_dir

def main():
    parser = argparse.ArgumentParser(description="处理对话音频和文本数据并生成JSON数据集")
    parser.add_argument("--filepath", required=True, help="压缩包路径（.tar.gz或.zip）")
    parser.add_argument("--target_dir", default="dataset", help="解压目录")
    parser.add_argument("--train_json", default="dataset/train.json", help="训练集JSON文件名")
    parser.add_argument("--test_json", default="dataset/test.json", help="测试集JSON文件名")
    parser.add_argument("--language", default="Chinese", help="语言（默认：Chinese）")
    parser.add_argument("--expected_sr", type=int, default=16000, help="预期采样率（默认：16000）")
    parser.add_argument("--max_duration", type=float, default=30.0, help="最大音频片段时长（秒，默认：30）")
    args = parser.parse_args()

    if not os.path.exists(args.filepath):
        raise FileNotFoundError(f"文件不存在: {args.filepath}")

    os.makedirs(args.target_dir, exist_ok=True)
    unpack(args.filepath, args.target_dir)

    # 自动检测解压后的目录结构
    wav_dir, txt_dir = find_audio_text_dirs(args.target_dir)
    
    if wav_dir is None or txt_dir is None:
        logger.error("解压后的目录结构不符合预期")
        logger.error("当前目录结构:")
        for root, dirs, files in os.walk(args.target_dir):
            level = root.replace(args.target_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            logger.error(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for f in files[:5]:  # 只显示前5个文件
                logger.error(f"{subindent}{f}")
            if len(files) > 5:
                logger.error(f"{subindent}... 还有 {len(files) - 5} 个文件")
        raise FileNotFoundError("未找到音频或文本目录")

    # 创建处理后的音频目录
    processed_audio_dir = os.path.join(args.target_dir, "processed_audio")
    segment_dir = os.path.join(args.target_dir, "audio_segments")
    os.makedirs(processed_audio_dir, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)

    logger.info("正在生成数据集...")
    dataset = []
    
    # 支持多种音频格式
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend([f for f in os.listdir(wav_dir) if f.lower().endswith(ext)])
    
    logger.info(f"找到 {len(audio_files)} 个音频文件")

    for audio_file in tqdm(audio_files, desc="处理音频文件"):
        # 根据音频文件名查找对应的文本文件
        base_name = os.path.splitext(audio_file)[0]
        txt_file = base_name + ".txt"
        txt_path = os.path.join(txt_dir, txt_file)

        if not os.path.exists(txt_path):
            logger.warning(f"未找到对应的文本文件: {txt_path}")
            continue

        # 首先统一处理音频采样率
        original_audio_path = os.path.join(wav_dir, audio_file)
        processed_audio_path = os.path.join(processed_audio_dir, base_name + ".wav")
        
        if not resample_audio_to_16k(original_audio_path, processed_audio_path, args.expected_sr):
            logger.warning(f"音频重采样失败，跳过: {audio_file}")
            continue

        # 检查处理后的音频
        total_duration = check_audio(processed_audio_path, args.expected_sr)
        if total_duration is None:
            logger.warning(f"音频检查失败，跳过: {processed_audio_path}")
            continue

        try:
            # 读取并解析文本文件
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 解析对话转录内容
            segments = parse_transcript_content(content)
            if not segments:
                logger.warning(f"文本文件 {txt_path} 没有有效的对话内容")
                continue
            
            # 合并为完整的句子
            sentences = merge_segments_to_sentences(segments)
            if not sentences:
                logger.warning(f"文本文件 {txt_path} 无法生成有效的句子")
                continue
            
            # 生成完整的句子文本
            full_text = " ".join(s["text"] for s in sentences)
            full_text = preprocess_text(full_text)
            
            if len(full_text) < 1 or len(full_text) > 20000:
                logger.warning(f"文本 {txt_path} 长度 {len(full_text)} 超出范围 [1, 20000]")
                continue
                
        except Exception as e:
            logger.error(f"处理文本文件 {txt_path} 失败: {str(e)}")
            continue

        # 按时长切分音频
        audio_segments = split_audio_by_duration(processed_audio_path, segment_dir, args.max_duration)
        
        for seg_path, seg_start, seg_end in audio_segments:
            seg_duration = round(seg_end - seg_start, 2)
            if seg_duration < 0.5:
                logger.warning(f"音频片段 {seg_path} 时长 {seg_duration} 过短，跳过")
                continue
            
            # 找到该音频片段时间范围内的句子
            seg_sentences = []
            for sentence in sentences:
                s_start = sentence["start"]
                s_end = sentence["end"]
                
                # 检查句子是否与音频片段有重叠
                if s_start < seg_end and s_end > seg_start:
                    # 调整句子的时间戳相对于音频片段开始时间
                    adjusted_start = max(0, s_start - seg_start)
                    adjusted_end = min(seg_duration, s_end - seg_start)
                    
                    # 确保调整后的时间戳有效
                    if adjusted_end > adjusted_start:
                        seg_sentences.append({
                            "start": round(adjusted_start, 2),
                            "end": round(adjusted_end, 2),
                            "text": sentence["text"]
                        })
            
            if not seg_sentences:
                logger.warning(f"音频片段 {seg_path} 没有匹配的句子，跳过")
                continue
            
            # 生成该片段的完整文本
            seg_full_text = " ".join(s["text"] for s in seg_sentences)
            seg_full_text = preprocess_text(seg_full_text)
            
            if len(seg_full_text) < 1 or len(seg_full_text) > 20000:
                logger.warning(f"音频片段 {seg_path} 句子长度 {len(seg_full_text)} 超出范围 [1, 20000]")
                continue
            
            # 创建数据项
            data_item = {
                "audio": {"path": seg_path},
                "sentence": seg_full_text,
                "language": args.language,
                "sentences": seg_sentences,
                "duration": seg_duration
            }
            
            # 验证数据格式
            is_valid, error_msg = validate_data_format(data_item)
            if not is_valid:
                logger.warning(f"数据格式验证失败: {seg_path} - {error_msg}")
                continue
            
            dataset.append(data_item)

    logger.info(f"成功处理 {len(dataset)} 个音频片段")
    
    if len(dataset) == 0:
        logger.error("没有生成任何有效的数据项，请检查输入文件格式")
        return
    
    # 打乱数据集
    random.shuffle(dataset)
    split_index = int(len(dataset) * 0.9)
    train_dataset = dataset[:split_index]
    test_dataset = dataset[split_index:]

    # 保存训练集
    with open(args.train_json, "w", encoding="utf-8") as f:
        for item in train_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"训练集已生成（JSON Lines 格式） -> {args.train_json}, 共 {len(train_dataset)} 个样本")

    # 保存测试集
    with open(args.test_json, "w", encoding="utf-8") as f:
        for item in test_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    logger.info(f"测试集已生成（JSON Lines 格式） -> {args.test_json}, 共 {len(test_dataset)} 个样本")

    # 输出格式示例
    if len(dataset) > 0:
        logger.info("数据格式示例:")
        print(json.dumps(dataset[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()