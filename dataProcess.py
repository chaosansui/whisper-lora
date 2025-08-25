import os
import json
import tarfile
import zipfile
import soundfile
from pydub import AudioSegment
from tqdm import tqdm
import logging
import re
import argparse
import librosa
import shutil
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import list_repo_files

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def unpack(filepath, target_dir):
    """解压压缩文件到目标目录"""
    try:
        os.makedirs(target_dir, exist_ok=True)
        if filepath.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=target_dir)
        elif filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        else:
            raise ValueError("不支持的压缩包格式，请使用 .tar.gz 或 .zip")
        logger.info(f"解压完成 -> {target_dir}")
    except Exception as e:
        logger.error(f"解压失败: {str(e)}")
        raise RuntimeError(f"解压失败: {str(e)}")

def parse_timestamp(timestamp_str):
    """解析时间戳字符串，支持 HH:MM:SS, MM:SS, SS 格式"""
    try:
        clean_str = re.sub(r'[^\d:]', '', timestamp_str)
        parts = clean_str.split(':')
        
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        elif len(parts) == 1:
            return int(parts[0])
        else:
            raise ValueError("无法识别的时间戳格式")
    except Exception as e:
        logger.warning(f"时间戳解析失败: {timestamp_str} - {str(e)}")
        raise ValueError(f"时间戳解析失败: {timestamp_str} - {str(e)}")

def preprocess_text(text):
    """
    清理文本，保留中文、英文、数字和常见标点。
    """
    if not text:
        return ""
    
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。？！、：；""''（）《》【】.,!?]', '', text)
    text = ' '.join(text.strip().split())
    
    return text

def count_text_length(text, method='char'):
    """计算文本长度"""
    if method == 'char':
        return len(text)
    elif method == 'word':
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return chinese_chars + english_words
    elif method == 'token':
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        return chinese_chars + english_chars // 4
    else:
        return len(text)

def split_sentences_by_length(sentences, max_length=200, method='char'):
    """按句子长度切分句子组"""
    if not sentences:
        return []
    
    groups = []
    current_group = []
    current_length = 0
    
    for sentence in sentences:
        text = sentence['text']
        text_length = count_text_length(text, method)
        
        if text_length > max_length:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_length = 0
            
            split_sentences = split_long_sentence(sentence, max_length, method)
            groups.extend([[s] for s in split_sentences])
            continue
        
        if current_length + text_length > max_length and current_group:
            groups.append(current_group)
            current_group = [sentence]
            current_length = text_length
        else:
            current_group.append(sentence)
            current_length += text_length
    
    if current_group:
        groups.append(current_group)
    
    logger.info(f"句子切分完成: {len(sentences)} 句 -> {len(groups)} 组")
    return groups

def split_long_sentence(sentence, max_length, method='char'):
    """切分过长的单个句子"""
    text = sentence['text']
    if count_text_length(text, method) <= max_length:
        return [sentence]
    
    delimiters = ['。', '！', '？', '.', '!', '?', '；', ';', '，', ',']
    parts = []
    current_part = ""
    
    for char in text:
        current_part += char
        if char in delimiters:
            if count_text_length(current_part, method) <= max_length:
                parts.append(current_part.strip())
                current_part = ""
            else:
                if parts:
                    parts[-1] += current_part
                else:
                    parts.append(current_part.strip())
                current_part = ""
    
    if current_part.strip():
        if parts and count_text_length(parts[-1] + current_part, method) <= max_length:
            parts[-1] += current_part
        else:
            parts.append(current_part.strip())
    
    result = []
    duration = sentence['end'] - sentence['start']
    part_duration = duration / len(parts) if parts else duration
    
    for i, part in enumerate(parts):
        if part.strip():
            new_sentence = {
                'start': sentence['start'] + i * part_duration,
                'end': sentence['start'] + (i + 1) * part_duration,
                'text': part.strip()
            }
            if 'speaker' in sentence:
                new_sentence['speaker'] = sentence['speaker']
            result.append(new_sentence)
    
    return result

def parse_transcript_content(content, filename):
    """解析文本内容，支持 AF 文件（带客服/客人标签）和 youtube 文件（无标签）"""
    logger.info(f"解析文件: {filename}")
    logger.debug(f"文件内容:\n{content}")
    lines = content.strip().split('\n')
    parsed_segments = []
    current_time = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        logger.debug(f"处理行: {line}")
        
        if filename.startswith("AF"):
            match = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s+(客服|客人):\s*(.+)', line)
            if match:
                timestamp_str, speaker, text = match.groups()
                try:
                    current_time = parse_timestamp(timestamp_str)
                    if text := text.strip():
                        parsed_segments.append({
                            'start': current_time,
                            'text': preprocess_text(text),
                            'speaker': speaker
                        })
                    continue
                except ValueError as e:
                    logger.warning(f"时间戳解析失败: {line} - {e}")
        
        elif filename.startswith("youtube"):
            match = re.match(r'\[?(\d{2}:\d{2}:\d{2})\]?\s*(.*)', line)
            if match:
                timestamp_str, raw_text = match.groups()
                try:
                    current_time = parse_timestamp(timestamp_str)
                    cleaned_text = re.sub(r'^\s*\d+\s*', '', raw_text, count=1) 
                    
                    if cleaned_text := cleaned_text.strip():
                        parsed_segments.append({
                            'start': current_time,
                            'text': preprocess_text(cleaned_text)
                        })
                    continue
                except ValueError as e:
                    logger.warning(f"时间戳解析失败: {line} - {e}")
        
        if line:
            parsed_segments.append({
                'start': current_time,
                'text': preprocess_text(line)
            })
            current_time += 3.0
    
    final_segments = []
    for segment in parsed_segments:
        text = segment.get('text', '')
        cleaned_text = re.sub(r'^\d+\s*', '', text, count=1)
        segment['text'] = cleaned_text
        if segment['text'].strip():
            final_segments.append(segment)

    if not final_segments:
        logger.warning(f"无有效内容解析: {filename}")
    return final_segments

def resample_audio_to_16k(audio_path, output_path, target_sr=16000):
    """将音频重采样为 16kHz 单声道"""
    try:
        try:
            audio = AudioSegment.from_file(audio_path)
            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            audio.export(output_path, format="wav")
            logger.info(f"音频重采样成功: {output_path}")
            return True
        except Exception as e:
            logger.warning(f"pydub 处理失败，尝试备用方案: {audio_path} - {str(e)}")
            try:
                data, sr = soundfile.read(audio_path)
                if sr != target_sr:
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                soundfile.write(output_path, data, target_sr)
                logger.info(f"音频重采样成功: {output_path}")
                return True
            except Exception as e:
                logger.error(f"soundfile 处理失败: {audio_path} - {str(e)}")
                return False
    except Exception as e:
        logger.error(f"音频处理异常: {audio_path} - {str(e)}")
        return False

def check_audio(audio_path, expected_sr=16000):
    """检查音频采样率和时长"""
    try:
        samples, sr = soundfile.read(audio_path)
        duration = round(len(samples) / sr, 2)
        if sr != expected_sr:
            logger.warning(f"音频采样率不匹配: {audio_path}, 期望 {expected_sr}, 实际 {sr}")
            return None
        logger.info(f"音频检查通过: {audio_path}, 时长 {duration}秒")
        return duration
    except Exception as e:
        logger.error(f"读取音频失败: {audio_path} - {str(e)}")
        return None

def merge_segments_to_sentences(segments):
    """将文本片段合并为句子，添加结束时间"""
    if not segments:
        return []
    
    sentences = []
    for i, segment in enumerate(segments):
        start_time = segment['start']
        text = segment['text']
        speaker = segment.get('speaker', None)
        end_time = segments[i+1]['start'] if i < len(segments)-1 else start_time + 3.0
        
        sentence = {
            'start': float(start_time),
            'end': float(end_time),
            'text': text
        }
        if speaker:
            sentence['speaker'] = speaker
        sentences.append(sentence)
    
    return sentences

def split_audio_by_sentences(audio_path, sentence_groups, output_dir):
    """根据句子组切分音频"""
    try:
        audio = AudioSegment.from_wav(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        segments = []
        
        for i, group in enumerate(sentence_groups):
            if not group:
                continue
                
            start_time = group[0]['start']
            end_time = group[-1]['end']
            
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            end_ms = min(end_ms, len(audio))
            
            if end_ms <= start_ms:
                continue
                
            segment = audio[start_ms:end_ms]
            duration = (end_ms - start_ms) / 1000.0
            
            if duration < 0.5:
                continue
                
            output_path = os.path.join(output_dir, f"{base_name}_sent_{i}.wav")
            segment.export(output_path, format="wav")
            segments.append((output_path, start_time, end_time, group))
            logger.info(f"生成句子音频片段: {output_path}, {start_time}秒 - {end_time}秒")
        
        return segments
    except Exception as e:
        logger.error(f"句子切分音频失败: {audio_path} - {str(e)}")
        return []

def split_audio_by_duration(audio_path, output_dir, max_duration=30.0):
    """按最大时长切分音频"""
    try:
        audio = AudioSegment.from_wav(audio_path)
        total_duration = len(audio) / 1000.0
        
        if total_duration <= max_duration:
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_0.wav")
            audio.export(output_path, format="wav")
            logger.info(f"音频未切分: {output_path}, 时长 {total_duration}秒")
            return [(output_path, 0.0, total_duration)]
        
        segments = []
        segment_length_ms = int(max_duration * 1000)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        for i in range(0, len(audio), segment_length_ms):
            segment = audio[i:i + segment_length_ms]
            start_time = i / 1000.0
            end_time = min((i + segment_length_ms) / 1000.0, total_duration)
            output_path = os.path.join(output_dir, f"{base_name}_{int(start_time)}.wav")
            segment.export(output_path, format="wav")
            segments.append((output_path, start_time, end_time))
            logger.info(f"生成音频片段: {output_path}, {start_time}秒 - {end_time}秒")
        
        return segments
    except Exception as e:
        logger.error(f"切分音频失败: {audio_path} - {str(e)}")
        return []

def validate_data_format(data_item):
    """验证数据项的格式"""
    required_fields = ["audio", "sentence", "language", "sentences", "duration"]
    for field in required_fields:
        if field not in data_item:
            return False, f"缺少字段: {field}"
    
    if not isinstance(data_item["sentence"], str) or not data_item["sentence"]:
        return False, "无效句子"
    
    if not isinstance(data_item["sentences"], list) or not data_item["sentences"]:
        return False, "无效句子列表"
    
    for sentence in data_item["sentences"]:
        if not all(k in sentence for k in ["start", "end", "text"]):
            return False, "句子缺少字段"
        if sentence["start"] < 0 or sentence["end"] <= sentence["start"]:
            return False, "无效时间戳"
    
    return True, "格式正确"

def find_audio_text_dirs(target_dir):
    """查找音频和文本目录"""
    possible_audio = ['WAV', 'wav', 'audio', 'Audio']
    possible_text = ['TXT', 'txt', 'text', 'Text']
    
    def search_dir(base_dir):
        wav, txt = None, None
        for item in os.listdir(base_dir):
            path = os.path.join(base_dir, item)
            if os.path.isdir(path):
                if item in possible_audio:
                    wav = path
                elif item in possible_text:
                    txt = path
        return wav, txt
    
    wav_dir, txt_dir = search_dir(target_dir)
    if not (wav_dir and txt_dir):
        for item in os.listdir(target_dir):
            path = os.path.join(target_dir, item)
            if os.path.isdir(path):
                sub_wav, sub_txt = search_dir(path)
                if sub_wav and sub_txt:
                    wav_dir, txt_dir = sub_wav, sub_txt
                    break
    
    if wav_dir and txt_dir:
        logger.info(f"找到音频目录: {wav_dir}")
        logger.info(f"找到文本目录: {txt_dir}")
    else:
        logger.error("未找到音频或文本目录")
    
    return wav_dir, txt_dir

# 新增的函数
def load_huggingface_dataset(dataset_name, subset=None, split="train", **kwargs):
    """从 Hugging Face Hub 加载一个数据集"""
    logger.info(f"尝试从 Hugging Face Hub 加载数据集: {dataset_name} ({subset or ''})")
    try:
        # 使用流式加载以节省内存，对于大型数据集特别有用
        dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            streaming=False,
            **kwargs
        )
        logger.info(f"成功加载 Hugging Face 数据集。数据条数: {len(dataset)}")
        return dataset
    except Exception as e:
        logger.error(f"加载 Hugging Face 数据集失败: {str(e)}")
        return None

def process_and_save_dataset(args):
    """核心处理逻辑，处理音频和文本，生成并保存JSON数据集"""
    try:
        # 获取文件名（不带扩展名），作为子目录名称
        zip_base_name = Path(args.input_path).stem
        target_sub_dir = os.path.join(args.target_dir, zip_base_name)
        
        # 检查子目录是否已存在，如果存在则跳过解压
        if os.path.exists(target_sub_dir):
            logger.info(f"目录 {target_sub_dir} 已存在，跳过解压。")
        else:
            unpack(args.input_path, target_sub_dir)

        wav_dir, txt_dir = find_audio_text_dirs(target_sub_dir)
        if not wav_dir or not txt_dir:
            raise FileNotFoundError(f"未在 {target_sub_dir} 中找到音频或文本目录")

        processed_audio_dir = os.path.join(target_sub_dir, "processed_audio")
        segment_dir = os.path.join(target_sub_dir, "audio_segments")
        os.makedirs(processed_audio_dir, exist_ok=True)
        os.makedirs(segment_dir, exist_ok=True)

        dataset = []
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.WAV', '.MP3']
        audio_files = [f for f in os.listdir(wav_dir) 
                      if os.path.splitext(f)[1].lower() in audio_extensions]
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        logger.info(f"切分方法: {args.split_method}")

        for audio_file in tqdm(audio_files, desc="处理进度"):
            base_name = os.path.splitext(audio_file)[0]
            txt_file = base_name + ".txt"
            txt_path = os.path.join(txt_dir, txt_file)

            if not os.path.exists(txt_path):
                logger.warning(f"未找到文本文件: {txt_path}")
                continue

            audio_path = os.path.join(wav_dir, audio_file)
            processed_path = os.path.join(processed_audio_dir, f"{base_name}.wav")
            
            if not resample_audio_to_16k(audio_path, processed_path, args.expected_sr):
                logger.warning(f"音频处理失败，跳过: {audio_file}")
                continue

            duration = check_audio(processed_path, args.expected_sr)
            if not duration:
                logger.warning(f"音频检查失败，跳过: {audio_file}")
                continue

            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                segments = parse_transcript_content(content, base_name)
                if not segments:
                    logger.warning(f"无有效内容: {txt_path}")
                    continue
                
                sentences = merge_segments_to_sentences(segments)
                if not sentences:
                    logger.warning(f"无法生成句子: {txt_path}")
                    continue
                
                full_text = " ".join(s["text"] for s in sentences)
                if not (1 <= len(full_text) <= 20000):
                    logger.warning(f"文本长度异常: {txt_path}, 长度 {len(full_text)}")
                    continue
                    
            except Exception as e:
                logger.error(f"处理文本失败: {txt_path} - {str(e)}")
                continue

            if args.split_method == 'sentence':
                sentence_groups = split_sentences_by_length(
                    sentences, args.max_sentence_length, args.length_method)
                audio_segments = split_audio_by_sentences(processed_path, sentence_groups, segment_dir)
                
                for seg_path, seg_start, seg_end, group in audio_segments:
                    seg_duration = seg_end - seg_start
                    seg_sentences = []
                    
                    for s in group:
                        adjusted_start = max(0, s["start"] - seg_start)
                        adjusted_end = min(seg_duration, s["end"] - seg_start)
                        seg_sentences.append({
                            "start": round(adjusted_start, 2),
                            "end": round(adjusted_end, 2),
                            "text": s["text"],
                            "speaker": s.get("speaker", None)
                        })
                    
                    seg_text = " ".join(s["text"] for s in seg_sentences)
                    data_item = {
                        "audio": {"path": seg_path},
                        "sentence": seg_text,
                        "language": args.language,
                        "sentences": seg_sentences,
                        "duration": round(seg_duration, 2)
                    }
                    
                    is_valid, msg = validate_data_format(data_item)
                    if is_valid:
                        dataset.append(data_item)
                        logger.debug(f"添加句子切分数据项: {seg_path}")
                    else:
                        logger.warning(f"数据格式验证失败: {seg_path} - {msg}")
                        
            elif args.split_method == 'duration':
                audio_segments = split_audio_by_duration(processed_path, segment_dir, args.max_duration)
                
                for seg_path, seg_start, seg_end in audio_segments:
                    seg_duration = seg_end - seg_start
                    if seg_duration < 0.5:
                        logger.debug(f"跳过过短片段: {seg_path}, 时长 {seg_duration}秒")
                        continue
                    
                    seg_sentences = []
                    for s in sentences:
                        if s["start"] < seg_end and s["end"] > seg_start:
                            adjusted_start = max(0, s["start"] - seg_start)
                            adjusted_end = min(seg_duration, s["end"] - seg_start)
                            if adjusted_end > adjusted_start:
                                seg_sentences.append({
                                    "start": round(adjusted_start, 2),
                                    "end": round(adjusted_end, 2),
                                    "text": s["text"],
                                    "speaker": s.get("speaker", None)
                                })
                    
                    if not seg_sentences:
                        logger.debug(f"无匹配句子: {seg_path}")
                        continue
                    
                    seg_text = " ".join(s["text"] for s in seg_sentences)
                    data_item = {
                        "audio": {"path": seg_path},
                        "sentence": seg_text,
                        "language": args.language,
                        "sentences": seg_sentences,
                        "duration": round(seg_duration, 2)
                    }
                    
                    is_valid, msg = validate_data_format(data_item)
                    if is_valid:
                        dataset.append(data_item)
                        logger.debug(f"添加时长切分数据项: {seg_path}")
                    else:
                        logger.warning(f"数据格式验证失败: {seg_path} - {msg}")
                        
            elif args.split_method == 'both':
                sentence_groups = split_sentences_by_length(
                    sentences, args.max_sentence_length, args.length_method)
                
                for group in sentence_groups:
                    if not group:
                        continue
                        
                    group_start = group[0]['start']
                    group_end = group[-1]['end']
                    group_duration = group_end - group_start
                    
                    if group_duration > args.max_duration:
                        temp_audio = AudioSegment.from_wav(processed_path)[
                            int(group_start * 1000):int(group_end * 1000)]
                        temp_path = os.path.join(segment_dir, f"{base_name}_temp.wav")
                        temp_audio.export(temp_path, format="wav")
                        
                        time_segments = split_audio_by_duration(temp_path, segment_dir, args.max_duration)
                        os.remove(temp_path)
                        
                        for seg_path, rel_start, rel_end in time_segments:
                            abs_start = group_start + rel_start
                            abs_end = group_start + rel_end
                            
                            seg_sentences = []
                            for s in group:
                                if s["start"] < abs_end and s["end"] > abs_start:
                                    adjusted_start = max(0, s["start"] - abs_start)
                                    adjusted_end = min(rel_end - rel_start, s["end"] - abs_start)
                                    if adjusted_end > adjusted_start:
                                        seg_sentences.append({
                                            "start": round(adjusted_start, 2),
                                            "end": round(adjusted_end, 2),
                                            "text": s["text"],
                                            "speaker": s.get("speaker", None)
                                        })
                            
                            if seg_sentences:
                                seg_text = " ".join(s["text"] for s in seg_sentences)
                                data_item = {
                                    "audio": {"path": seg_path},
                                    "sentence": seg_text,
                                    "language": args.language,
                                    "sentences": seg_sentences,
                                    "duration": round(rel_end - rel_start, 2)
                                }
                                
                                is_valid, msg = validate_data_format(data_item)
                                if is_valid:
                                    dataset.append(data_item)
                                    logger.debug(f"添加混合切分数据项: {seg_path}")
                                else:
                                    logger.warning(f"数据格式验证失败: {seg_path} - {msg}")
                    else:
                        seg_path = os.path.join(segment_dir, f"{base_name}_sent_group_{len(dataset)}.wav")
                        group_audio = AudioSegment.from_wav(processed_path)[
                            int(group_start * 1000):int(group_end * 1000)]
                        group_audio.export(seg_path, format="wav")
                        
                        seg_sentences = []
                        for s in group:
                            adjusted_start = max(0, s["start"] - group_start)
                            adjusted_end = min(group_duration, s["end"] - group_start)
                            seg_sentences.append({
                                "start": round(adjusted_start, 2),
                                "end": round(adjusted_end, 2),
                                "text": s["text"],
                                "speaker": s.get("speaker", None)
                            })
                        
                        seg_text = " ".join(s["text"] for s in seg_sentences)
                        data_item = {
                            "audio": {"path": seg_path},
                            "sentence": seg_text,
                            "language": args.language,
                            "sentences": seg_sentences,
                            "duration": round(group_duration, 2)
                        }
                        
                        is_valid, msg = validate_data_format(data_item)
                        if is_valid:
                            dataset.append(data_item)
                            logger.debug(f"添加句子组数据项: {seg_path}")
                        else:
                            logger.warning(f"数据格式验证失败: {seg_path} - {msg}")

        if not dataset:
            raise ValueError("未生成有效数据")
        
        # 保存所有处理好的数据到一个文件
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            for item in dataset:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"已保存 {len(dataset)} 条数据到 {args.output_json}")
            
        logger.info(f"总数据条数: {len(dataset)}")
        if dataset:
            durations = [item['duration'] for item in dataset]
            lengths = [len(item['sentence']) for item in dataset]
            logger.info(f"音频时长统计: 最短 {min(durations):.2f}秒, 最长 {max(durations):.2f}秒, 平均 {sum(durations)/len(durations):.2f}秒")
            logger.info(f"文本长度统计: 最短 {min(lengths)}字符, 最长 {max(lengths)}字符, 平均 {sum(lengths)/len(lengths):.1f}字符")
            
            logger.info("数据示例:")
            print(json.dumps(dataset[0], ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"主程序异常: {str(e)}")
        raise

def preprocess_text(text: str) -> str:
    """
    对文本进行基本的预处理。
    """
    filtered_text = text.replace("like/subscribe to YouTube channel", "").replace("subtitles by [xxxx]", "")
    return " ".join(filtered_text.split()).strip()

def validate_data_format(item: dict) -> tuple[bool, str]:
    """
    验证数据项是否符合目标格式。
    """
    required_keys = ["audio", "sentence", "language", "sentences", "duration"]
    if not all(key in item for key in required_keys):
        return False, f"缺少必需的键: {required_keys}"
    
    if not isinstance(item["audio"], dict) or "path" not in item["audio"]:
        return False, "audio 键必须是一个包含 path 的字典"
    
    if not isinstance(item["sentences"], list) or not all(isinstance(s, dict) for s in item["sentences"]):
        return False, "sentences 键必须是一个字典列表"
        
    return True, "格式正确"

def process_huggingface_dataset(args: argparse.Namespace):
    """
    加载并处理 Hugging Face 数据集，将其转换为目标 JSONL 格式。
    该函数通过手动控制文件列表来智能跳过损坏的分片。
    """
    logger.info("开始处理 Hugging Face 数据集...")
    
    # 2. 创建输出目录和文件路径
    hf_output_dir = os.path.join(args.target_dir, "huggingface_processed")
    os.makedirs(hf_output_dir, exist_ok=True)
    
    audio_output_dir = os.path.join(hf_output_dir, "audio")
    os.makedirs(audio_output_dir, exist_ok=True)
    
    output_jsonl_path = os.path.join(hf_output_dir, "hf_data.jsonl")

    # 1. 检查已处理的记录，实现断点续传
    processed_ids = set()
    if os.path.exists(output_jsonl_path):
        logger.info("检测到已存在的 JSONL 文件，正在加载已处理的记录...")
        try:
            with open(output_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        audio_path = data['audio']['path']
                        audio_id = Path(audio_path).stem
                        processed_ids.add(audio_id)
                    except json.JSONDecodeError:
                        logger.warning(f"跳过 JSON 解码错误的行: {line.strip()}")
        except Exception as e:
            logger.error(f"加载已处理记录失败: {e}。将从头开始处理。")
            processed_ids.clear()
    
    logger.info(f"已加载 {len(processed_ids)} 条已处理的记录。")

    # 3. 获取数据集的所有分片文件列表
    logger.info("正在获取数据集的所有分片文件列表...")
    data_files = []
    try:
        # 使用 huggingface_hub 来获取所有分片文件
        repo_files = list_repo_files(args.hf_dataset_name, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        data_files = [f"hf://datasets/{args.hf_dataset_name}/train/{file_name}" for file_name in repo_files if file_name.startswith("train-") and file_name.endswith(".parquet")]
        logger.info(f"成功找到 {len(data_files)} 个分片文件。")
    except Exception as e:
        logger.error(f"无法获取数据集分片列表: {e}")
        return

    processed_count_this_run = 0
    with open(output_jsonl_path, "a", encoding="utf-8") as f:
        # 4. 遍历每个分片文件
        for file_path in data_files:
            try:
                # 尝试加载整个分片到内存
                logger.info(f"正在处理分片: {file_path}")
                
                # 创建一个 Parquet 文件读取器，它只包含 `audio` 和 `transcript_whisper` 列
                table = pq.read_table(
                    file_path, 
                    columns=['audio', 'transcript_whisper']
                )
                
                # 将 Parquet 表转换为可迭代的 Python 字典列表
                batch_data = table.to_pydict()
                
                # 遍历分片中的每一条记录
                for i in range(len(batch_data['audio'])):
                    item = {
                        'audio': batch_data['audio'][i],
                        'transcript_whisper': batch_data['transcript_whisper'][i],
                        # 确保其他可能需要的字段存在
                        'id': batch_data.get('id', [None]*len(batch_data['audio']))[i],
                    }

                    # 检查此项是否已处理过
                    item_id = item.get('id', 'unknown_id')
                    if item_id in processed_ids:
                        continue
                    
                    # === 核心过滤逻辑 ===
                    if not item['audio'] or not item['transcript_whisper']:
                        logger.warning(f"跳过缺少 'audio' 或 'transcript_whisper' 字段的数据项: {item_id}")
                        continue
                    
                    try:
                        # 文本预处理
                        transcript = item.get('transcript_whisper')
                        cleaned_text = preprocess_text(transcript)
                        if not cleaned_text.strip():
                            continue

                        # 音频重采样并保存到本地
                        audio_data = item.get('audio')
                        audio_array = audio_data['array']
                        sampling_rate = audio_data['sampling_rate']
                        
                        audio_filename = f"{item_id}.wav"
                        audio_path = os.path.join(audio_output_dir, audio_filename)
                        
                        if sampling_rate != args.expected_sr:
                            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=args.expected_sr)
                        
                        soundfile.write(audio_path, audio_array, args.expected_sr)
                        
                        # 格式化为 JSONL
                        duration = round(len(audio_array) / args.expected_sr, 2)
                        
                        processed_item = {
                            "audio": {"path": audio_path},
                            "sentence": cleaned_text,
                            "language": args.language,
                            "sentences": [
                                {
                                    "start": 0.0,
                                    "end": duration,
                                    "text": cleaned_text,
                                    "speaker": None
                                }
                            ],
                            "duration": duration
                        }

                        # 验证并写入文件
                        is_valid, msg = validate_data_format(processed_item)
                        if is_valid:
                            json.dump(processed_item, f, ensure_ascii=False)
                            f.write("\n")
                            processed_count_this_run += 1
                            if processed_count_this_run % 1000 == 0:
                                logger.info(f"本次运行已成功处理 {processed_count_this_run} 条新记录。")
                        else:
                            logger.warning(f"Hugging Face 数据项格式验证失败: {msg}")

                    except Exception as e:
                        # 这个块处理在处理单个数据项时发生的错误。
                        logger.error(f"处理 Hugging Face 数据项失败: {item.get('id', 'unknown')} - 错误: {e}")
                        
            except Exception as e:
                # 捕获在加载整个分片时发生的任何错误，并跳过该分片
                logger.error(f"无法加载分片 {file_path}。跳过该分片。错误: {e}")
                
    logger.info(f"本次运行成功处理 {processed_count_this_run} 条 Hugging Face 数据到 {output_jsonl_path}")
    logger.info(f"Hugging Face 数据处理完成，文件位于: {hf_output_dir}")

def main():

    parser = argparse.ArgumentParser(description="处理音频和文本数据生成JSON数据集")
    parser.add_argument("--input_path", type=str, default=None, help="输入的本地压缩包路径（可选）")
    parser.add_argument("--output_json", required=True, help="输出本地JSON文件路径")
    parser.add_argument("--target_dir", default="dataset", help="解压目录，默认是./dataset")
    parser.add_argument("--language", default="Cantonese", help="语言标签")
    parser.add_argument("--expected_sr", type=int, default=16000, help="目标采样率")
    parser.add_argument("--max_duration", type=float, default=30.0, help="最大音频时长（秒）")
    parser.add_argument("--min_duration", type=float, default=0.5, help="最小音频时长（秒）")
    parser.add_argument("--split_method", choices=['duration', 'sentence', 'both'], default='duration', help="切分方法")
    parser.add_argument("--max_sentence_length", type=int, default=1200, help="最大句子长度")
    parser.add_argument("--length_method", choices=['char', 'word', 'token'], default='char', help="长度计算方法")
    
    parser.add_argument("--hf_dataset_name", type=str, default="alvanlii/cantonese-youtube", help="Hugging Face 数据集名称，如果使用则指定")
    parser.add_argument("--hf_subset", type=str, default=None, help="Hugging Face 数据集的子集名称")

    args = parser.parse_args()

    if args.input_path:
        process_and_save_dataset(args)
    else:
        logger.info("未提供本地输入路径，跳过本地数据处理。")
    
    if args.hf_dataset_name:
        process_huggingface_dataset(args)
        
if __name__ == "__main__":
    main()