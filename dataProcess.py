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

def main():
    """主函数：处理音频和文本，生成 JSON 数据集"""
    parser = argparse.ArgumentParser(description="处理音频和文本数据生成JSON数据集")
    parser.add_argument("--input_path", required=True, help="输入的压缩包路径（支持 .tar.gz 或 .zip）")
    parser.add_argument("--output_json", required=True, help="输出JSON文件路径")
    parser.add_argument("--target_dir", default="dataset", help="解压目录，默认是./dataset")
    parser.add_argument("--language", default="Cantonese", help="语言标签（如 zh, en）") #Cantonese
    parser.add_argument("--expected_sr", type=int, default=16000, help="目标采样率")
    parser.add_argument("--max_duration", type=float, default=30.0, help="最大音频时长（秒）")
    parser.add_argument("--min_duration", type=float, default=0.5, help="最小音频时长（秒）")
    parser.add_argument("--split_method", choices=['duration', 'sentence', 'both'], default='duration', help="切分方法：duration按时长，sentence按句子，both两种都用")
    parser.add_argument("--max_sentence_length", type=int, default=200, help="最大句子长度")
    parser.add_argument("--length_method", choices=['char', 'word', 'token'], default='char', help="长度计算方法：char字符，word词，token令牌")
    args = parser.parse_args()

    process_and_save_dataset(args)

if __name__ == "__main__":
    main()