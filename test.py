import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import argparse
import time
import numpy as np
from pydub import AudioSegment  # 需要安装 pydub

def parse_args():
    parser = argparse.ArgumentParser(description="测试语音识别模型（支持长音频）")
    parser.add_argument("--model_path", type=str, required=True, help="合并后的模型路径")
    parser.add_argument("--audio_path", type=str, required=True, help="要测试的音频文件路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="运行设备 (cuda/cpu)")
    parser.add_argument("--language", type=str, default="zh", help="语音语言代码 (如zh/en)")
    parser.add_argument("--chunk_length", type=int, default=30, help="分段长度（秒）")
    parser.add_argument("--overlap", type=int, default=2, help="分段重叠长度（秒）")
    return parser.parse_args()

def load_model(model_path, device):
    print(f"正在加载模型从 {model_path}...")
    start_time = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"模型加载完成，耗时 {time.time()-start_time:.2f}秒")
    return model, processor

def load_long_audio(audio_path, target_sr=16000):
    print(f"正在加载长音频文件 {audio_path}...")
    
    # 使用pydub处理各种格式的音频
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1)
    duration_sec = len(audio) / 1000
    print(f"音频总时长: {duration_sec:.2f}秒")
    
    # 转换为numpy数组
    samples = np.array(audio.get_array_of_samples())
    return samples, target_sr, duration_sec

def split_audio(audio_array, sample_rate, chunk_length=30, overlap=2):
    """将长音频分割成带重叠的片段"""
    chunk_size = chunk_length * sample_rate
    overlap_size = overlap * sample_rate
    stride = chunk_size - overlap_size
    
    chunks = []
    for start in range(0, len(audio_array), stride):
        end = min(start + chunk_size, len(audio_array))
        chunk = audio_array[start:end]
        chunks.append(chunk)
        if end == len(audio_array):
            break
    
    print(f"分割为 {len(chunks)} 个片段，每段 {chunk_length}秒 (重叠 {overlap}秒)")
    return chunks

def transcribe_long_audio(model, processor, audio_chunks, sample_rate, device, language="zh"):
    full_transcript = []
    
    for i, chunk in enumerate(audio_chunks, 1):
        print(f"\n处理片段 {i}/{len(audio_chunks)}...")
        start_time = time.time()
        
        inputs = processor(
            chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, language=language)
        
        transcript = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        full_transcript.append(transcript)
        
        print(f"片段 {i} 识别完成，耗时 {time.time()-start_time:.2f}秒")
        print(f"当前结果: {transcript}")
    
    return " ".join(full_transcript)

if __name__ == "__main__":
    args = parse_args()
    
    # 加载模型
    model, processor = load_model(args.model_path, args.device)
    
    # 加载音频（支持长音频）
    audio_array, sample_rate, duration = load_long_audio(args.audio_path)
    
    # 分割音频
    audio_chunks = split_audio(
        audio_array, 
        sample_rate, 
        chunk_length=args.chunk_length,
        overlap=args.overlap
    )
    
    # 分段识别
    result = transcribe_long_audio(
        model=model,
        processor=processor,
        audio_chunks=audio_chunks,
        sample_rate=sample_rate,
        device=args.device,
        language=args.language
    )
    
    # 打印结果
    print("\n" + "="*50)
    print(f"音频文件: {args.audio_path}")
    print(f"总时长: {duration:.2f}秒")
    print(f"最终识别结果:\n{result}")
    print("="*50)