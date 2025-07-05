import json
import os

for file_path in ["dataset/train.json", "dataset/test.json"]:
    print(f"检查文件: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                print(f"第 {i} 行为空")
                continue
            try:
                data = json.loads(line)
                audio_path = data["audio"]["path"]
                duration = data["duration"]
                sentence = data.get("sentence", "")
                sentences = data.get("sentences", [])
                language = data.get("language", "")
                sentence_len = len(sentence) if sentence else sum(len(s["text"]) for s in sentences)
                print(f"第 {i} 行，音频: {audio_path}, 时长: {duration}, 语言: {language}, 句子长度: {sentence_len}")
                if not os.path.exists(audio_path):
                    print(f"  - 音频文件不存在: {audio_path}")
                if duration < 0.5 or duration > 3600:
                    print(f"  - 时长 {duration} 超出范围 [0.5, 3600]")
                if sentence_len < 1 or sentence_len > 100000:
                    print(f"  - 句子长度 {sentence_len} 超出范围 [1, 100000]")
            except json.JSONDecodeError as e:
                print(f"第 {i} 行，JSON 解析失败: {e}")