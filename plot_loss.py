import os
import json
import matplotlib.pyplot as plt

def plot_full_training_curve(output_dir):
    # 加载父目录的全局日志
    global_log_path = os.path.join(output_dir, "trainer_state.json")
    
    # 检查文件是否存在
    if not os.path.exists(global_log_path):
        raise FileNotFoundError(f"无法找到日志文件: {global_log_path}")
    
    with open(global_log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        logs = data.get("log_history", [])
    
    if not logs:
        print("警告：日志中没有数据！")
        return
    
    # 提取数据
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    
    for entry in logs:
        if "loss" in entry:
            train_steps.append(entry.get("step", len(train_steps)+1))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)+1))
            eval_loss.append(entry["eval_loss"])
    
    # 打印调试信息
    print(f"找到 {len(train_loss)} 个训练损失点")
    print(f"找到 {len(eval_loss)} 个评估损失点")
    
    # 绘图
    plt.figure(figsize=(12, 6))
    
    if train_loss:
        plt.plot(train_steps, train_loss, label="Training Loss", color="blue")
    
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Evaluation Loss", 
                linestyle="--", color="orange")
    
    if not train_loss and not eval_loss:
        print("错误：没有可绘制的数据点！")
        return
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(output_dir, "full_loss_curve.png")
    plt.savefig(output_path)
    print(f"图表已保存到: {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_full_training_curve("output/whisper-large-v3")