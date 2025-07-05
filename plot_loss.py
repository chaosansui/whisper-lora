import json
import matplotlib.pyplot as plt

def plot_loss(output_dir):
    # 读取日志文件
    with open(f"{output_dir}/whisper-large-v3/checkpoint-605/trainer_state.json", "r") as f:
        logs = json.load(f)
    
    # 提取损失数据
    train_loss = [log["loss"] for log in logs["log_history"] if "loss" in log]
    eval_loss = [log["eval_loss"] for log in logs["log_history"] if "eval_loss" in log]
    steps = [log["step"] for log in logs["log_history"] if "loss" in log]

    # 绘制曲线
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss, label="Training Loss")
    if eval_loss:
        eval_steps = steps[:len(eval_loss)]  # 假设评估步数与训练步数对齐
        plt.plot(eval_steps, eval_loss, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/loss_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_loss("output/")  # 替换为实际输出目录