import subprocess
import signal
import sys

datasets = ["AQA", "SER", "VSC"]
processes = []

def cleanup(signal_received, frame):
    print("Killing current process...")
    if processes and processes[-1]:
        processes[-1].terminate()  # 终止当前正在运行的进程
    sys.exit(1)

# 监听 Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, cleanup)

# 按顺序运行所有 Python 脚本
print("开始按顺序执行 qwen2.py...")
for dataset in datasets:
    print(f"运行: python qwen2.py --dataset {dataset}")
    p = subprocess.Popen(["python", "qwen2.py", "--dataset", dataset])
    processes.append(p)
    p.wait()  # 等待当前进程完成后再继续

print("开始按顺序执行 qwen.py...")
for dataset in datasets:
    print(f"运行: python qwen.py --dataset {dataset}")
    p = subprocess.Popen(["python", "qwen.py", "--dataset", dataset])
    processes.append(p)
    p.wait()  # 等待当前进程完成后再继续

print("所有脚本已执行完毕")