#!/bin/python3
import subprocess
import sys
import os

def main():
    # 获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts_to_run = [
        os.path.join(script_dir, "common.sh"),
        os.path.join(script_dir, "1.sh"),
        os.path.join(script_dir, "2.sh"),
        os.path.join(script_dir, "3.sh")
    ]
    
    processes = []
    
    print("开始同时运行录制脚本...")
    
    # 启动所有进程
    for script in scripts_to_run:
        print(f">>> 启动: {script}")
        # 使用 bash 运行脚本
        # preexec_fn=os.setsid 确保可以向整个进程组发送信号，方便关闭
        p = subprocess.Popen(["bash", script], preexec_fn=os.setsid)
        processes.append(p)

    try:
        # 等待所有进程结束
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n检测到退出信号 (Ctrl+C)，正在终止所有录制进程...")
        for p in processes:
            try:
                # 终止整个进程组，确保相关的 rosbag 进程也被结束
                os.killpg(os.getpgid(p.pid), 15) # 15 是 SIGTERM
            except Exception as e:
                print(f"无法终止进程组 {p.pid}: {e}")
        
        print("所有录制进程已终止。")
        sys.exit(0)

if __name__ == "__main__":
    main()
