#!/usr/bin/env python3
"""
自动更新训练跟踪 markdown 文档。
由 Claude Code 创建，用于持续跟踪 tmux train 会话。
"""
import os
import re
import subprocess
import json
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652"
DOCS_DIR = Path("/disk/rl/starVLA/docs_zh/starflow_vla")
MD_PATH = DOCS_DIR / f"{RUN_ID}.md"
COST_PER_HOUR = 5.58  # 元/小时


def run(cmd, default=""):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def parse_tmux_progress():
    text = run("tmux capture-pane -pt train -S -1000 2>/dev/null", "")
    # Match lines like: 0%| | 15/80000 [02:30<210:37:12, 9.48s/it, data_times=0.001, model_times=2.315]
    # Also handles elapsed > 1h: [1:02:51<208:42:27, ...]
    pattern = r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)s/it,\s*data_times=([\d.]+),\s*model_times=([\d.]+)\]"
    matches = re.findall(pattern, text)
    if matches:
        m = matches[-1]  # Take the latest progress bar
        return {
            "percent": float(m[0]),
            "step": int(m[1]),
            "total_steps": int(m[2]),
            "elapsed": m[3],
            "eta": m[4],
            "sec_per_it": float(m[5]),
            "data_time": float(m[6]),
            "model_time": float(m[7]),
        }

    # Fallback: looser match, also take last
    matches = re.findall(r"(\d+)/(\d+)\s*\[.*?([\d.]+)s/it", text)
    if matches:
        m = matches[-1]
        return {
            "step": int(m[0]),
            "total_steps": int(m[1]),
            "percent": round(int(m[0]) / int(m[1]) * 100, 2),
            "elapsed": "",
            "eta": "",
            "sec_per_it": float(m[2]),
            "data_time": None,
            "model_time": None,
        }
    return None


def get_gpu_stats():
    line = run(
        "nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits",
        "",
    )
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    return {
        "timestamp": parts[0] if len(parts) > 0 else "",
        "name": parts[1] if len(parts) > 1 else "",
        "mem_total": parts[2] if len(parts) > 2 else "",
        "mem_used": parts[3] if len(parts) > 3 else "",
        "mem_free": parts[4] if len(parts) > 4 else "",
        "gpu_util": parts[5] if len(parts) > 5 else "",
        "mem_util": parts[6] if len(parts) > 6 else "",
        "temp": parts[7] if len(parts) > 7 else "",
        "power_draw": parts[8] if len(parts) > 8 else "",
        "power_limit": parts[9] if len(parts) > 9 else "",
    }


def get_system_stats():
    mem = {"total": "", "used": "", "free": "", "available": ""}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem["total"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                elif line.startswith("MemFree:"):
                    mem["free"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                elif line.startswith("MemAvailable:"):
                    mem["available"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                elif line.startswith("Buffers:"):
                    mem["buffers"] = int(line.split()[1])
                elif line.startswith("Cached:"):
                    mem["cached"] = int(line.split()[1])
        mem["used"] = round(mem["total"] - mem.get("available", 0), 1)
    except Exception:
        pass

    cpu = run("top -bn1 | grep 'Cpu(s)' | head -1", "")
    return mem, cpu


def get_process_elapsed_hours():
    # Find the train_starvla python process by checking /proc/*/cmdline
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            try:
                with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                    cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            except Exception:
                continue
            if "train_starvla" in cmdline and "python" in cmdline:
                etime = run(f"ps -p {pid_str} -o etime=", "").strip()
                return etime_to_hours(etime)
    except Exception:
        pass

    # Fallback: find the main python process in starVLA env with long etime
    text = run("ps -eo pid,cmd,etime --sort=-pcpu", "")
    for line in text.splitlines():
        if "starVLA/bin/python" in line and "train" in line.lower():
            parts = line.strip().split()
            etime = parts[-1]
            return etime_to_hours(etime)
    return 0.0


def etime_to_hours(etime):
    # etime formats: MM:SS, HH:MM:SS, DD-HH:MM:SS
    try:
        if "-" in etime:
            days, rest = etime.split("-")
            h, m, s = rest.split(":")
            return int(days) * 24 + int(h) + int(m) / 60 + int(s) / 3600
        parts = etime.split(":")
        if len(parts) == 2:
            m, s = parts
            return int(m) / 60 + int(s) / 3600
        elif len(parts) == 3:
            h, m, s = parts
            return int(h) + int(m) / 60 + int(s) / 3600
    except Exception:
        pass
    return 0.0


def format_hours(h):
    days = int(h // 24)
    rem = h % 24
    hours = int(rem)
    minutes = int((rem - hours) * 60)
    if days > 0:
        return f"{days}天 {hours}小时 {minutes}分钟"
    return f"{hours}小时 {minutes}分钟"


def update_section(content, section_title, new_body):
    # Section headers are like "## 3. 训练进度"
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # Try matching section that already has a "最后更新" line
    pattern = rf"(## \d+\.\s*{re.escape(section_title)}\n\n)> 最后更新：.*?(\n\n)(.*?)(?=\n---|\n## \d+\. )"
    replacement = rf"\1> 最后更新：{now_str}\2{new_body}"
    new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count:
        return new_content

    # Fallback: section without "最后更新" line
    pattern2 = rf"(## \d+\.\s*{re.escape(section_title)}\n\n)(.*?)(?=\n---|\n## \d+\. )"
    replacement2 = rf"\1> 最后更新：{now_str}\n\n{new_body}"
    new_content, count = re.subn(pattern2, replacement2, content, count=1, flags=re.DOTALL)
    if count == 0:
        print(f"Warning: section '{section_title}' not found or not updated.")
    return new_content


def main():
    if not MD_PATH.exists():
        print(f"Markdown file not found: {MD_PATH}")
        return

    content = MD_PATH.read_text(encoding="utf-8")

    # 1. Training progress
    prog = parse_tmux_progress()
    if prog:
        total_steps = prog["total_steps"]
        step = prog["step"]
        sec_per_it = prog["sec_per_it"]
        remaining_steps = total_steps - step
        eta_sec = remaining_steps * sec_per_it
        eta_hours = eta_sec / 3600
        elapsed_hours = get_process_elapsed_hours()
        total_hours = elapsed_hours + eta_hours

        progress_body = f"""| 指标 | 值 |
|------|-----|
| **当前 Step** | {step} / {total_steps} |
| **完成比例** | {prog['percent']}% |
| **单步耗时** | ~{sec_per_it:.2f} s/it |
| **数据加载耗时** | {prog.get('data_time', 'N/A')} s |
| **模型前向/反向耗时** | {prog.get('model_time', 'N/A')} s |
| **已运行时间** | {format_hours(elapsed_hours)} |
| **预计剩余时间** | {format_hours(eta_hours)} |
| **预计总耗时** | {format_hours(total_hours)} |
"""
        content = update_section(content, "训练进度", progress_body)

    # 2. System resources
    gpu = get_gpu_stats()
    mem, cpu = get_system_stats()
    resource_body = ""
    if gpu:
        resource_body += f"""### 4.1 GPU（{gpu['name']}）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | {gpu['gpu_util']}% |
| **显存使用** | {gpu['mem_used']} MiB / {gpu['mem_total']} MiB ({round(int(gpu['mem_used'])/int(gpu['mem_total'])*100, 1)}%) |
| **显存空闲** | {gpu['mem_free']} MiB |
| **功耗** | {gpu['power_draw']} W / {gpu['power_limit']} W |
| **温度** | {gpu['temp']}°C |

"""
    resource_body += f"""### 4.2 CPU / 内存

| 指标 | 值 |
|------|-----|
| **CPU 使用率** | {cpu.strip() if cpu else 'N/A'} |
| **内存总量** | {mem.get('total', 'N/A')} GiB |
| **内存已用** | {mem.get('used', 'N/A')} GiB |
| **内存可用** | {mem.get('available', 'N/A')} GiB |
"""
    content = update_section(content, "系统资源占用", resource_body)

    # 3. Cost estimate
    elapsed_hours = get_process_elapsed_hours()
    current_cost = elapsed_hours * COST_PER_HOUR
    if prog:
        total_hours_est = (prog["total_steps"] * prog["sec_per_it"]) / 3600
        total_cost_est = total_hours_est * COST_PER_HOUR
        remaining_cost_est = total_cost_est - current_cost
        cost_body = f"""| 项目 | 计算 |
|------|------|
| **已产生成本** | {format_hours(elapsed_hours)} × {COST_PER_HOUR} 元/h ≈ **{current_cost:.2f} 元** |
| **预计总成本** | {format_hours(total_hours_est)} × {COST_PER_HOUR} 元/h ≈ **{total_cost_est:.2f} 元** |
| **剩余预计成本** | {format_hours(total_hours_est - elapsed_hours)} × {COST_PER_HOUR} 元/h ≈ **{remaining_cost_est:.2f} 元** |
"""
    else:
        cost_body = f"""| 项目 | 计算 |
|------|------|
| **已产生成本** | {format_hours(elapsed_hours)} × {COST_PER_HOUR} 元/h ≈ **{current_cost:.2f} 元** |
| **预计总成本** | 等待训练进度信息... |
| **剩余预计成本** | 等待训练进度信息... |
"""
    content = update_section(content, "成本估算", cost_body)

    MD_PATH.write_text(content, encoding="utf-8")
    print(f"Updated {MD_PATH}")


if __name__ == "__main__":
    main()
