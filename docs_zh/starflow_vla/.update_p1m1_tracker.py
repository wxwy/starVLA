#!/usr/bin/env python3
"""
自动更新 P1-M1-E-H2b-02 训练跟踪 markdown 文档。
每 30 分钟运行一次，更新训练进度、系统资源、成本估算和自动监控状态。
"""
import os
import re
import subprocess
import json
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = "P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901"
TMUX_SESSION = "train-2"
DOCS_DIR = Path("/disk/rl/starVLA/docs_zh/starflow_vla")
MD_PATH = DOCS_DIR / f"{RUN_ID}.md"
CHECKPOINTS_DIR = Path("/disk/rl/starVLA/playground/Checkpoints")
COST_PER_HOUR = 5.58  # 元/小时


def run(cmd, default=""):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def parse_tmux_progress():
    text = run(f"tmux capture-pane -pt {TMUX_SESSION} -S -1000 2>/dev/null", "")
    pattern = (
        r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)s/it,\s*"
        r"data_times=([\d.]+),\s*model_times=([\d.]+)\]"
    )
    matches = re.findall(pattern, text)
    if matches:
        m = matches[-1]
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
        "nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu,"
        "utilization.memory,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits",
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
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            try:
                with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                    cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            except Exception:
                continue
            if RUN_ID in cmdline and "train_starvla" in cmdline:
                etime = run(f"ps -p {pid_str} -o etime=", "").strip()
                return etime_to_hours(etime)
    except Exception:
        pass
    return 0.0


def etime_to_hours(etime):
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


def format_runtime(seconds):
    if seconds < 60:
        return f"{int(seconds)}秒"
    if seconds < 3600:
        return f"{int(seconds // 60)}分钟 {int(seconds % 60)}秒"
    if seconds < 86400:
        return f"{int(seconds // 3600)}小时 {int((seconds % 3600) // 60)}分钟"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{days}天 {hours}小时 {minutes}分钟"


def update_section(content, section_title, new_body):
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # Match "## 3. Title" or "## Title"
    pattern = rf"(## (?:\d+\.\s*)?{re.escape(section_title)}\n\n)> 最后更新：.*?(\n\n)(.*?)(?=\n---|\n## (?:\d+\. )?|\Z)"
    replacement = rf"\1> 最后更新：{now_str}\2{new_body}"
    new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count:
        return new_content

    pattern2 = rf"(## (?:\d+\.\s*)?{re.escape(section_title)}\n\n)(.*?)(?=\n---|\n## (?:\d+\. )?|\Z)"
    replacement2 = rf"\1> 最后更新：{now_str}\n\n{new_body}"
    new_content, count = re.subn(pattern2, replacement2, content, count=1, flags=re.DOTALL)
    if count == 0:
        print(f"Warning: section '{section_title}' not found or not updated.")
    return new_content


def get_latest_checkpoint():
    checkpoint_dir = CHECKPOINTS_DIR / RUN_ID / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    cks = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("steps_"):
            try:
                step = int(d.name.split("_")[1])
                mtime = d.stat().st_mtime
                cks.append((step, mtime))
            except ValueError:
                pass
    if not cks:
        return None
    cks.sort(key=lambda x: x[1], reverse=True)
    return cks[0]


def update_auto_monitoring(content, prog, gpu, elapsed_hours, latest_ckpt):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    runtime_seconds = int(elapsed_hours * 3600)
    runtime_str = format_runtime(runtime_seconds) if runtime_seconds > 0 else "未知"

    rows = [
        f"| 监控时间 | {now} |",
        f"| 训练状态 | {'🟢 运行中' if prog else '🟡 未检测到活跃进度条'} |",
        f"| run_id | `{RUN_ID}` |",
        f"| tmux 会话 | `{TMUX_SESSION}` |",
    ]

    if prog:
        rows.extend([
            f"| 已运行时间 | {runtime_str} |",
            f"| 当前步数 | **{prog['step']} / {prog['total_steps']}** |",
            f"| 完成比例 | {prog['percent']}% |",
            f"| 训练速度 | ~{prog['sec_per_it']:.2f} s/it |",
            f"| data_time | {prog['data_time']} s |" if prog.get("data_time") is not None else "",
            f"| model_time | {prog['model_time']} s |" if prog.get("model_time") is not None else "",
        ])
    elif latest_ckpt:
        rows.append(f"| 当前步数 | {latest_ckpt[0]} / 80000（来自最新 checkpoint） |")

    if latest_ckpt:
        rows.append(f"| 最新完整 checkpoint | `steps_{latest_ckpt[0]}` |")
    else:
        rows.append("| 最新完整 checkpoint | 暂无 |")

    if gpu:
        try:
            mem_pct = round(int(gpu["mem_used"]) / int(gpu["mem_total"]) * 100, 1)
        except Exception:
            mem_pct = "N/A"
        rows.extend([
            f"| GPU | {gpu['name']} |",
            f"| GPU 利用率 | {gpu['gpu_util']}% |",
            f"| 显存使用 | {gpu['mem_used']} MiB / {gpu['mem_total']} MiB ({mem_pct}%) |",
            f"| 功耗 | {gpu['power_draw']} W / {gpu['power_limit']} W |",
            f"| 温度 | {gpu['temp']}°C |",
        ])

    rows = [r for r in rows if r]

    section_header = "## 自动监控状态"
    section_body = "\n".join(["| 字段 | 值 |", "| --- | --- |"] + rows)
    new_section = f"{section_header}\n\n{section_body}\n"

    if section_header in content:
        content = re.sub(
            rf"{re.escape(section_header)}\n.*?(?=\n## |\Z)",
            new_section.rstrip(),
            content,
            count=1,
            flags=re.DOTALL,
        )
    else:
        content = content.rstrip() + "\n\n" + new_section
    return content


def main():
    if not MD_PATH.exists():
        print(f"Markdown file not found: {MD_PATH}")
        return

    content = MD_PATH.read_text(encoding="utf-8")

    # 1. Training progress
    prog = parse_tmux_progress()
    elapsed_hours = get_process_elapsed_hours()
    latest_ckpt = get_latest_checkpoint()

    if prog:
        total_steps = prog["total_steps"]
        step = prog["step"]
        sec_per_it = prog["sec_per_it"]
        remaining_steps = total_steps - step
        eta_sec = remaining_steps * sec_per_it
        eta_hours = eta_sec / 3600
        total_hours_est = elapsed_hours + eta_hours

        progress_body = f"""| 指标 | 值 |
|------|-----|
| **当前 Step** | {step} / {total_steps} |
| **完成比例** | {prog['percent']}% |
| **单步耗时** | ~{sec_per_it:.2f} s/it |
| **数据加载耗时** | {prog.get('data_time', 'N/A')} s |
| **模型前向/反向耗时** | {prog.get('model_time', 'N/A')} s |
| **已运行时间** | {format_hours(elapsed_hours)} |
| **预计剩余时间** | {format_hours(eta_hours)} |
| **预计总耗时** | {format_hours(total_hours_est)} |
"""
        content = update_section(content, "训练进度", progress_body)
    else:
        print("Warning: could not parse tmux progress.")

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

    # 4. Auto-monitoring section
    content = update_auto_monitoring(content, prog, gpu, elapsed_hours, latest_ckpt)

    MD_PATH.write_text(content, encoding="utf-8")
    print(f"Updated {MD_PATH}")


if __name__ == "__main__":
    main()
