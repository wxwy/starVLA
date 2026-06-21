#!/usr/bin/env python3
"""
自动发现并更新所有活跃 train_starvla 进程的训练记录 markdown。
由 Claude Code 创建，每 30 分钟运行一次。
"""
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/disk/rl/starVLA")
DOCS_DIR = REPO_ROOT / "docs_zh/starflow_vla"
CHECKPOINTS_DIR = REPO_ROOT / "playground/Checkpoints"


def run(cmd, default=""):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def find_active_runs():
    """Discover active train_starvla processes and their run_ids."""
    runs = {}
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        try:
            with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except Exception:
            continue
        if "train_starvla" not in cmdline or "--run_id" not in cmdline:
            continue
        m = re.search(r"--run_id\s+(\S+)", cmdline)
        if not m:
            continue
        run_id = m.group(1)
        if run_id not in runs:
            runs[run_id] = []
        runs[run_id].append(pid_str)
    return runs


def get_run_id_base(run_id):
    """Extract the experiment base id, e.g. P0-M7-E-H2a-01 from full run_id."""
    parts = run_id.split("_")
    # Full run_id looks like P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048
    # Keep everything up to and including the experiment code segment.
    base_parts = []
    for p in parts:
        base_parts.append(p)
        if re.match(r"^[PE]\d+-M\d+-[A-Z]\d+[a-z]?-\d+[a-z]?$", p):
            break
    return "_".join(base_parts) if base_parts else run_id


def find_markdown(run_id):
    """Find the markdown tracker file for a run_id."""
    # Exact match first
    for candidate in DOCS_DIR.glob(f"*{run_id}*.md"):
        return candidate
    # Then by base experiment id
    base = get_run_id_base(run_id)
    for candidate in DOCS_DIR.glob(f"*{base}*.md"):
        return candidate
    return None


def get_latest_checkpoint(run_id):
    checkpoint_dir = CHECKPOINTS_DIR / run_id / "checkpoints"
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


def parse_tmux_progress(session_name="train"):
    """Parse the latest tqdm-style progress bar from a tmux session."""
    text = run(f"tmux capture-pane -pt {session_name} -S -2000 2>/dev/null", "")
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
    # Fallback looser match
    matches = re.findall(r"(\d+)/(\d+)\s*\[.*?([\d.]+)s/it", text)
    if matches:
        m = matches[-1]
        return {
            "step": int(m[0]),
            "total_steps": int(m[1]),
            "percent": round(int(m[0]) / int(m[1]) * 100, 2),
            "sec_per_it": float(m[2]),
            "data_time": None,
            "model_time": None,
        }
    return None


def get_gpu_stats():
    line = run(
        "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,"
        "power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits",
        "",
    )
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 7:
        return None
    return {
        "name": parts[0],
        "mem_total": parts[1],
        "mem_used": parts[2],
        "gpu_util": parts[3],
        "power_draw": parts[4],
        "power_limit": parts[5],
        "temp": parts[6],
    }


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


def get_process_runtime(pids):
    """Return the longest runtime among the given pids in seconds."""
    max_seconds = 0
    for pid in pids:
        etime = run(f"ps -p {pid} -o etime=", "").strip()
        if not etime:
            continue
        # etime formats: MM:SS, HH:MM:SS, DD-HH:MM:SS
        try:
            if "-" in etime:
                days, rest = etime.split("-")
                h, m, s = rest.split(":")
                seconds = int(days) * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
            else:
                parts = etime.split(":")
                if len(parts) == 2:
                    seconds = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else:
                    continue
            max_seconds = max(max_seconds, seconds)
        except Exception:
            continue
    return max_seconds


def update_markdown(md_path, run_id, pids, prog, latest_ckpt, gpu):
    content = md_path.read_text(encoding="utf-8")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    runtime_seconds = get_process_runtime(pids)
    runtime_str = format_runtime(runtime_seconds) if runtime_seconds else "未知"

    rows = [
        f"| 监控时间 | {now} |",
        f"| 训练状态 | 🟢 运行中 |",
        f"| run_id | `{run_id}` |",
        f"| 活跃进程数 | {len(pids)} |",
        f"| 已运行时间 | {runtime_str} |",
    ]

    if prog:
        rows.extend([
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

    # Remove empty strings
    rows = [r for r in rows if r]

    section_header = "## 自动监控状态"
    section_body = "\n".join(["| 字段 | 值 |", "| --- | --- |"] + rows)
    new_section = f"{section_header}\n\n{section_body}\n"

    if section_header in content:
        # Replace existing section (up to next ## or end of file)
        content = re.sub(
            rf"{re.escape(section_header)}\n.*?(?=\n## |\Z)",
            new_section.rstrip(),
            content,
            count=1,
            flags=re.DOTALL,
        )
    else:
        content = content.rstrip() + "\n\n" + new_section

    md_path.write_text(content, encoding="utf-8")
    print(f"Updated {md_path}")


def main():
    active_runs = find_active_runs()
    if not active_runs:
        print("No active train_starvla processes found.")
        return

    print(f"Active runs: {list(active_runs.keys())}")
    prog = parse_tmux_progress("train")
    gpu = get_gpu_stats()

    for run_id, pids in active_runs.items():
        md_path = find_markdown(run_id)
        if not md_path:
            print(f"No markdown tracker found for {run_id}, skipping.")
            continue
        latest_ckpt = get_latest_checkpoint(run_id)
        update_markdown(md_path, run_id, pids, prog, latest_ckpt, gpu)


if __name__ == "__main__":
    main()
