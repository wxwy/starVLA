#!/usr/bin/env python3
"""
监控非 tmux 环境下运行的训练进程，更新训练记录 markdown。
通过检测进程存活 + 最新 checkpoint + 系统状态来追踪训练进度。
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


def find_active_train_runs():
    """Find all running train_starvla processes and extract their --run_id."""
    active = {}
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        try:
            with open(f"/proc/{pid_str}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except Exception:
            continue
        if "train_starvla" not in cmdline:
            continue
        # Check if it's a parent process (not a worker spawned by accelerate)
        try:
            with open(f"/proc/{pid_str}/status", "rb") as f:
                status = f.read().decode("utf-8", errors="ignore")
            ppid_match = re.search(r"PPid:\s+(\d+)", status)
            ppid = ppid_match.group(1) if ppid_match else ""
            # Only count the top-level accelerate process (parent is usually 1 or shell)
            # The key check: this process name should be "python" and have accelerate args
        except Exception:
            continue

        m = re.search(r"--run_id\s+(\S+)", cmdline)
        if not m:
            continue
        run_id = m.group(1)

        # Skip duplicates (multiple workers with same run_id)
        if run_id in active:
            # Keep the one with lower PID (parent)
            continue
        active[run_id] = {
            "pid": pid_str,
            "cmdline": cmdline,
        }
    return active


def is_process_alive(pid):
    """Check if a process is still running."""
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def find_markdown(run_id):
    """Find the markdown tracker file for a run_id."""
    # Direct match first
    for candidate in DOCS_DIR.glob(f"*{run_id}*.md"):
        return candidate
    # Try base experiment id match
    parts = run_id.split("_")
    base_parts = []
    for p in parts:
        base_parts.append(p)
        if re.match(r"^[PE]\d+-M\d+-[A-Z]\d+[a-z]?-\d+[a-z]?$", p):
            break
    base = "_".join(base_parts) if base_parts else run_id
    for candidate in DOCS_DIR.glob(f"*{base}*.md"):
        return candidate
    return None


def get_latest_checkpoint(run_id):
    """Get the latest checkpoint (step, mtime) for a run."""
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
            except (ValueError, IndexError):
                pass
    if not cks:
        return None
    cks.sort(key=lambda x: x[1], reverse=True)
    return cks[0]


def get_gpu_stats():
    """Get GPU stats from nvidia-smi."""
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


def get_cpu_mem_stats():
    """Get CPU memory stats."""
    text = run("free -h 2>/dev/null | head -2", "")
    if not text:
        return None
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None
    parts = lines[1].split()
    if len(parts) < 4:
        return None
    return {"total": parts[1], "used": parts[2], "free": parts[3]}


def get_disk_stats():
    """Get disk usage for key mount points."""
    text = run("df -h /disk/rl /localdisk-tmp 2>/dev/null", "")
    if not text:
        return None
    lines = text.strip().splitlines()
    result = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        mount = parts[5]
        result[mount] = {
            "size": parts[1],
            "used": parts[2],
            "avail": parts[3],
            "use_pct": parts[4],
        }
    return result


def get_process_start_time(pid):
    """Get the start time of a process from /proc/PID/stat."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read().decode("utf-8", errors="ignore")
        # Field 22 is starttime (in clock ticks since boot)
        fields = stat.split()
        if len(fields) >= 22:
            # starttime is field 21 (0-indexed)
            return int(fields[21])
    except Exception:
        pass
    return None


def get_system_uptime_seconds():
    """Get system uptime in seconds."""
    try:
        with open("/proc/uptime", "rb") as f:
            uptime = f.read().decode("utf-8", errors="ignore")
        return float(uptime.split()[0])
    except Exception:
        return None


def estimate_current_step(run_id, latest_ckpt):
    """Estimate the current step based on checkpoint timing and save_interval."""
    if not latest_ckpt:
        return None
    step, mtime = latest_ckpt
    now = datetime.now().timestamp()
    elapsed_since_ckpt = now - mtime
    # Based on config, save_interval is 125 steps, ~6s/it
    # Each save cycle: 125 * 6 = 750s = 12.5 min
    sec_per_it = 6.0  # Conservative estimate
    steps_since = int(elapsed_since_ckpt / sec_per_it)
    estimated = step + steps_since
    return estimated, elapsed_since_ckpt


def parse_mse_from_md(md_path):
    """Extract the latest MSE values from the markdown tracker."""
    if not md_path or not md_path.exists():
        return []
    content = md_path.read_text(encoding="utf-8")
    # Match MSE eval lines: | ~N | STEP | MSE | ... |
    # Also match: | ~104 | **26000** | 0.00914 | ↑ 波动 |
    entries = []
    # Pattern for eval table rows with step and MSE
    pattern = r"\|\s*~?\d+\s*\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*\*{0,2}([\d.]+)\*{0,2}\s*\|"
    for m in re.finditer(pattern, content):
        step = int(m.group(1))
        mse = float(m.group(2))
        entries.append((step, mse))
    return entries


def update_markdown(run_id, md_path, info):
    """Update the '自动监控状态' section in the markdown file."""
    content = md_path.read_text(encoding="utf-8")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")

    process_alive = info.get("process_alive", False)
    status_icon = "🟢 运行中" if process_alive else "🔴 已停止"

    rows = [
        f"| 监控时间 | {now} |",
        f"| 训练状态 | {status_icon} |",
        f"| run_id | `{run_id}` |",
    ]

    ckpt = info.get("latest_checkpoint")
    if ckpt:
        rows.append(f"| 最新完整 checkpoint | `steps_{ckpt[0]}` |")

    est = info.get("estimated_step")
    if est:
        step_est, elapsed = est
        rows.extend([
            f"| 预估当前步数 | **~{step_est}** / 80000 |",
            f"| 预估完成比例 | ~{step_est/80000*100:.1f}% |",
            f"| 距上次 checkpoint | ~{int(elapsed//60)} 分钟 |",
        ])

    # Speed estimate
    sec_per_it = info.get("sec_per_it", 6.0)
    rows.append(f"| 预估训练速度 | ~{sec_per_it:.1f} s/it |")

    # Runtime estimate
    if est:
        step_est, _ = est
        estimated_total_sec = step_est * sec_per_it
        remaining_steps = 80000 - step_est
        estimated_remaining_sec = remaining_steps * sec_per_it

        rows.append(f"| 预估已运行 | ~{estimated_total_sec/3600:.1f} 小时 |")
        rows.append(f"| 预估剩余 | ~{estimated_remaining_sec/3600:.1f} 小时 ({estimated_remaining_sec/86400:.1f} 天) |")

        # Cost estimate
        cost_per_hour = 5.58
        elapsed_cost = estimated_total_sec / 3600 * cost_per_hour
        remaining_cost = estimated_remaining_sec / 3600 * cost_per_hour
        total_cost = 80000 * sec_per_it / 3600 * cost_per_hour

        rows.extend([
            f"| 已产生成本 | ~¥{elapsed_cost:.2f} |",
            f"| 预估剩余成本 | ~¥{remaining_cost:.2f} |",
            f"| 完整训练预估成本 | ~¥{total_cost:.2f} |",
        ])

    gpu = info.get("gpu")
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

    cpu_mem = info.get("cpu_mem")
    if cpu_mem:
        rows.extend([
            f"| 系统内存总量 | {cpu_mem['total']} |",
            f"| 系统内存已用 | {cpu_mem['used']} |",
            f"| 系统内存空闲 | {cpu_mem['free']} |",
        ])

    disk = info.get("disk")
    if disk:
        for mount in ["/disk/rl", "/localdisk-tmp"]:
            dinfo = disk.get(mount)
            if dinfo:
                rows.append(
                    f"| 存储 `{mount}` | {dinfo['used']} / {dinfo['size']} ({dinfo['use_pct']} 已用) |"
                )

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

    md_path.write_text(content, encoding="utf-8")
    return True


def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting no-tmux monitor...")

    # Find active training runs
    active_runs = find_active_train_runs()
    if not active_runs:
        print("No active train_starvla processes found.")
        return

    print(f"Found {len(active_runs)} active run(s): {list(active_runs.keys())}")

    # Collect system stats (shared across runs)
    gpu = get_gpu_stats()
    cpu_mem = get_cpu_mem_stats()
    disk = get_disk_stats()

    updated_any = False

    for run_id, proc_info in active_runs.items():
        print(f"\n--- Processing: {run_id} ---")

        alive = is_process_alive(proc_info["pid"])
        print(f"  Process alive: {alive} (PID {proc_info['pid']})")

        md_path = find_markdown(run_id)
        if not md_path:
            print(f"  No markdown tracker found for {run_id}, skipping.")
            continue
        print(f"  Tracker: {md_path.name}")

        latest_ckpt = get_latest_checkpoint(run_id)
        if latest_ckpt:
            print(f"  Latest checkpoint: steps_{latest_ckpt[0]}")
        else:
            print(f"  No checkpoints found")

        estimated_step = estimate_current_step(run_id, latest_ckpt)
        if estimated_step:
            est_step, elapsed = estimated_step
            print(f"  Estimated current step: ~{est_step} ({est_step/800:.1f}%)")

        info = {
            "process_alive": alive,
            "latest_checkpoint": latest_ckpt,
            "estimated_step": estimated_step,
            "gpu": gpu,
            "cpu_mem": cpu_mem,
            "disk": disk,
            "sec_per_it": 6.0,
        }

        if update_markdown(run_id, md_path, info):
            print(f"  Updated {md_path.name}")
            updated_any = True

    return updated_any


if __name__ == "__main__":
    main()
