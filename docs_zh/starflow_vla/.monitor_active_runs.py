#!/usr/bin/env python3
"""
自动从 tmux `train` 会话发现并更新训练记录 markdown。
只监控 tmux session `train` 中的实验，不管其他活跃进程。
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


def get_tmux_train_text():
    """Capture the full text of the tmux `train` session pane."""
    return run("tmux capture-pane -pt train -S -2000 2>/dev/null", "")


def get_tmux_pane_pid(session_name="train"):
    """Get the shell PID of the tmux session pane."""
    return run(f"tmux list-panes -t {session_name} -F '#{{pane_pid}}' 2>/dev/null", "")


def get_child_pids(pid):
    """Return direct child PIDs of a given PID."""
    children = []
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        try:
            with open(f"/proc/{pid_str}/status", "rb") as f:
                status = f.read().decode("utf-8", errors="ignore")
            ppid_match = re.search(r"PPid:\s+(\d+)", status)
            if ppid_match and ppid_match.group(1) == pid:
                children.append(pid_str)
        except Exception:
            continue
    return children


def get_run_id_from_tmux_process(session_name="train"):
    """
    Find the train_starvla process running under the tmux session and return
    its --run_id argument. This is more reliable than parsing tmux text because
    it reads the actual process command line.
    """
    pane_pid = get_tmux_pane_pid(session_name)
    if not pane_pid:
        return None

    queue = [pane_pid]
    seen = {pane_pid}
    while queue:
        current = queue.pop(0)
        try:
            with open(f"/proc/{current}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except Exception:
            cmdline = ""

        if "train_starvla" in cmdline:
            m = re.search(r"--run_id\s+(\S+)", cmdline)
            if m:
                return m.group(1)

        for child in get_child_pids(current):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return None


def find_run_id_from_tmux_text(text):
    """Extract the run_id from the tmux `train` session text (fallback)."""
    if not text:
        return None
    # Prefer the expanded RUN_ID= line printed by the run script (e.g. RUN_ID=P0-M6-...)
    # This appears after the shell command that launched training.
    m = re.search(r"^RUN_ID=([A-Za-z0-9_\.\-]+)$", text, re.MULTILINE)
    if m:
        return m.group(1)
    # Fallback: extract run_id from checkpoint paths shown in tmux output
    matches = re.findall(
        r"(?:/disk/rl/starVLA/playground/Checkpoints|/localdisk-tmp)/([^/\s]+)/checkpoints/steps_\d+",
        text,
    )
    if matches:
        return matches[-1]
    return None


def find_run_id_from_tmux():
    """Determine the run_id currently running in tmux `train`."""
    run_id = get_run_id_from_tmux_process("train")
    if run_id:
        return run_id
    return find_run_id_from_tmux_text(get_tmux_train_text())


def get_run_id_base(run_id):
    """Extract the experiment base id, e.g. P0-M7-E-H2a-04 from full run_id."""
    parts = run_id.split("_")
    base_parts = []
    for p in parts:
        base_parts.append(p)
        if re.match(r"^[PE]\d+-M\d+-[A-Z]\d+[a-z]?-\d+[a-z]?$", p):
            break
    return "_".join(base_parts) if base_parts else run_id


def find_markdown(run_id):
    """Find the markdown tracker file for a run_id."""
    for candidate in DOCS_DIR.glob(f"*{run_id}*.md"):
        return candidate
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


def parse_tmux_progress():
    """Parse the latest tqdm-style progress bar from tmux `train` session."""
    text = get_tmux_train_text()
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
            "sec_per_it": float(m[2]),
            "data_time": None,
            "model_time": None,
        }
    return None


def parse_step_loss_logs(text, max_entries=10):
    """Parse the last ~10 Step loss lines from tmux text."""
    pattern = r"Step\s+(\d+),\s+Loss:\s+\{([^}]+)\}"
    matches = re.findall(pattern, text)
    entries = []
    for step_str, inner in matches[-max_entries:]:
        try:
            step = int(step_str)
        except ValueError:
            continue
        loss_match = re.search(r"'action_dit_loss':\s*([\d.eE+-]+)", inner)
        loss = float(loss_match.group(1)) if loss_match else None
        entries.append((step, loss))
    return entries


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


def get_cpu_mem_stats():
    text = run("free -h 2>/dev/null | head -2", "")
    if not text:
        return None
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return None
    # Parse Mem line
    parts = lines[1].split()
    if len(parts) < 4:
        return None
    return {
        "total": parts[1],
        "used": parts[2],
        "free": parts[3],
    }


def get_disk_stats():
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


def parse_elapsed(elapsed_str):
    """Parse elapsed time string like 1:23:45 or 23:45 into seconds."""
    parts = elapsed_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
    except Exception:
        pass
    return 0


def update_markdown(md_path, run_id, prog, latest_ckpt, gpu, cpu_mem, disk):
    content = md_path.read_text(encoding="utf-8")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")

    # Build automatic monitoring section
    rows = [
        f"| 监控时间 | {now} |",
        f"| 训练状态 | 🟢 运行中（来自 tmux `train`） |",
        f"| run_id | `{run_id}` |",
    ]

    if prog:
        rows.extend([
            f"| 当前步数 | **{prog['step']} / {prog['total_steps']}** |",
            f"| 完成比例 | {prog['percent']}% |",
            f"| 训练速度 | ~{prog['sec_per_it']:.2f} s/it |",
            f"| data_time | {prog['data_time']} s |" if prog.get("data_time") is not None else "",
            f"| model_time | {prog['model_time']} s |" if prog.get("model_time") is not None else "",
            f"| 已运行时间 | {prog['elapsed']} |",
            f"| 预计剩余时间 | {prog['eta']} |",
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

    if cpu_mem:
        rows.extend([
            f"| 内存总量 | {cpu_mem['total']} |",
            f"| 内存已用 | {cpu_mem['used']} |",
            f"| 内存空闲 | {cpu_mem['free']} |",
        ])

    if disk:
        for mount in ["/disk/rl", "/localdisk-tmp"]:
            info = disk.get(mount)
            if info:
                rows.append(
                    f"| 存储 `{mount}` | {info['used']} / {info['size']} ({info['use_pct']} 已用) |"
                )

    # Cost estimate
    if prog:
        sec_per_it = prog.get("sec_per_it", 0)
        if sec_per_it:
            cost_per_step = 5.58 / 3600 * sec_per_it
            elapsed_sec = parse_elapsed(prog.get("elapsed", "0"))
            elapsed_cost = 5.58 / 3600 * elapsed_sec
            total_sec_est = sec_per_it * prog["total_steps"]
            total_cost_est = 5.58 / 3600 * total_sec_est
            rows.extend([
                f"| 每 step 成本 | ~{cost_per_step:.4f} 元 |",
                f"| 已产生成本 | ~{elapsed_cost:.2f} 元 |",
                f"| 完整训练预估成本 | ~{total_cost_est:.2f} 元 |",
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

    md_path.write_text(content, encoding="utf-8")
    print(f"Updated {md_path}")


def main():
    tmux_text = get_tmux_train_text()
    if not tmux_text:
        print("tmux session `train` not found or empty.")
        return

    run_id = find_run_id_from_tmux()
    if not run_id:
        print("No --run_id found in tmux `train` session.")
        return

    print(f"Monitoring tmux `train` run: {run_id}")
    prog = parse_tmux_progress()
    latest_ckpt = get_latest_checkpoint(run_id)
    gpu = get_gpu_stats()
    cpu_mem = get_cpu_mem_stats()
    disk = get_disk_stats()

    md_path = find_markdown(run_id)
    if not md_path:
        print(f"No markdown tracker found for {run_id}, skipping.")
        return

    update_markdown(md_path, run_id, prog, latest_ckpt, gpu, cpu_mem, disk)


if __name__ == "__main__":
    main()
