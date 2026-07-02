#!/usr/bin/env python3
"""每小时自动监控 P1-M1 continuous_head 训练并推送更新。"""
import subprocess, re, os, sys
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))
NOW = datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S CST')
MD_FILE = 'docs_zh/starflow_vla/bs32/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021.md'

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def main():
    # 1. 从 tmux 抓取日志
    train_out = run('tmux capture-pane -t train -p -S -80 2>/dev/null')
    if not train_out:
        print(f'[{NOW}] ⚠️ tmux train 不可用')
        return

    # 解析进度条 — 取最后一条（最新）
    bars = re.findall(r'(\d+)/(\d+)\s*\[.*?([\d.]+)s/it,\s*data_times=([\d.]+),\s*model_times=([\d.]+)\]', train_out)
    if not bars:
        print(f'[{NOW}] ⚠️ 未找到进度条')
        return
    cur_step, max_step, speed, data_time, model_time = bars[-1]

    # 解析最新 Loss 行
    loss_lines = re.findall(r"Step (\d+), Loss: \{'action_dit_loss': ([\d.]+)", train_out)
    if not loss_lines:
        print(f'[{NOW}] ⚠️ 未找到 Loss 行')
        return
    last_loss_step, last_loss_val = loss_lines[-1]

    # 解析学习率和 epoch
    lr_line = re.findall(r"Step \d+, Loss:.*'learning_rate/action_model':\s*([\d.e+-]+).*'learning_rate/base':\s*([\d.e+-]+).*'epoch':\s*([\d.]+)", train_out, re.DOTALL)
    lr_action, lr_base, epoch = lr_line[-1] if lr_line else ('?', '?', '?')

    # 2. GPU
    gpu = run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv,noheader 2>/dev/null")
    if gpu:
        parts = [x.strip().replace(' %','%').replace(' MiB','').replace(' W','') for x in gpu.split(',')]
        gpu_util, gpu_mem_used, gpu_mem_total, gpu_power, gpu_power_limit, gpu_temp = parts
    else:
        gpu_util, gpu_mem_used, gpu_mem_total, gpu_power, gpu_temp = '?','?','?','?','?'

    # 3. Docker 内存
    mem = run("free -h | awk '/^Mem:/{print $2, $3, $7}'")
    mem_total, mem_used, mem_avail = mem.split() if mem else ('?', '?', '?')

    # 确保数值和单位之间有空格 (free -h 可能输出 "503Gi" 无空格)
    mem_total = re.sub(r'^(\d+\.?\d*)([GM])i$', r'\1 \2iB', mem_total) if mem_total else '?'
    mem_used = re.sub(r'^(\d+\.?\d*)([GM])i$', r'\1 \2iB', mem_used) if mem_used else '?'
    mem_avail = re.sub(r'^(\d+\.?\d*)([GM])i$', r'\1 \2iB', mem_avail) if mem_avail else '?'

    # 4. 最新 checkpoint
    ckpt = run('ls -t /disk/rl/starVLA/playground/Checkpoints/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021/checkpoints/ 2>/dev/null | head -1')
    latest_ckpt = ckpt or 'N/A'

    # 5. 计算百分比
    try:
        pct = f"{int(cur_step)/int(max_step)*100:.1f}"
    except:
        pct = '?'

    # 6. 更新 md 文件
    with open(MD_FILE) as f:
        content = f.read()

    content = re.sub(r'> \*\*当前更新\*\*: .*', f'> **当前更新**: {NOW}', content)
    content = re.sub(r'> 最后更新：.*', f'> 最后更新：{NOW}', content)
    content = re.sub(r'\*\*当前 Step\*\* \| \*\*[\d?]+ / [\d?]+\*\*（[\d.]+%）', f'**当前 Step** | **{cur_step} / {max_step}**（{pct}%）', content)
    content = re.sub(r'\*\*完成比例\*\* \| [\d.]+%', f'**完成比例** | {pct}%', content)
    content = re.sub(r'\*\*单步耗时\*\* \| ~[\d.–]+ s/it', f'**单步耗时** | ~{speed} s/it', content)
    content = re.sub(r'\*\*数据加载耗时\*\* \| ~[\d.–]+ s', f'**数据加载耗时** | ~{data_time} s', content)
    content = re.sub(r'\*\*模型前向/反向耗时\*\* \| ~[\d.–]+ s', f'**模型前向/反向耗时** | ~{model_time} s', content)
    content = re.sub(r'\*\*最新 checkpoint\*\* \| `[^`]*`', f'**最新 checkpoint** | `{latest_ckpt}`', content)
    content = re.sub(r'\*\*GPU 利用率\*\* \| [\d?%]+', f'**GPU 利用率** | {gpu_util}', content)
    mem_pct = f"{float(gpu_mem_used)/float(gpu_mem_total)*100:.1f}%" if gpu_mem_used.isdigit() and gpu_mem_total.isdigit() else "?"
    content = re.sub(r'\*\*显存使用\*\* \| .*', f'**显存使用** | {gpu_mem_used} MiB / {gpu_mem_total} MiB（{mem_pct}）', content)
    content = re.sub(r'\*\*功耗\*\* \| [\d.]+\s*W\s*/\s*[\d.]+\s*W', f'**功耗** | {gpu_power} W / {gpu_power_limit} W', content)
    content = re.sub(r'\*\*温度\*\* \| [\d]+°C', f'**温度** | {gpu_temp}°C', content)
    content = re.sub(r'\*\*Docker 内存总量\*\* \| .*', f'**Docker 内存总量** | {mem_total}', content)
    content = re.sub(r'\*\*Docker 内存已用\*\* \| .*', f'**Docker 内存已用** | ~{mem_used}', content)
    content = re.sub(r'\*\*Docker 内存可用\*\* \| .*', f'**Docker 内存可用** | ~{mem_avail}', content)

    with open(MD_FILE, 'w') as f:
        f.write(content)

    print(f'[{NOW}] ✅ md 已更新: step={cur_step}/{max_step} ({pct}%), speed={speed}s/it, loss={last_loss_val}')

    # 7. git 提交推送
    os.chdir('/disk/rl/starVLA')
    run(f'git add -f {MD_FILE}')
    diff_check = run('git diff --cached --quiet 2>/dev/null')
    if diff_check is not None:
        msg = (
            f"docs: update P1-M1 continuous training tracker — step {cur_step} ({pct}%)\n\n"
            f"Auto-update at {NOW}:\n"
            f"- Step {cur_step}/{max_step} ({pct}%), speed {speed}s/it\n"
            f"- Loss: {last_loss_val}, LR action_model: {lr_action}\n"
            f"- GPU: {gpu_util}, {gpu_mem_used}/{gpu_mem_total} MiB, {gpu_temp}°C\n"
            f"- Memory: {mem_used}/{mem_total} Docker\n\n"
            f"Co-Authored-By: Claude <noreply@anthropic.com>"
        )
        run(f'git commit -m "{msg}"')
        push_out = run('git push origin merge-official-starvla-dev 2>&1')
        print(f'[{NOW}] ✅ 已提交推送: {push_out}')
    else:
        print(f'[{NOW}] ⏭️ 无变更')

if __name__ == '__main__':
    main()
