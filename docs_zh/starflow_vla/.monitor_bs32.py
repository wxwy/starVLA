#!/usr/bin/env python3
"""每小时自动监控当前机器 tmux train 中运行的 StarVLA 实验并推送更新。

- 从 ps aux 检测 run_id（过滤 train_starvla.py）
- 从 tmux capture-pane 抓取 step / loss
- 获取 GPU（nvidia-smi）和 Docker 内存（cgroup v2）
- 更新 docs_zh/starflow_vla/bs32/ 下对应 run_id 的 md tracker
- git commit & push
"""
import subprocess, re, os, sys, glob
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

def detect_experiment():
    """从 ps aux 检测当前运行的 StarVLA 实验，返回 (run_id, md_file, exp_code, display_step)。"""
    ps_out = run("ps aux | grep 'train_starvla.py' | grep -v grep")
    if not ps_out:
        return None, None, None, None

    # 解析 run_id
    m = re.search(r'--run_id\s+(\S+)', ps_out)
    if not m:
        print('ERROR: 无法从进程参数解析 run_id')
        return None, None, None, None

    run_id = m.group(1)

    # 匹配 bs32 目录下的 md 文件
    md_candidates = glob.glob(f'docs_zh/starflow_vla/bs32/*{run_id}*.md')
    md_file = md_candidates[0] if md_candidates else None

    # 提取实验代号（如 H2b-02, H2a-04）
    exp_match = re.search(r'(P\d+-M\d+-E-H2[a-z]-\d+)', run_id)
    exp_code = exp_match.group(1) if exp_match else run_id.split('_')[0]

    # 提取 display_step（如 "continuous_ft32" 或 "ft64"）
    step_match = re.search(r'_(ft\d+|continuous_ft\d+)', run_id)
    display_step = step_match.group(1) if step_match else 'unknown'

    return run_id, md_file, exp_code, display_step


def get_training_metrics():
    """从 tmux train 抓取训练指标。"""
    out = run('tmux capture-pane -t train -p -S -80 2>/dev/null')
    if not out:
        return None

    # 进度条 — 取最后一条
    bars = re.findall(r'(\d+)/(\d+)\s*\[.*?([\d.]+)s/it,\s*data_times=([\d.]+),\s*model_times=([\d.]+)\]', out)
    if not bars:
        return None
    cur_step, max_step, speed, data_time, model_time = bars[-1]

    # Loss — 跨行匹配
    losses = re.findall(r"Step (\d+),.*?action_dit_loss':.*?([\d]+\.[\d]+)", out, re.DOTALL)
    if not losses:
        return None
    last_loss_step, last_loss_val = losses[-1]

    # 学习率
    lr_matches = re.findall(
        r"Step \d+,.*?learning_rate/action_model':\s*([\d.e+-]+).*?"
        r"learning_rate/base':\s*([\d.e+-]+).*?"
        r"epoch':\s*([\d.]+)",
        out, re.DOTALL
    )
    lr_action, lr_base, epoch = lr_matches[-1] if lr_matches else ('?', '?', '?')

    return {
        'cur_step': cur_step, 'max_step': max_step, 'speed': speed,
        'data_time': data_time, 'model_time': model_time,
        'loss_step': last_loss_step, 'loss_val': last_loss_val,
        'lr_action': lr_action, 'lr_base': lr_base, 'epoch': epoch,
    }


def get_system_metrics(run_id):
    """获取 GPU 和 Docker 内存指标。"""
    # GPU
    gpu = run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu --format=csv,noheader 2>/dev/null")
    if gpu:
        parts = [x.strip().replace(' %','%').replace(' MiB','').replace(' W','') for x in gpu.split(',')]
        gpu_util, gpu_mem_used, gpu_mem_total, gpu_power, gpu_power_limit, gpu_temp = parts
        try:
            gpu_mem_pct = f"{float(gpu_mem_used)/float(gpu_mem_total)*100:.1f}%"
        except:
            gpu_mem_pct = '?'
    else:
        gpu_util = gpu_mem_used = gpu_mem_total = gpu_power = gpu_power_limit = gpu_temp = '?'
        gpu_mem_pct = '?'

    # Docker 内存 (cgroup v2)
    try:
        with open('/sys/fs/cgroup/memory.current') as f:
            mem_bytes = int(f.read().strip())
        with open('/sys/fs/cgroup/memory.max') as f:
            mem_max_raw = f.read().strip()
        mem_limit = 56 * 1024**3  # 56 GiB (用户指定)
        mem_used_gib = mem_bytes / (1024**3)
        mem_pct = f"{mem_bytes/mem_limit*100:.1f}%"
    except Exception:
        mem_used_gib = 0
        mem_pct = '?'

    # 最新 checkpoint — 从 run_id 匹配对应目录
    latest_ckpt = 'N/A'
    run_suffix = run_id.split('_')[-1]  # e.g. "260702_2021"
    for ckpt_dir in glob.glob(f'/disk/rl/starVLA/playground/Checkpoints/*{run_suffix}/checkpoints'):
        ckpts = sorted(glob.glob(f'{ckpt_dir}/steps_*'))
        if ckpts:
            latest_ckpt = os.path.basename(ckpts[-1])
        break

    return {
        'gpu_util': gpu_util, 'gpu_mem_used': gpu_mem_used,
        'gpu_mem_total': gpu_mem_total, 'gpu_mem_pct': gpu_mem_pct,
        'gpu_power': gpu_power, 'gpu_power_limit': gpu_power_limit,
        'gpu_temp': gpu_temp,
        'mem_used_gib': f'{mem_used_gib:.1f}',
        'mem_pct': mem_pct,
        'mem_limit_gib': '56',
        'latest_ckpt': latest_ckpt or 'N/A',
    }


def update_md(md_file, run_id, exp_code, display_step, train, sys_metrics):
    """更新 md 文件中的关键字段。"""
    now = datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S CST')

    with open(md_file) as f:
        content = f.read()

    cur = train['cur_step']
    max_s = train['max_step']
    try:
        pct = f"{int(cur)/int(max_s)*100:.1f}"
    except:
        pct = '?'

    sm = sys_metrics

    # 时间戳
    content = re.sub(r'> \*\*当前更新\*\*: .*', f'> **当前更新**: {now}', content)
    content = re.sub(r'> 最后更新：.*', f'> 最后更新：{now}', content)

    # 训练进度表
    content = re.sub(r'\*\*当前 Step\*\* \| .*', f'**当前 Step** | **{cur} / {max_s}**（{pct}%）', content)
    content = re.sub(r'\*\*完成比例\*\* \| .*', f'**完成比例** | {pct}%', content)
    content = re.sub(r'\*\*单步耗时\*\* \| .*', f'**单步耗时** | ~{train["speed"]} s/it', content)
    content = re.sub(r'\*\*数据加载耗时\*\* \| .*', f'**数据加载耗时** | ~{train["data_time"]} s', content)
    content = re.sub(r'\*\*模型前向/反向耗时\*\* \| .*', f'**模型前向/反向耗时** | ~{train["model_time"]} s', content)
    content = re.sub(r'\*\*最新 checkpoint\*\* \| .*', f'**最新 checkpoint** | `{sm["latest_ckpt"]}`', content)

    # GPU
    content = re.sub(r'\*\*GPU 利用率\*\* \| .*', f'**GPU 利用率** | {sm["gpu_util"]}', content)
    content = re.sub(r'\*\*显存使用\*\* \| .*', f'**显存使用** | {sm["gpu_mem_used"]} MiB / {sm["gpu_mem_total"]} MiB（{sm["gpu_mem_pct"]}）', content)
    content = re.sub(r'\*\*功耗\*\* \| .*', f'**功耗** | {sm["gpu_power"]} W / {sm["gpu_power_limit"]} W', content)
    content = re.sub(r'\*\*温度\*\* \| .*', f'**温度** | {sm["gpu_temp"]}°C', content)

    # Docker 内存
    content = re.sub(r'\*\*Docker 内存总量\*\* \| .*', f'**Docker 内存总量** | {sm["mem_limit_gib"]} GiB', content)
    content = re.sub(r'\*\*Docker 内存已用\*\* \| .*', f'**Docker 内存已用** | ~{sm["mem_used_gib"]} GiB（{sm["mem_pct"]}）', content)

    with open(md_file, 'w') as f:
        f.write(content)

    return now, cur, max_s, pct


def main():
    run_id, md_file, exp_code, display_step = detect_experiment()

    if not run_id:
        print('⏭️ 无运行中的训练进程，跳过')
        return

    if not md_file:
        print(f'⚠️ 未找到 run_id={run_id} 对应的 md 文件，跳过')
        return

    train = get_training_metrics()
    if not train:
        print('⚠️ 无法从 tmux 获取训练指标，跳过')
        return

    sm = get_system_metrics(run_id)
    now, cur, max_s, pct = update_md(md_file, run_id, exp_code, display_step, train, sm)

    print(f'[{now}] ✅ {exp_code} step={cur}/{max_s} ({pct}%), '
          f'speed={train["speed"]}s/it, loss={train["loss_val"]}')

    # git 操作
    os.chdir('/disk/rl/starVLA')
    run(f'git add {md_file}')
    diff = run('git diff --cached --quiet 2>/dev/null')
    # diff 返回空字符串表示无变更，返回非空表示有变更
    if diff or run('git diff --cached --name-only 2>/dev/null'):
        msg = (f"docs: update {exp_code} tracker — step {cur} ({pct}%)\n\n"
               f"Auto-update at {now}:\n"
               f"- Step {cur}/{max_s} ({pct}%), speed {train['speed']}s/it\n"
               f"- Loss: {train['loss_val']}, GPU: {sm['gpu_util']}, {sm['gpu_temp']}°C\n"
               f"- Memory: {sm['mem_used_gib']}/{sm['mem_limit_gib']} GiB Docker\n\n"
               f"Co-Authored-By: Claude <noreply@anthropic.com>")
        run(f'git commit -m "{msg}"')
        push = run('git push origin merge-official-starvla-dev 2>&1')
        if 'Everything up-to-date' in push or 'error' not in push.lower():
            print(f'[{now}] ✅ 已提交推送')
        else:
            print(f'[{now}] ⚠️ push 失败: {push}')
    else:
        print(f'[{now}] ⏭️ 无变更')


if __name__ == '__main__':
    main()
