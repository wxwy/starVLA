#!/usr/bin/env python3
"""StarFlow VLA 多 Worker 任务池评估管理器

启动方式:
  1) 先启动 server_policy.py (一个 server 服务所有 worker)
  2) 再运行此脚本

用法:
  python playground/eval_pool_manager.py \\
    --ckpt-path /path/to/checkpoint \\
    --port 6694 \\
    --workers 4 \\
    --suites libero_goal,libero_10,libero_object,libero_spatial \\
    --video-root /disk/rl/starVLA/playground/eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s | %(message)s",
    datefmt="%m/%d [%H:%M:%S]",
)

LIBERO_PY = "/disk/rl/starVLA/.libero/bin/python"
EVAL_SCRIPT = "examples/LIBERO/eval_files/eval_libero.py"

# Each LIBERO suite has exactly 10 tasks (0-9)
SUITE_TASKS: dict[str, int] = {
    "libero_goal": 10,
    "libero_10": 10,
    "libero_object": 10,
    "libero_spatial": 10,
}
# Conservative step limits from eval_libero.py
SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


class TaskPool:
    def __init__(
        self,
        ckpt_path: str,
        port: int,
        workers: int,
        suites: list[str],
        video_root: str,
        num_trials: int = 50,
        max_tasks: int = 0,
        resume: bool = False,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.port = port
        self.workers = workers
        self.suites = suites
        self.video_root = Path(video_root)
        self.num_trials = num_trials
        self._max_tasks = max_tasks
        self.resume = resume
        self.exp_name = self.ckpt_path.parent.parent.name
        self.step_name = self.ckpt_path.name

        self.task_queue: Queue[tuple[str, int]] = Queue()
        self.results: dict[tuple[str, int], dict[str, Any]] = {}
        self.results_lock = Lock()
        self.active_count = 0
        self.active_lock = Lock()
        self.abort = False
        self.abort_lock = Lock()
        self.runtime_error_threshold = 0.3  # >30% runtime_error → 熔断

        # Build task queue
        task_count = 0
        for suite in suites:
            n_tasks = SUITE_TASKS.get(suite, 10)
            for task_id in range(n_tasks):
                self.task_queue.put((suite, task_id))
                task_count += 1
                if self._max_tasks > 0 and task_count >= self._max_tasks:
                    break
            if self._max_tasks > 0 and task_count >= self._max_tasks:
                break

        self.total_tasks = self.task_queue.qsize()
        logging.info(
            f"任务池初始化: {self.total_tasks} tasks, {workers} workers, "
            f"exp={self.exp_name}, step={self.step_name}, port={port}"
        )

    def _check_runtime_error_abort(
        self, worker_id: int, suite: str, task_id: int, report: dict[str, Any]
    ) -> None:
        """检查 task 报告的 runtime_error 比例，超过阈值则熔断中断所有 worker"""
        episodes = report.get("episodes", [])
        if not episodes:
            return
        runtime_errs = sum(1 for ep in episodes if ep.get("runtime_error"))
        ratio = runtime_errs / len(episodes)
        if ratio <= self.runtime_error_threshold:
            return
        # 熔断！
        with self.abort_lock:
            if self.abort:
                return  # 已触发过，不再重复
            self.abort = True
        sample_err = next(
            (ep.get("runtime_error", "")[:200] for ep in episodes if ep.get("runtime_error")),
            "",
        )
        logging.critical(
            f"🔥 熔断！[W{worker_id}] {suite}/task_{task_id} "
            f"runtime_error={runtime_errs}/{len(episodes)} ({ratio*100:.0f}%) "
            f"超过阈值 {self.runtime_error_threshold*100:.0f}%\n"
            f"  sample: {sample_err}\n"
            f"  Server ckpt: {report.get('checkpoint_path', '?')}\n"
            f"  Server port: {self.port}\n"
            f"  可能原因：OOM / ckpt 损坏 / server 异常，请检查后重试"
        )
        # 清空队列，阻止新 task 启动
        while True:
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except Exception:
                break

    def _worker_loop(self, worker_id: int) -> None:
        """Worker 线程：从队列取任务，运行 eval_libero.py"""
        while True:
            with self.abort_lock:
                if self.abort:
                    break
            try:
                suite, task_id = self.task_queue.get_nowait()
            except Exception:
                break  # 队列为空

            # 输出目录: {video_root}/{suite}/{exp_name}/{step_name}/
            out_dir = (
                self.video_root
                / suite
                / self.exp_name
                / self.step_name
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            report_filename = f"eval_report_task_{task_id}.json"

            # 检查是否已完成
            report_path = out_dir / report_filename
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                    if report.get("total_episodes", 0) >= self.num_trials:
                        sr = report.get("success_rate", 0) * 100
                        logging.info(
                            f"[W{worker_id}] ✅ {suite}/task_{task_id} 已完成 "
                            f"(SR={sr:.1f}%), 跳过"
                        )
                        self._check_runtime_error_abort(worker_id, suite, task_id, report)
                        with self.results_lock:
                            self.results[(suite, task_id)] = report
                        self.task_queue.task_done()
                        continue
                except Exception:
                    pass  # 报告损坏，重新跑

            logging.info(f"[W{worker_id}] 🚀 {suite}/task_{task_id} 开始...")
            t0 = time.time()

            cmd = [
                LIBERO_PY, EVAL_SCRIPT,
                "--args.pretrained-path", str(self.ckpt_path),
                "--args.host", "127.0.0.1",
                "--args.port", str(self.port),
                "--args.task-suite-name", suite,
                "--args.task-ids", str(task_id),
                "--args.num-trials-per-task", str(self.num_trials),
                "--args.video-out-path", str(out_dir),
                "--args.report-filename", report_filename,
            ]
            if getattr(self, 'resume', False):
                cmd.append("--args.resume-eval")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2h max per task
                    env={
                        **os.environ,
                        "PYTHONPATH": "/disk/rl/starVLA/LIBERO:/disk/rl/starVLA",
                        "LIBERO_CONFIG_PATH": "/disk/rl/starVLA/LIBERO/libero/libero",
                        "LIBERO_HOME": "/disk/rl/starVLA/LIBERO",
                        "MUJOCO_GL": "glfw",
                    },
                )
            except subprocess.TimeoutExpired:
                logging.error(
                    f"[W{worker_id}] ⏰ {suite}/task_{task_id} 超时(7200s)，重新入队!"
                )
                self.task_queue.put((suite, task_id))
                self.task_queue.task_done()
                continue
            except Exception as e:
                logging.error(
                    f"[W{worker_id}] ❌ {suite}/task_{task_id} 子进程异常: {e}，重新入队!"
                )
                self.task_queue.put((suite, task_id))
                self.task_queue.task_done()
                continue

            elapsed = time.time() - t0
            task_report = None
            success_rate = 0.0

            # 读取结果
            if report_path.exists():
                try:
                    task_report = json.loads(report_path.read_text())
                    success_rate = task_report.get("success_rate", 0) * 100
                    n_ep = task_report.get("total_episodes", 0)
                    n_ok = task_report.get("total_successes", 0)
                    logging.info(
                        f"[W{worker_id}] ✅ {suite}/task_{task_id} done: "
                        f"SR={success_rate:.1f}% ({n_ok}/{n_ep}), {elapsed:.0f}s"
                    )
                except Exception as e:
                    logging.error(
                        f"[W{worker_id}] ❌ {suite}/task_{task_id} 报告解析失败: {e}"
                    )
            # 熔断检查：runtime_error 比例过高时中断全部 worker
            if task_report is not None:
                self._check_runtime_error_abort(worker_id, suite, task_id, task_report)
            if self.abort:
                # 标记已完成，让 pool 退出
                with self.results_lock:
                    self.results[(suite, task_id)] = task_report
                self.task_queue.task_done()
                continue
            # 正常完成：检查是否足量，不足则重试
            n_ep = task_report.get("total_episodes", 0) if task_report else 0
            if n_ep < self.num_trials:
                logging.error(
                    f"[W{worker_id}] ⚠️ {suite}/task_{task_id} 仅完成 "
                    f"{n_ep}/{self.num_trials} 轮，重新入队!"
                )
                # 重新入队重试
                self.task_queue.put((suite, task_id))
                self.task_queue.task_done()
                continue
            with self.results_lock:
                self.results[(suite, task_id)] = task_report
            self.task_queue.task_done()
            # 无报告生成时的错误处理
            if not report_path.exists():
                if result.returncode != 0:
                    logging.error(
                        f"[W{worker_id}] ❌ {suite}/task_{task_id} 无报告生成 "
                        f"(exit={result.returncode})，重新入队!"
                    )
                    self.task_queue.put((suite, task_id))
                    self.task_queue.task_done()
                    continue
                logging.error(
                    f"[W{worker_id}] ❌ {suite}/task_{task_id} 无报告生成!\n"
                    f"  stdout: {result.stdout[-500:]}\n  stderr: {result.stderr[-500:]}"
                )

    def run(self) -> dict[str, dict[str, Any]]:
        """启动所有 worker，等待完成，返回合并后的 suite 级报告"""
        threads = []
        for i in range(self.workers):
            t = Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        # 等待队列清空
        self.task_queue.join()

        # 等待所有线程自然结束
        for t in threads:
            t.join(timeout=5)

        # 合并报告
        suite_reports: dict[str, dict[str, Any]] = {}
        for suite in self.suites:
            suite_report = self._merge_suite_report(suite)
            if suite_report:
                suite_reports[suite] = suite_report
                self._write_merged_report(suite, suite_report)

        # 打印汇总
        self._print_summary(suite_reports)
        return suite_reports

    def _merge_suite_report(self, suite: str) -> dict[str, Any] | None:
        """合并某个 suite 下所有 task 的报告"""
        all_episodes: list[dict[str, Any]] = []
        task_results: dict[int, dict[str, Any]] = {}

        for (s, tid), report in self.results.items():
            if s != suite or report is None:
                continue
            episodes = report.get("episodes", [])
            all_episodes.extend(episodes)

            n_ok = sum(1 for ep in episodes if ep.get("success"))
            sr = (n_ok / len(episodes)) * 100 if episodes else 0
            task_results[tid] = {
                "task_id": tid,
                "episodes": len(episodes),
                "successes": n_ok,
                "success_rate": sr / 100,
            }

        if not all_episodes:
            return None

        total_ok = sum(1 for ep in all_episodes if ep.get("success"))
        total_ep = len(all_episodes)

        # Use the first report's metadata as base
        first_task_key = next(
            ((s, t) for (s, t) in self.results if s == suite and self.results[(s, t)]),
            None,
        )
        base = self.results.get(first_task_key, {}) if first_task_key else {}

        # 按 task_description 汇总
        tc: Counter[str] = Counter()
        task_rollup: dict[str, dict[str, Any]] = {}
        for ep in all_episodes:
            desc = ep.get("task_description", "unknown")
            if desc not in task_rollup:
                task_rollup[desc] = {"episodes": 0, "successes": 0}
            task_rollup[desc]["episodes"] += 1
            task_rollup[desc]["successes"] += int(ep.get("success", False))
            fc = ep.get("failure_category")
            if fc:
                tc[fc] += 1

        tasks = []
        for desc, info in sorted(task_rollup.items()):
            ep = info["episodes"]
            ok = info["successes"]
            tasks.append({
                "task_description": desc,
                "episodes": ep,
                "successes": ok,
                "success_rate": ok / ep if ep else 0.0,
            })

        return {
            "task_suite_name": suite,
            "num_trials_per_task": self.num_trials,
            "max_tasks": -1,
            "checkpoint_path": str(self.ckpt_path),
            "checkpoint_hash": base.get("checkpoint_hash"),
            "config_path": base.get("config_path"),
            "config_hash": base.get("config_hash"),
            "data_version": base.get("data_version"),
            "total_episodes": total_ep,
            "total_successes": total_ok,
            "success_rate": total_ok / total_ep if total_ep else 0.0,
            "failure_category": dict(tc),
            "tasks": tasks,
            "episodes": all_episodes,
        }

    def _write_merged_report(self, suite: str, report: dict[str, Any]) -> None:
        """写入合并后的 suite 级 eval_report.json"""
        out_dir = (
            self.video_root
            / suite
            / self.exp_name
            / self.step_name
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "eval_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sr = report["success_rate"] * 100
        logging.info(f"📊 {suite} 合并报告: SR={sr:.1f}% -> {report_path}")

    def _flatten_task_dirs(self, suite: str) -> None:
        """将 task_x 下所有文件移到上级目录，删除 task_x 目录"""
        base = (
            self.video_root
            / suite
            / self.exp_name
            / self.step_name
        )
        if not base.exists():
            return
        import shutil
        for task_dir in sorted(base.glob("task_*")):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name  # "task_0", "task_1", ...
            for f in task_dir.iterdir():
                if f.is_dir():
                    continue
                # Rename eval_report.json → task_0_eval_report.json
                if f.name == "eval_report.json":
                    dest = base / f"{task_id}_eval_report.json"
                else:
                    dest = base / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
            # Remove empty task dir
            remaining = list(task_dir.iterdir())
            if not remaining:
                shutil.rmtree(str(task_dir))

    def _print_summary(self, suite_reports: dict[str, dict[str, Any]]) -> None:
        print("\n" + "=" * 60)
        print(f"  📊 {self.exp_name} @ {self.step_name} 评估汇总")
        print("=" * 60)
        for suite, report in suite_reports.items():
            sr = report["success_rate"] * 100
            n_ok = report["total_successes"]
            n_ep = report["total_episodes"]
            n_tasks = len(report.get("tasks", []))
            print(f"  {suite:20s} SR={sr:5.1f}%  ({n_ok:3d}/{n_ep:3d})  {n_tasks} tasks")
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="StarFlow VLA 多 Worker 任务池评估")
    parser.add_argument("--ckpt-path", required=True, help="Checkpoint 路径")
    parser.add_argument("--port", type=int, default=6694, help="Server 端口")
    parser.add_argument("--workers", type=int, default=2, help="并发 Worker 数")
    parser.add_argument(
        "--suites",
        default="libero_goal,libero_10,libero_object,libero_spatial",
        help="逗号分隔的 suite 列表",
    )
    parser.add_argument(
        "--video-root",
        default="/disk/rl/starVLA/playground/eval_results",
        help="视频输出根目录",
    )
    parser.add_argument("--num-trials", type=int, default=50, help="每个 task 的试次数")
    parser.add_argument("--max-tasks", type=int, default=0, help="限制总任务数（0=不限）")
    parser.add_argument("--resume", action="store_true", default=False, help="启用 resume 模式续跑已有部分进度的任务")
    args = parser.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    pool = TaskPool(
        ckpt_path=args.ckpt_path,
        port=args.port,
        workers=args.workers,
        suites=suites,
        video_root=args.video_root,
        num_trials=args.num_trials,
        max_tasks=args.max_tasks,
    )
    pool.resume = args.resume
    pool.run()


if __name__ == "__main__":
    main()
