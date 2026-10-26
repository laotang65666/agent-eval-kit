"""运行器：按臂 × 任务跑一遍，每次运行独立沙盒，结果落盘。

"臂"（arm）就是一组实验条件。这个包不规定你有几条臂，
但强烈建议至少包含下面四条，否则结论会被一句话问穿：

    A  baseline     什么都不加
    B  prompt_only  只把提示词写得尽可能强
    C  treatment    你的干预
    D  upfront      运行前就把"该做什么"全告诉它（近似 oracle）

B 排除"你只是换了提示词"，D 排除"提前告诉它就够了"。
缺了它们，"C 比 A 好"可以被解释成任何原因。
"""
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent_eval.scorer import score, summarize
from agent_eval.tools import Sandbox


@dataclass
class Arm:
    """一组实验条件。三个钩子足以表达绝大多数干预方式。"""
    name: str
    system_prompt: object = None      # str，或 callable(task) -> str
    post_turn_hook: object = None     # callable(messages, at_end) -> str | None
    max_interventions: int = 3
    meta: dict = field(default_factory=dict)

    def prompt_for(self, task):
        if callable(self.system_prompt):
            return self.system_prompt(task)
        return self.system_prompt


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def run_one(agent, registry, task, arm, out_dir, repeat=0):
    """跑一条任务。agent 需要实现 run(task, registry, arm, sandbox, system_prompt) -> Trace。"""
    tag = f"{arm.name}__{task.id}__r{repeat}"
    d = Path(out_dir) / _safe_name(tag)
    if d.exists():
        shutil.rmtree(d)
    sb = Sandbox(d / "workspace")
    sb.seed(task.fixtures)

    t0 = time.time()
    try:
        trace = agent.run(task=task, registry=registry, arm=arm,
                          sandbox=sb, system_prompt=arm.prompt_for(task))
    except Exception as e:  # noqa: BLE001
        # 一次失败不该毁掉整批实验：记一行错误，继续
        row = _error_row(task, arm, tag, e)
        _dump(d, row, [], trace_err=str(e))
        return row

    row = score(trace, task, sb)
    row.update({"arm": arm.name, "run": tag,
                "wall_ms": int((time.time() - t0) * 1000)})
    _dump(d, row, trace.messages)
    return row


def _error_row(task, arm, tag, err):
    return {
        "task_id": task.id, "arm": arm.name, "run": tag,
        "pred_tools": [], "exp_tools": list(task.expected_tools or []),
        "missing": [], "extra": [], "forbidden": [],
        "step_recall": 0.0, "order_ok": False, "min_calls_ok": False,
        "redundant_calls": 0, "artifact_ok": False,
        "artifact_reason": "run_failed", "success": False,
        "fake_success": False, "turns": 0, "interventions": 0,
        "prompt_tokens": 0, "bad_format": 0, "finished": False,
        "latency_ms": 0, "wall_ms": 0, "error": f"{type(err).__name__}: {err}",
    }


def _dump(run_dir: Path, row: dict, messages: list, trace_err: str = ""):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    if messages:
        with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    if trace_err:
        (run_dir / "error.txt").write_text(trace_err, encoding="utf-8")


def run_grid(agent, registry, tasks, arms, out_dir="runs", repeats=1,
             group_key=None, cooldown=0.0, verbose=True):
    """跑完整网格。

    group_key: 可选，callable(task) -> str，用于把任务分成几组分别汇总
               （比如 "static" / "dynamic"）。分组必须分开看，否则
               两类难度混在一个均值里，什么差异都会糊掉。
    """
    rows = []
    for arm in arms:
        for r in range(repeats):
            for t in tasks:
                row = run_one(agent, registry, t, arm, out_dir, repeat=r)
                if group_key:
                    row["group"] = group_key(t)
                rows.append(row)
                if verbose:
                    mark = "✓" if row["success"] else ("△" if row["fake_success"] else "✗")
                    print(f"  {mark} {arm.name:<12}{t.id:<18}"
                          f"success={row['success']!s:<5} "
                          f"fake={row['fake_success']!s:<5} "
                          f"miss={row['missing']}")
                if cooldown:
                    time.sleep(cooldown)
    return rows


def report(rows) -> dict:
    """按 (臂, 组) 汇总。没有 group 时全部归到 "all"。"""
    buckets = {}
    for r in rows:
        buckets.setdefault((r["arm"], r.get("group", "all")), []).append(r)
    return {k: summarize(v) for k, v in sorted(buckets.items())}


def save(rows, path="results"):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (p / "summary.json").write_text(
        json.dumps({f"{a}|{g}": s for (a, g), s in report(rows).items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return p
