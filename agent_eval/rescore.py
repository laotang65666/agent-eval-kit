"""离线重判：判据改了，不用重跑 API。

只要你保留了每次运行的 trace.jsonl 和 workspace，就能重放并重新评分。
这在评测工作里非常重要 —— 判据几乎一定会改，而"改了判据就重跑一遍模型"
既贵又会让数据失去可比性。
"""
import json
from pathlib import Path

from agent_eval.scorer import score, summarize
from agent_eval.tools import Sandbox
from agent_eval.types import ToolCall, Trace


def trace_from_messages(messages, task_id=""):
    """从 OpenAI 风格的 messages 还原 Trace。"""
    tr = Trace(task_id=task_id)
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            tr.turns += 1
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:  # noqa: BLE001
                    args = {}
                tr.calls.append(ToolCall(name=fn.get("name", ""), args=args))
        elif role == "system" and m.get("content"):
            if m.get("audit", False):        # 兼容带标记的注入消息
                tr.interventions.append(m["content"])
    tr.finished = True
    return tr


def rescore_runs(runs_dir, tasks, trace_name="trace.jsonl",
                 result_name="result.json", interventions_marker="【"):
    """遍历 runs_dir 下的每个子目录，重放并重新评分。

    返回 (新 rows, 变更摘要)。
    会就地更新 result.json，不删除历史（trace 保留）。
    """
    runs_dir = Path(runs_dir)
    by_id = {t.id: t for t in tasks}
    rows, changed = [], []

    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        rp = d / result_name
        if not rp.exists():
            continue
        old = json.loads(rp.read_text(encoding="utf-8"))
        tid = old.get("task_id") or old.get("id")
        task = by_id.get(tid)
        if task is None:
            rows.append(old)
            continue

        tp = d / trace_name
        msgs = []
        if tp.exists():
            msgs = [json.loads(l) for l in
                    tp.read_text(encoding="utf-8").splitlines() if l.strip()]
        tr = trace_from_messages(msgs, tid)
        tr.interventions = [m.get("content", "") for m in msgs
                            if m.get("role") == "system"
                            and str(m.get("content", "")).find(interventions_marker) >= 0]
        tr.prompt_tokens = old.get("prompt_tokens", 0)
        tr.bad_format = old.get("bad_format", 0)

        sb = Sandbox(d / "workspace")
        new = score(tr, task, sb)
        new.update({"arm": old.get("arm"), "run": old.get("run", d.name)})
        if old.get("success") != new["success"]:
            changed.append({"run": new["run"], "task": tid,
                            "success": f"{old.get('success')} → {new['success']}",
                            "fake": f"{old.get('fake_success')} → {new['fake_success']}"})
        rp.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(new)

    return rows, changed


def summary_by_arm(rows):
    buckets = {}
    for r in rows:
        buckets.setdefault(r.get("arm", "?"), []).append(r)
    return {k: summarize(v) for k, v in sorted(buckets.items())}
