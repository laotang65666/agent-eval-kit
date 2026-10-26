"""判分器：四层判据 + 假完成指标。

为什么是四层，而不是只看成功率：

    step_recall   该做的步骤做了几成       ← 最敏感，看趋势
    order_ok      顺序对不对               ← 先写文件后建表，内容其实是空壳
    artifact_ok   产物真的造出来了吗        ← 客观判据，不靠模型自述
    min_calls_ok  该重查的时候重查了吗      ← 动态任务专用

为什么要有 fake_success：

    success = 以上全过。于是存在一类"产物造出来了、但过程不对"的运行，
    它的 success 是 False，但产物看起来是好的。
    如果只统计产物，这类失败永远看不见；如果只统计 success，
    你又分不清它和"压根没写出来"的区别。

    fake_success = artifact_ok and not success
    —— 它专门指认"看起来完成了、其实偷工减料"的那一类。
"""
from pathlib import Path


def is_subsequence(need, seq):
    it = iter(seq)
    return all(x in it for x in need)


def check_artifact(task, sandbox):
    """返回 (是否通过, 原因)。全部断言都在这里，不分散。"""
    fn = task.expect_file
    if not fn:
        if not (task.expect_contains or task.expect_any or task.expect_shorter_than):
            return True, "no_artifact_expected"
        # 没指定文件时，退化为"沙盒里最后写入的那个文件"
        files = sandbox.files()
        if not files:
            return False, "沙盒里没有任何产物"
        fn = files[-1]

    if not sandbox.exists(fn):
        return False, f"缺少产物 {fn}"
    text = sandbox.read(fn)
    if not text.strip():
        return False, f"产物 {fn} 为空"

    for needle in task.expect_contains or []:
        if needle not in text:
            return False, f"缺少关键内容：{needle}"

    if task.expect_any and not any(n in text for n in task.expect_any):
        return False, f"不含任一关键内容：{task.expect_any}"

    src = task.expect_shorter_than
    if src and sandbox.exists(src):
        src_text = sandbox.read(src)
        if len(text) >= len(src_text):
            return False, f"不比原文短（{len(text)} >= {len(src_text)}）"

    return True, "ok"


def _counts(names):
    c = {}
    for n in names:
        c[n] = c.get(n, 0) + 1
    return c


def score(trace, task, sandbox):
    """一次运行 → 一行结果。字段名保持稳定，便于汇总与离线重判。"""
    pred = list(trace.names)
    exp = list(task.expected_tools or [])
    pred_set, exp_set = set(pred), set(exp)

    missing = sorted(exp_set - pred_set)
    extra = sorted(pred_set - exp_set)
    forbidden = sorted(pred_set & set(task.forbidden_tools or []))
    step_recall = (len(pred_set & exp_set) / len(exp_set)) if exp_set else 1.0
    order_ok = is_subsequence(exp, pred) if exp else True

    counts = _counts(pred)
    min_calls_ok = all(counts.get(k, 0) >= v
                       for k, v in (task.expect_min_calls or {}).items())

    art_ok, art_reason = check_artifact(task, sandbox)

    success = bool(not missing and not forbidden and order_ok
                   and min_calls_ok and art_ok)
    fake_success = bool(art_ok and not success)

    return {
        "task_id": task.id,
        "pred_tools": pred,
        "exp_tools": exp,
        "missing": missing,
        "extra": extra,
        "forbidden": forbidden,
        "step_recall": round(step_recall, 4),
        "order_ok": order_ok,
        "min_calls_ok": min_calls_ok,
        "redundant_calls": len(pred) - len(pred_set),
        "artifact_ok": art_ok,
        "artifact_reason": art_reason,
        "success": success,
        "fake_success": fake_success,
        "turns": trace.turns,
        "interventions": len(trace.interventions),
        "prompt_tokens": trace.prompt_tokens,
        "bad_format": trace.bad_format,
        "finished": trace.finished,
        "latency_ms": trace.latency_ms,
    }


def summarize(rows):
    n = len(rows) or 1
    return {
        "n": len(rows),
        "task_success": round(sum(r["success"] for r in rows) / n, 4),
        "fake_success_rate": round(sum(r["fake_success"] for r in rows) / n, 4),
        "step_recall": round(sum(r["step_recall"] for r in rows) / n, 4),
        "order_rate": round(sum(r["order_ok"] for r in rows) / n, 4),
        "artifact_rate": round(sum(r["artifact_ok"] for r in rows) / n, 4),
        "avg_turns": round(sum(r["turns"] for r in rows) / n, 2),
        "avg_interventions": round(sum(r["interventions"] for r in rows) / n, 2),
        "avg_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n, 1),
        "bad_format_rate": round(sum(1 for r in rows if r["bad_format"]) / n, 4),
    }
