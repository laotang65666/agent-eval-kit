"""统计：置信区间 + 配对检验 + 表格输出。

写这一块的原因是：**成功率的差值本身说明不了任何事。**

两条臂 63.3% 和 60.0%，看起来差 3 个点，但 30 条任务里那只是 1 条任务的差别，
和抛一次硬币分不开。判断"这个差异是不是真的"，只有两个数有用：
    · 置信区间的宽度
    · 配对检验的 p 值（同一批任务逐条比较）
"""
import math
from math import comb


def wilson_interval(k: int, n: int, z: float = 1.96):
    """二项比例的置信区间。比正态近似稳，小样本/极端比例也能用。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            / (1 + z * z / n))
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(pairs):
    """配对 McNemar 精确检验。

    pairs: 可迭代的 (bool_a, bool_b)，同一道任务在两条臂上的成败。

    返回 (只有 a 成功的数, 只有 b 成功的数, 分歧数, p 值)。
    只统计分歧项 —— 两边都成/都败的样本不提供信息。
    """
    b = c = 0
    for a_ok, b_ok in pairs:
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 0, 1.0
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)
    return b, c, n, p


def paired_compare(rows_a, rows_b, key="task_id", metric="success"):
    """按任务配对比较两批结果。返回 dict，可直接打印。"""
    da = {r[key]: r for r in rows_a}
    db = {r[key]: r for r in rows_b}
    common = sorted(set(da) & set(db))
    pairs = [(bool(da[i][metric]), bool(db[i][metric])) for i in common]
    b, c, n, p = mcnemar_exact(pairs)
    return {
        "n_tasks": len(common),
        "only_a": b, "only_b": c, "discordant": n, "p": round(p, 4),
        "significant": p < 0.05,
        "divergent_tasks": [i for i, (x, y) in zip(common, pairs) if x != y],
    }


def format_table(table, labels=None, title=""):
    """把 {(arm, group): summary} 打成一张对齐的表。"""
    labels = labels or {}
    out = []
    if title:
        out.append(title)
    head = (f"{'臂':<16}{'组':<8}{'成功率':>9}{'假完成':>9}"
            f"{'步骤':>9}{'干预':>7}{'轮数':>7}{'token':>9}")
    out.append(head)
    out.append("-" * len(head))
    for (arm, group), s in table.items():
        out.append(
            f"{labels.get(arm, arm):<16}{str(group):<8}"
            f"{s['task_success']:>8.1%} {s['fake_success_rate']:>8.1%} "
            f"{s['step_recall']:>8.1%} {s['avg_interventions']:>7.2f} "
            f"{s['avg_turns']:>7.2f} {s['avg_prompt_tokens']:>9.0f}")
    return "\n".join(out)


def format_compare(name_a, name_b, res, n_total):
    lines = [f"{name_a} vs {name_b}（配对，n={res['n_tasks']}）",
             f"  只有 {name_a} 成功：{res['only_a']}",
             f"  只有 {name_b} 成功：{res['only_b']}",
             f"  分歧 {res['discordant']} 条，p = {res['p']:.3f} → "
             + ("显著" if res["significant"] else "不显著（与抛硬币分不开）")]
    if n_total and res["n_tasks"]:
        lines.append(f"  提醒：{res['n_tasks']} 条任务里 1 条值 "
                     f"{1 / res['n_tasks']:.1%}，差 1 条通常只是噪声")
    return "\n".join(lines)
