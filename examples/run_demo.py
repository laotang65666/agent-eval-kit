"""离线示例：不需要 API Key，直接跑出一条四臂结果表。

    python examples/run_demo.py

它用 ScriptedAgent 模拟四种"Agent 行为"，演示本包能回答的问题：

  · 我的改进到底有没有真的改变行为？
  · 那个提升是来自干预，还是来自"提前告诉它"？
  · 有没有"产物对、过程错"的假完成被漏掉？

真实使用时把 ScriptedAgent 换成 LoopAgent（接你的模型），
或者实现你自己的 agent.run(...)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_eval import Arm, Registry, Task, run_grid, report
from agent_eval.agents import ScriptedAgent
from agent_eval.stats import format_table, format_compare, paired_compare
from agent_eval.tools import Tool, impl_email, impl_read, impl_table, impl_write


# ---------------------------------------------------------------- 1. 工具

def make_registry():
    return Registry([
        Tool("search_papers",
             "学术检索，返回结构化元数据（标题/年份/DOI）。",
             {"type": "object", "properties": {"query": {"type": "string"}},
              "required": ["query"]},
             lambda sandbox, query: '{"source":"papers","count":2,"results":['
                                    '{"title":"A Survey of World Models"},'
                                    '{"title":"World Models in Robotics"}]}'),
        Tool("read_file", "读取文本文件。",
             {"type": "object", "properties": {"path": {"type": "string"}},
              "required": ["path"]}, impl_read),
        Tool("make_table", "把二维数据渲染成 Markdown 表格。",
             {"type": "object",
              "properties": {"headers": {"type": "array"},
                             "rows": {"type": "array"}},
              "required": ["headers", "rows"]}, impl_table),
        Tool("write_file", "写入文件。", {"type": "object",
                                          "properties": {"path": {"type": "string"},
                                                         "content": {"type": "string"}},
                                          "required": ["path", "content"]},
             impl_write, delivery=True),
        Tool("send_email", "发送邮件。", {"type": "object",
                                          "properties": {"to": {"type": "string"},
                                                         "subject": {"type": "string"},
                                                         "body": {"type": "string"}},
                                          "required": ["to", "subject", "body"]},
             impl_email, delivery=True),
        # 一个干扰工具：能调，但不产出论文元数据
        Tool("search_news", "新闻检索，返回标题与日期，不含 DOI。",
             {"type": "object", "properties": {"topic": {"type": "string"}},
              "required": ["topic"]},
             lambda sandbox, topic: '{"source":"news","results":[{"headline":"..."}]}'),
    ])


# ---------------------------------------------------------------- 2. 任务

TASKS = [
    Task(id="t1", prompt="查世界模型的论文，整理成表格存到 a.md",
         expected_tools=["search_papers", "make_table", "write_file"],
         expect_file="a.md", expect_contains=["A Survey of World Models"],
         forbidden_tools=["search_news"]),
    Task(id="t2", prompt="读 draft.md，摘要后写到 b.md",
         expected_tools=["read_file", "write_file"],
         expect_file="b.md", expect_contains=["项目背景"],
         fixtures={"draft.md": "项目背景：这是一段草稿。结论：需要继续推进。"}),
    Task(id="t3", prompt="查论文，做成表格存到 c.md",
         expected_tools=["search_papers", "make_table", "write_file"],
         expect_file="c.md", expect_contains=["World Models in Robotics"]),
    Task(id="t4", prompt="查论文，整理后发邮件给 boss@example.com",
         expected_tools=["search_papers", "make_table", "send_email"],
         expect_file="outbox.jsonl"),
]


# ---------------------------------------------------------------- 3. 剧本

CONTENT = "| 标题 |\n| --- |\n| A Survey of World Models |\n"

# 基线行为：两条任务会漏步骤
BASE_PLANS = {
    "t1": [("search_papers", {"query": "world models"}),
           ("write_file", {"path": "a.md", "content": CONTENT})],          # 漏 make_table
    "t2": [("read_file", {"path": "draft.md"}),
           ("write_file", {"path": "b.md", "content": "项目背景：草稿摘要。"})],
    "t3": [("search_papers", {"query": "world models"}),
           ("make_table", {"headers": ["标题"], "rows": [["World Models in Robotics"]]}),
           ("write_file", {"path": "c.md", "content": "| 标题 |\n| World Models in Robotics |"})],
    "t4": [("search_papers", {"query": "world models"}),
           ("make_table", {"headers": ["标题"], "rows": [["x"]]}),
           ("send_email", {"to": "boss@example.com", "subject": "汇总", "body": "见表格"})],
}

# 干预被采纳后补做的动作
REPAIR = {
    # 注意：补做完表格后必须**重新写一遍文件**，否则调用序列变成
    # [search, write, make_table] —— 集合对了，顺序错了，仍然判失败。
    # 这个坑在真实项目里踩过，所以示例里特意保留正确做法。
    "t1": [("make_table", {"headers": ["标题"], "rows": [["A Survey of World Models"]]}),
           ("write_file", {"path": "a.md", "content": CONTENT})],
}

# 事前被告知完整步骤时的行为
UPFRONT_PLANS = dict(BASE_PLANS)
UPFRONT_PLANS["t1"] = [("search_papers", {"query": "world models"}),
                       ("make_table", {"headers": ["标题"], "rows": [["A Survey of World Models"]]}),
                       ("write_file", {"path": "a.md", "content": CONTENT})]


def find_missing_rule():
    """一个极简的干预钩子：发现"漏了 make_table 就直接写文件"就提醒。"""

    def hook(messages, at_end=False):
        called = []
        for m in messages:
            for tc in (m.get("tool_calls") or []):
                called.append(tc["function"]["name"])
        if "search_papers" in called and "make_table" not in called and "write_file" in called:
            return "【审计】你在建表之前就把内容写进了文件，文件里的数据没有经过表格工具，请补做。"
        return None

    return hook


# ---------------------------------------------------------------- 4. 跑

def main():
    registry = make_registry()

    arms = [
        Arm("A_baseline", system_prompt="你是一个任务型助理。"),
        Arm("B_prompt_only",
            system_prompt="你是一个任务型助理。必须逐步执行，一步都不能跳过。"),
        Arm("C_treatment",
            system_prompt="你是一个任务型助理。",
            post_turn_hook=find_missing_rule()),
        Arm("D_upfront",
            system_prompt="你是一个任务型助理。步骤：先检索，再建表，最后写文件。"),
    ]

    print("=" * 78)
    print("离线演示：四臂 × 4 条任务（ScriptedAgent 模拟行为，不调用任何模型）")
    print("=" * 78)

    rows = []
    for arm in arms:
        # 每条臂用不同的剧本，模拟"行为真的不同"
        if arm.name == "C_treatment":
            agent = ScriptedAgent(BASE_PLANS, repair=REPAIR, obey_hook=True)
        elif arm.name == "D_upfront":
            agent = ScriptedAgent(UPFRONT_PLANS)
        else:
            agent = ScriptedAgent(BASE_PLANS)
        rows += run_grid(agent, registry, TASKS, [arm], out_dir="demo_runs",
                         verbose=False)
        print(f"  {arm.name} 跑完")

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    table = {k: v for k, v in report(rows).items()}
    print()
    print(format_table(table, labels={a.name: a.name for a in arms}))

    print("\n配对比较（相对基线）：")
    for arm in ["B_prompt_only", "C_treatment", "D_upfront"]:
        res = paired_compare(by_arm["A_baseline"], by_arm[arm])
        print(format_compare("A_baseline", arm, res, len(TASKS)))

    fake = [r for r in rows if r["fake_success"]]
    if fake:
        print(f"\n被 fake_success 挑出来的假完成：{len(fake)} 条")
        for r in fake:
            print(f"  {r['arm']} {r['task_id']}  漏={r['missing']}  "
                  f"产物={r['artifact_reason']}")

    print("\n结论提示：")
    print("  · 若 B ≈ A → 单靠提示词没有改变行为")
    print("  · 若 C > A 且 C > B → 干预本身有效，不是提示词的功劳")
    print("  · 若 D ≈ C → 你的干预价值可能只是'提前告知'这条信息")
    print("  · fake_success > 0 → 存在'产物对、过程错'的失败，只看成功率会漏掉")


if __name__ == "__main__":
    main()
