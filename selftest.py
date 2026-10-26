"""本包自身的自检：不联网、不调模型，验证每个零件。

    python selftest.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_eval import Registry, Sandbox, Task, Tool, score, summarize
from agent_eval.stats import mcnemar_exact, paired_compare, wilson_interval
from agent_eval.tools import impl_read, impl_write
from agent_eval.types import ToolCall, Trace

FAIL = []


def ck(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        FAIL.append(msg)


def mk_trace(names):
    return Trace(task_id="t", calls=[ToolCall(name=n) for n in names], turns=len(names))


def main():
    tmp = Path(tempfile.mkdtemp())
    sb = Sandbox(tmp / "ws")

    # ---- 工具与沙盒
    print("[1] 工具与沙盒")
    reg = Registry([
        Tool("write_file", "写文件", {"type": "object",
                                     "properties": {"path": {}, "content": {}},
                                     "required": ["path", "content"]},
             impl_write, delivery=True),
        Tool("read_file", "读文件", {"type": "object",
                                    "properties": {"path": {}},
                                    "required": ["path"]}, impl_read),
    ])
    out = reg.execute(sb, "write_file", {"path": "a.txt", "content": "hi"})
    ck('"ok": true' in out, "正常写入成功")
    ck(sb.exists("a.txt"), "产物真的落盘")
    ck('tool_not_found' in reg.execute(sb, "nope", {}), "未知工具返回明确错误")
    ck('missing_params' in reg.execute(sb, "write_file", {"path": "b.txt"}),
       "缺必填参数被拒绝")
    try:
        sb.resolve("../逃逸.txt")
        ck(False, "路径越界应被拒绝")
    except ValueError:
        ck(True, "路径越界被拒绝")
    ck(reg.delivery_names() == ["write_file"], "delivery 标记可用")

    # ---- 判分
    print("\n[2] 判分四层")
    task = Task(id="t", prompt="p",
                expected_tools=["read_file", "write_file"],
                expect_file="a.txt", expect_contains=["hi"])
    r = score(mk_trace(["read_file", "write_file"]), task, sb)
    ck(r["success"] and not r["fake_success"], "全对 → success")

    r = score(mk_trace(["read_file"]), task, sb)
    ck(not r["success"] and r["missing"] == ["write_file"], "漏步骤被抓到")
    ck(r["fake_success"], "产物对但漏步骤 → 判为假完成")

    r = score(mk_trace(["write_file", "read_file"]), task, sb)
    ck(not r["order_ok"], "顺序错被判出来")

    r = score(mk_trace(["read_file", "write_file", "read_file"]), task, sb)
    ck(r["success"], "重复调用不影响成败（只记 redundant_calls）")
    ck(r["redundant_calls"] == 1, "重复次数被单独记账")

    r = score(mk_trace(["read_file", "write_file", "hack_tool"]), task, sb)
    ck(r["success"], "多调了未声明的工具不判失败（效率问题）")

    task2 = Task(id="t2", prompt="p", expected_tools=["read_file"],
                 forbidden_tools=["hack_tool"])
    r = score(mk_trace(["read_file", "hack_tool"]), task2, sb)
    ck(not r["success"] and r["forbidden"] == ["hack_tool"], "调了禁区工具判失败")

    task3 = Task(id="t3", prompt="p", expected_tools=["read_file"],
                 expect_min_calls={"read_file": 2})
    ck(not score(mk_trace(["read_file"]), task3, sb)["success"], "次数不足判失败")
    ck(score(mk_trace(["read_file", "read_file"]), task3, sb)["success"],
       "次数达标判成功")

    # ---- 摘要类断言
    print("\n[3] 摘要类断言")
    sb.write("src.md", "这是一个很长的源文件内容" * 10)
    sb.write("sum.md", "这是摘要")
    t4 = Task(id="t4", prompt="p", expected_tools=[],
              expect_file="sum.md", expect_shorter_than="src.md")
    ck(score(mk_trace([]), t4, sb)["artifact_ok"], "摘要比原文短 → 通过")
    sb.write("sum.md", "这是一个很长的源文件内容" * 10)
    ck(not score(mk_trace([]), t4, sb)["artifact_ok"], "不比原文短 → 拒绝")

    # ---- 统计
    print("\n[4] 统计")
    lo, hi = wilson_interval(18, 30)
    ck(lo < 0.6 < hi, f"Wilson 区间包住点估计：{lo:.2f}–{hi:.2f}")
    b, c, n, p = mcnemar_exact([(True, False)] * 6 + [(True, True)] * 10)
    ck(p < 0.05, f"6:0 的分歧应判显著（p={p:.3f}）")
    b, c, n, p = mcnemar_exact([(True, False)] * 1 + [(True, True)] * 20)
    ck(p > 0.05, f"1:0 的分歧应判不显著（p={p:.3f}）")
    b, c, n, p = mcnemar_exact([(True, True)] * 5)
    ck(p == 1.0, "无分歧时 p = 1.0")

    # ---- 汇总
    print("\n[5] 汇总")
    rows = [{"success": True, "fake_success": False, "step_recall": 1.0,
             "order_ok": True, "artifact_ok": True, "turns": 3,
             "interventions": 0, "prompt_tokens": 100, "bad_format": 0},
            {"success": False, "fake_success": True, "step_recall": 0.5,
             "order_ok": False, "artifact_ok": True, "turns": 4,
             "interventions": 1, "prompt_tokens": 200, "bad_format": 0}]
    s = summarize(rows)
    ck(s["task_success"] == 0.5 and s["fake_success_rate"] == 0.5, "汇总正确")

    print("\n" + "=" * 50)
    if FAIL:
        print(f"失败 {len(FAIL)} 项：")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("全部通过。")


if __name__ == "__main__":
    main()
