# agent-eval-kit

[![selftest](https://github.com/laotang65666/agent-eval-kit/actions/workflows/selftest.yml/badge.svg)](https://github.com/laotang65666/agent-eval-kit/actions/workflows/selftest.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Deps](https://img.shields.io/badge/dependencies-0-brightgreen)](pyproject.toml)

用**四臂对照**判断"你对 Agent 的改进到底有没有用"，并提供一个被普遍忽略的指标：**假完成（fake_success）**。

零第三方依赖，只用 Python 3.10+ 标准库。

```bash
python examples/run_demo.py     # 不需要 API Key，离线跑出一条四臂结果表
```

---

## 为什么需要这个包

做过 Agent 改进的人大多遇到过这个场面：

> "我把成功率从 60% 提到 63% 了。"
> "多少条任务？"
> "30 条。"
> "……那就是 1 条任务的差别，和抛硬币分不开。"

单臂成功率本身**说明不了任何事**。要判断一个改进是否成立，至少需要三样东西：

1. **对照组**——而且要能排除掉最常见的两种解释
2. **配对比较**——同一批任务逐条比，而不是比较两个均值
3. **过程指标**——只看产物会漏掉一整类失败

这个包把这三件事做成了可直接用的组件。

---

## 三个核心主张

### 主张一：至少四条臂，其中两条是对照

| 臂 | 配置 | 它排除什么 |
|---|---|---|
| **A baseline** | 什么都不加 | —— |
| **B prompt_only** | 把提示词写到最强 | "你是不是只换了提示词" |
| **C treatment** | 你的干预 | 被检验对象 |
| **D upfront** | 运行前就把该做的步骤全告诉它 | "提前告诉它不就完了" |

没有 B 和 D，"C 比 A 好"可以被解释成任何原因。

`Arm` 只提供三个钩子，足以表达绝大多数干预：

```python
Arm(name="C_treatment",
    system_prompt="...",                     # str 或 callable(task) -> str
    post_turn_hook=lambda msgs, at_end: ..., # 返回字符串即注入一条提醒
    max_interventions=3)
```

### 主张二：fake_success —— 产物对、过程错

```python
fake_success = artifact_ok and not success
```

它专门指认这一类运行：**文件写出来了、内容看着也对，但该做的中间步骤被跳过了**。

这类失败有两个特点：

- 不会报错，不会产生空文件，人眼扫一眼产物看不出问题
- 只看成功率会和"压根没写出来"混在一起，看不出差别

实测中，这类失败占全部运行的 3%～7%（取决于任务难度），并且在"模型越强越不容易漏"的直觉下依然稳定出现。

### 主张三：离线重判

判据几乎一定会改。改一次就重跑一遍模型，既贵又让历史数据失去可比性。

```python
from agent_eval.rescore import rescore_runs
rows, changed = rescore_runs("runs/", tasks)
```

只要有 `runs/<tag>/trace.jsonl` 和 `workspace/`，就能重放并重新评分。

---

## 快速开始

```bash
git clone https://github.com/laotang65666/agent-eval-kit
cd agent-eval-kit
python examples/run_demo.py
```

输出示例（离线模拟）：

```
臂               组             成功率      假完成       步骤     干预     轮数
A_baseline      all        75.0%    25.0%    91.7%    0.00    2.50
B_prompt_only   all        75.0%    25.0%    91.7%    0.00    2.50
C_treatment     all       100.0%     0.0%   100.0%    0.25    2.50
D_upfront       all       100.0%     0.0%   100.0%    0.00    2.75
```

读法：

- `B ≈ A` → 只换提示词没有改变行为
- `C > A` 且 `C > B` → 干预本身有效
- `D ≈ C` → **你的干预可能只是"提前告知"这条信息**，没有别的价值
- `fake_success > 0` → 存在"产物对、过程错"的失败

---

## 接你自己的 Agent

### 方式一：用内置循环 + 任意模型

```python
from agent_eval import LoopAgent, Registry, Task, Arm, run_grid

class MyClient:
    def __call__(self, model, messages, tools, temperature=0.0):
        # 调你的厂商，返回 .content / .tool_calls / .prompt_tokens / .completion_tokens
        ...

agent = LoopAgent(client=MyClient(), model="your-model", max_turns=8)
rows = run_grid(agent, registry, tasks, arms, out_dir="runs")
```

### 方式二：接管整个循环

```python
from agent_eval import Trace, ToolCall

class MyAgent:
    def run(self, task, registry, arm, sandbox, system_prompt) -> Trace:
        tr = Trace(task_id=task.id)
        # 你自己怎么跑都行，只要把调用序列填进 tr.calls
        return tr
```

只要有 `Trace`，判分、统计、重判全部可复用。

---

## 任务怎么描述

```python
Task(
    id="t1",
    prompt="查世界模型的论文，整理成表格存到 a.md",
    expected_tools=["search_papers", "make_table", "write_file"],  # 顺序即期望顺序
    expect_file="a.md",
    expect_contains=["A Survey of World Models"],   # 必须全部出现
    expect_any=["摘要", "summary"],                  # 至少出现其一
    expect_shorter_than="draft.md",                  # 摘要类任务：必须比原文短
    expect_min_calls={"search_papers": 2},           # 要求"重查一次"
    forbidden_tools=["search_news"],                 # 出现即判失败（干扰项）
    fixtures={"draft.md": "初始内容"},                # 起始文件
)
```

断言用"能从工具结果里拿到的内容"，不要用"只有标准答案才知道的东西"——否则任务永远失败。

---

## 判据设计上的四个坑（都踩过）

**坑 1：把"重复调用"当"选错工具"。**
模型连搜 4 次同一个工具，是效率问题，不是正确性问题。
→ 本包用**集合**判选择，重复次数单独记在 `redundant_calls`。

**坑 2：不判顺序。**
先 `write_file` 后 `make_table`，集合完全正确，但文件里其实没有表格。
→ 本包用**子序列**判顺序。

**坑 3：精确短语校验制造假失败。**
任务说"改写成精简版"，却要求产物精确出现某个原文短语——模型一改写就丢词，判据却记它失败。
→ 本包提供 `expect_any` 和 `expect_shorter_than`，避免把"合理改写"判死。

**坑 4：判据一改，历史数据全废。**
→ 本包提供离线重判。

---

## 它**不**做什么

- **不提升你的 Agent。** 它只测量。
- **不替代人工判断。** 断言写得差，结果就是垃圾。
- **不提供模型。** 你可以接任何厂商，也可以完全不接（用 `ScriptedAgent` 离线自检）。
- **不保证结论显著。** 样本太小的时候，它会明确告诉你"分不开"。

---

## 目录

```
agent_eval/
├── types.py      Task / Trace / ToolCall
├── tools.py      Tool / Registry / Sandbox（含越界保护、必填参数校验）
├── scorer.py     四层判据 + fake_success
├── stats.py      Wilson 区间 + McNemar 配对检验 + 表格输出
├── runner.py     Arm / run_one / run_grid / report
├── agents.py     LoopAgent（真跑）/ ScriptedAgent（离线）
└── rescore.py    离线重判
examples/
└── run_demo.py   离线四臂演示
```

---

## 一个真实的使用结论

这个包是从一次课程作业里抽出来的。那次实验的结论是：

> 干预让步骤完成度从 93.3% 提到 100%、假完成从 6.7% 降到 0%，
> 但任务成功率只从 60.0% 提到 63.3%，与"开局告知"对照组**零分歧**（p = 1.00）。

**过程指标的改善不等于任务成功率的改善。**
能得出这个结论，比得出一个"提升了 X%"更有价值——而这正是本包想帮你做到的事。

---

## License

MIT © 2026 汤嘉颖
