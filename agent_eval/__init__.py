"""agent-eval-kit —— 用四臂对照判断"你的 Agent 改进到底有没有用"。

核心提供三样东西：

1. **四臂对照**：基线 / 仅提示词 / 你的干预 / 事前告知（oracle-ish）
   后两臂分别堵死"你是不是只换了提示词"和"提前告诉它不就完了"两个质疑。
2. **假完成指标**（fake_success）：产物造出来了，但过程偷工减料。
   只看成功率会漏掉这一整类失败。
3. **离线重判**：判据改了不用重新调 API，重放 trace 即可。

设计原则：不绑定任何模型、任何框架、任何具体工具。
"""

__version__ = "0.1.0"

from agent_eval.types import Task, ToolCall, Trace
from agent_eval.tools import Registry, Sandbox, Tool
from agent_eval.scorer import check_artifact, score, summarize
from agent_eval.stats import mcnemar_exact, wilson_interval, format_table
from agent_eval.runner import Arm, report, run_grid, run_one, save
from agent_eval.agents import LoopAgent, ScriptedAgent

__all__ = [
    "Task", "ToolCall", "Trace",
    "Tool", "Registry", "Sandbox",
    "score", "summarize", "check_artifact",
    "wilson_interval", "mcnemar_exact", "format_table",
    "Arm", "run_one", "run_grid", "report", "save",
    "LoopAgent", "ScriptedAgent",
    "__version__",
]
