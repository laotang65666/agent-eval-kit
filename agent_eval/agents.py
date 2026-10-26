"""两个可直接用的 Agent 实现。

* LoopAgent     —— 真跑：一个最小的"发消息 → 调工具 → 回填"循环。
                   只要给它一个 chat 回调，接任何厂商都行。
* ScriptedAgent —— 离线：按预置剧本回放工具调用。
                   用来在没有 API Key 的情况下验证判分与统计链路，
                   也可以当作"对照组行为"的模拟器。

接自己的框架时，只要实现一个方法即可：

    class MyAgent:
        def run(self, task, registry, arm, sandbox, system_prompt) -> Trace:
            ...
"""
import json
import time

from agent_eval.types import ToolCall, Trace


class LoopAgent:
    """最小 Agent 循环。

    client 需要是一个可调用对象：
        client(model=..., messages=..., tools=...) -> 响应对象
    响应对象要有 .content / .tool_calls / .prompt_tokens / .completion_tokens。
    tool_calls 里每项要有 .id / .name / .args。
    """

    def __init__(self, client, model: str, max_turns: int = 8,
                 temperature: float = 0.0):
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature

    def run(self, task, registry, arm, sandbox, system_prompt):
        messages = [{"role": "system", "content": system_prompt or "你是一个任务型助理。"},
                    {"role": "user", "content": task.prompt}]
        tr = Trace(task_id=task.id)
        t0 = time.time()

        for turn in range(self.max_turns):
            resp = self.client(model=self.model, messages=messages,
                               tools=registry.schemas(),
                               temperature=self.temperature)
            tr.prompt_tokens += getattr(resp, "prompt_tokens", 0) or 0
            tr.completion_tokens += getattr(resp, "completion_tokens", 0) or 0
            tr.turns = turn + 1
            if getattr(resp, "bad_format", False):
                tr.bad_format += 1

            assistant = {"role": "assistant", "content": resp.content or ""}
            if resp.tool_calls:
                assistant["tool_calls"] = [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.name,
                                 "arguments": json.dumps(tc.args, ensure_ascii=False)},
                } for tc in resp.tool_calls]
            messages.append(assistant)

            # —— 模型想收尾：给 Skill 一次介入机会
            if not resp.tool_calls:
                hint = None
                if arm.post_turn_hook and len(tr.interventions) < arm.max_interventions:
                    hint = arm.post_turn_hook(messages, True)
                if hint:
                    messages.append({"role": "system", "content": hint})
                    tr.interventions.append(hint)
                    continue
                tr.finished = True
                break

            for tc in resp.tool_calls:
                out = registry.execute(sandbox, tc.name, tc.args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                tr.calls.append(ToolCall(name=tc.name, args=tc.args, result=out))

            # —— 工具刚执行完：再给一次机会（拦"抢跑交付"）
            if arm.post_turn_hook and len(tr.interventions) < arm.max_interventions:
                hint = arm.post_turn_hook(messages, False)
                if hint:
                    messages.append({"role": "system", "content": hint})
                    tr.interventions.append(hint)

        tr.messages = messages
        tr.latency_ms = int((time.time() - t0) * 1000)
        return tr


class ScriptedAgent:
    """按剧本回放，不需要网络。

    plans  : {task_id: [(tool_name, args), ...]}
    repair : {task_id: [(tool_name, args), ...]}  —— 当 hook 给出提示时追加的补救动作。
             用来在离线环境下演示"干预是否被采纳"这条链路。
    obey_hook: 为 True 时，hook 返回非空就执行 repair。
    """

    def __init__(self, plans, repair=None, obey_hook=True):
        self.plans = plans
        self.repair = repair or {}
        self.obey_hook = obey_hook

    def run(self, task, registry, arm, sandbox, system_prompt):
        tr = Trace(task_id=task.id)
        messages = [{"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": task.prompt}]
        plan = list(self.plans.get(task.id, []))
        repaired = False

        for i, (name, args) in enumerate(plan):
            out = registry.execute(sandbox, name, args)
            tr.calls.append(ToolCall(name=name, args=args, result=out))
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": [{"id": f"c{i}", "type": "function",
                                             "function": {"name": name,
                                                          "arguments": json.dumps(args, ensure_ascii=False)}}]})
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": out})
            tr.turns = i + 1

            if self.obey_hook and arm.post_turn_hook and not repaired:
                hint = arm.post_turn_hook(messages, False)
                if hint:
                    tr.interventions.append(hint)
                    messages.append({"role": "system", "content": hint})
                    for j, (n2, a2) in enumerate(self.repair.get(task.id, [])):
                        o2 = registry.execute(sandbox, n2, a2)
                        tr.calls.append(ToolCall(name=n2, args=a2, result=o2))
                        messages.append({"role": "assistant", "content": "",
                                         "tool_calls": [{"id": f"r{j}", "type": "function",
                                                         "function": {"name": n2,
                                                                      "arguments": json.dumps(a2, ensure_ascii=False)}}]})
                        messages.append({"role": "tool", "tool_call_id": f"r{j}", "content": o2})
                    repaired = True

        # 收尾前再给一次机会
        if self.obey_hook and arm.post_turn_hook and not repaired:
            hint = arm.post_turn_hook(messages, True)
            if hint:
                tr.interventions.append(hint)
                messages.append({"role": "system", "content": hint})
                for j, (n2, a2) in enumerate(self.repair.get(task.id, [])):
                    o2 = registry.execute(sandbox, n2, a2)
                    tr.calls.append(ToolCall(name=n2, args=a2, result=o2))

        messages.append({"role": "assistant", "content": "完成。"})
        tr.messages = messages
        tr.finished = True
        tr.turns = max(tr.turns, 1)
        return tr
