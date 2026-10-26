"""核心数据结构。刻意保持最小：能描述一次运行就够了。"""
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """一次工具调用。result 是工具返回的原始字符串。"""
    name: str
    args: dict = field(default_factory=dict)
    result: str = ""

    def as_dict(self):
        return {"name": self.name, "args": self.args, "result": self.result}


@dataclass
class Trace:
    """一次完整运行的记录。评分只需要这个对象 + 沙盒。"""
    task_id: str = ""
    calls: list = field(default_factory=list)       # list[ToolCall]
    messages: list = field(default_factory=list)    # 原始对话，落盘用
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    interventions: list = field(default_factory=list)  # Skill 每次介入的文本
    finished: bool = False
    bad_format: int = 0          # 模型吐坏 JSON 的次数
    latency_ms: int = 0

    @property
    def names(self):
        return [c.name for c in self.calls]


@dataclass
class Task:
    """一道任务 + 它的验收条件。

    必填：
      id, prompt
    验收（至少给一项，否则无法判分）：
      expected_tools      该调哪些工具（顺序即期望顺序）
      expect_file         沙盒中应出现的文件
      expect_contains     文件必须包含的全部字符串
      expect_any          文件至少包含其一
      expect_shorter_than 产物必须比这个文件短（用于"摘要"类任务）
      expect_min_calls    {工具名: 最少调用次数}，用于"要重查一次"这类要求
      forbidden_tools     出现即判失败的工具（比如故意放的干扰项）
    """
    id: str
    prompt: str
    expected_tools: list = field(default_factory=list)
    expect_file: str | None = None
    expect_contains: list = field(default_factory=list)
    expect_any: list = field(default_factory=list)
    expect_shorter_than: str | None = None
    expect_min_calls: dict = field(default_factory=dict)
    forbidden_tools: list = field(default_factory=list)
    fixtures: dict = field(default_factory=dict)   # 起始文件 {文件名: 内容}
    meta: dict = field(default_factory=dict)       # 想挂什么挂什么（分难度组等）

    @staticmethod
    def from_dict(d: dict) -> "Task":
        known = {f for f in Task.__dataclass_fields__}
        return Task(**{k: v for k, v in d.items() if k in known})

    def artifact_spec(self):
        """用来判断"这个任务到底有没有产物断言"。"""
        return any([self.expect_file, self.expect_contains, self.expect_any,
                    self.expect_shorter_than])
