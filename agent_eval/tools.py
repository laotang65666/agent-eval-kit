"""工具与沙盒。

两条硬规则，都是为了"让失败能暴露出来"：

1. **调用不存在的工具必须返回明确错误**，不能静默成功。
2. **缺必填参数必须报错**。真实 API 就是这样；mock 如果比真 API 宽松，
   模型养成的坏习惯在实验里就显不出来，等于人为美化结果。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path


def _ok(**kw) -> str:
    return json.dumps({"ok": True, **kw}, ensure_ascii=False)


def _err(msg: str) -> str:
    return json.dumps({"ok": False, "error": msg}, ensure_ascii=False)


@dataclass
class Tool:
    """一个工具 = 说明书（给模型看）+ 实现（你执行）。

    delivery=True 表示"这是交付动作"（写文件、发消息、发邮件）。
    审计类 Skill 通常需要在交付之前检查前置步骤是否完成。
    """
    name: str
    description: str
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    impl: object = None                 # callable(**args) -> str
    delivery: bool = False              # 是否属于"对外交付"

    @property
    def required(self):
        return self.parameters.get("required", []) or []

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Sandbox:
    """一次运行一个目录。所有文件读写都关在这里面。"""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not str(p).startswith(str(self.root)):
            raise ValueError(f"路径越界：{rel}")
        return p

    def write(self, rel: str, content: str) -> int:
        p = self.resolve(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return len(content.encode("utf-8"))

    def read(self, rel: str) -> str:
        return self.resolve(rel).read_text(encoding="utf-8", errors="ignore")

    def exists(self, rel: str) -> bool:
        try:
            return self.resolve(rel).exists()
        except ValueError:
            return False

    def seed(self, fixtures: dict):
        for name, content in (fixtures or {}).items():
            self.write(name, content)

    def files(self):
        return sorted(p.relative_to(self.root).as_posix()
                      for p in self.root.rglob("*") if p.is_file())


class Registry:
    """工具表 + 执行器。"""

    def __init__(self, tools=None):
        self._tools = {t.name: t for t in (tools or [])}

    def add(self, tool: Tool):
        self._tools[tool.name] = tool
        return self

    def get(self, name):
        return self._tools.get(name)

    def names(self):
        return list(self._tools)

    def delivery_names(self):
        return [n for n, t in self._tools.items() if t.delivery]

    def schemas(self, names=None):
        names = names or list(self._tools)
        return [self._tools[n].schema() for n in names if n in self._tools]

    def execute(self, sandbox: Sandbox, name: str, args: dict) -> str:
        """执行一次调用。永远返回字符串，错误也返回 JSON 字符串。"""
        t = self._tools.get(name)
        if t is None:
            return _err(f"tool_not_found: {name}")
        if not isinstance(args, dict):
            return _err("arguments 必须是 JSON 对象")

        missing = [k for k in t.required if k not in args]
        if missing:
            return _err(f"missing_params: {missing}")

        if t.impl is None:
            return _err(f"tool_not_implemented: {name}")
        try:
            return str(t.impl(sandbox=sandbox, **args))
        except TypeError as e:
            return _err(f"bad_args: {e}")
        except Exception as e:  # noqa: BLE001
            return _err(f"exec_error: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- 常用工具实现
# 提供几个开箱可用的实现，示例里直接用；用户可以完全替换。


def impl_write(sandbox, path, content=""):
    n = sandbox.write(path, content)
    return _ok(path=path, bytes=n)


def impl_read(sandbox, path):
    if not sandbox.exists(path):
        return _err(f"文件不存在：{path}")
    return _ok(path=path, content=sandbox.read(path))


def impl_table(sandbox, headers=None, rows=None):
    headers, rows = headers or [], rows or []
    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "|" + "|".join([" --- "] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(map(str, r)) + " |")
    return _ok(markdown="\n".join(lines))


def impl_email(sandbox, to, subject, body):
    import json as _j
    rec = _j.dumps({"to": to, "subject": subject, "body": body}, ensure_ascii=False)
    p = sandbox.resolve("outbox.jsonl")
    with p.open("a", encoding="utf-8") as f:
        f.write(rec + "\n")
    return _ok(to=to)


def impl_echo(sandbox, **kwargs):
    """万用假工具：把参数原样回显。用于制造"调错工具也能拿到东西"的干扰项。"""
    return _ok(**{k: v for k, v in kwargs.items()}, source="echo")
