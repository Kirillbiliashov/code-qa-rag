from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class FileChunk(ABC):
    id: str = field(init=False)
    file_path: str
    type: str
    docstring: str

    def __post_init__(self):
        self.id = self.generate_id()

    @abstractmethod
    def generate_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_embedding_text(self) -> str:
        raise NotImplementedError

@dataclass
class ModuleChunk(FileChunk):
    imports: list[str]
    classes: list[str]
    functions: list[str]
    constants: list[str]

    def generate_id(self) -> str:
        return self.file_path

    def to_embedding_text(self) -> str:
        parts = [f"Top-level module {self.file_path}", self.docstring or ""]
        if self.imports:
            parts.append("Imports: " + ", ".join(self.imports))
        if self.classes:
            parts.append("Classes: " + ", ".join(self.classes))
        if self.functions:
            parts.append("Functions: " + ", ".join(self.functions))
        if self.constants:
            parts.append("Constants: " + ", ".join(self.constants))
        return "\n".join(parts).strip()

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "retrieval_text": self.to_embedding_text(),
            "metadata": {
                "symbol": self.file_path,
                "path": self.file_path,
                "parent_class": None,
                "decorators": [],
                "type": self.type,
                "is_async": False,
                "start_line": None,
                "end_line": None
            }
        }
    
@dataclass
class ClassChunk(FileChunk):
    name: str
    code: str
    parent_class: str | None
    decorators: list[str]
    start_line: int
    end_line: int
    methods: list[str]

    def generate_id(self) -> str:
        base = self.file_path
        prefix = f"{base}::{self.parent_class}." if self.parent_class else f"{base}::"
        return f"{prefix}{self.name}"

    def to_embedding_text(self) -> str:
        parts = [f"Class {self.name} in {self.file_path}", self.code, self.docstring or ""]
        return "\n".join([p for p in parts if p]).strip()

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "retrieval_text": self.to_embedding_text(),
            "metadata": {
                "symbol": self.name,
                "path": self.file_path,
                "parent_class": self.parent_class,
                "decorators": self.decorators,
                "type": self.type,
                "is_async": False,
                "start_line": self.start_line,
                "end_line": self.end_line
            }
        }
    
@dataclass
class FunctionChunk(FileChunk):
    name: str
    code: str
    args: list[str]
    returns: str | None
    decorators: list[str]
    start_line: int
    end_line: int
    async_func: bool
    parent_class: str | None

    def generate_id(self) -> str:
        base = self.file_path
        prefix = f"{base}::{self.parent_class}." if self.parent_class else f"{base}::"
        return f"{prefix}{self.name}"

    def to_embedding_text(self) -> str:
        signature = f"def {self.name}({', '.join(self.args)})"
        if self.returns:
            signature += f" -> {self.returns}"
        if self.async_func:
            signature = "async " + signature
        parts = [f"Function {self.name} in {self.file_path}", signature, self.code, self.docstring or ""]
        return "\n".join([p for p in parts if p]).strip()

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "retrieval_text": self.to_embedding_text(),
            "metadata": {
                "symbol": self.name,
                "path": self.file_path,
                "parent_class": self.parent_class,
                "decorators": self.decorators,
                "type": self.type,
                "is_async": self.async_func,
                "start_line": self.start_line,
                "end_line": self.end_line
            }
        }
    
@dataclass
class MethodChunk(FunctionChunk):
    pass