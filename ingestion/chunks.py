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
            parts.append("Imports:\n" + "\n".join(self.imports))
        if self.constants:
            parts.append("\nConstants:\n " + "\n".join(self.constants))
        if self.classes:
            parts.append("Classes:\n" + "\n".join(self.classes))
        if self.functions:
            parts.append("\nFunctions:\n" + "\n".join(self.functions))
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
    attributes: list[str]

    def generate_id(self) -> str:
        base = self.file_path
        prefix = f"{base}::{self.parent_class}." if self.parent_class else f"{base}::"
        return f"{prefix}{self.name}"

    def to_embedding_text(self) -> str:
        signature = self.code.split(":", 1)[0] + ":"
        parts = [f"Class {self.name} in {self.file_path}", signature, self.docstring or ""]
        if self.attributes:
            parts.append("Attributes:\n" + "\n".join(self.attributes))
        if self.methods:
            parts.append("Methods:\n" + "\n".join(self.methods))
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
    
    def get_signature(self) -> str:
        signature = f"def {self.name}({', '.join(self.args)})"
        if self.returns:
            signature += f" -> {self.returns}"
        if self.async_func:
            signature = "async " + signature
        return signature

    def to_embedding_text(self) -> str:
        parts = [f"Function {self.name} in {self.file_path}", self.code, self.docstring or ""]
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