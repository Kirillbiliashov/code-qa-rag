

import ast

from ingestion.chunks import ClassChunk, FileChunk, FunctionChunk, MethodChunk, ModuleChunk


class SemanticExtractor(ast.NodeVisitor):
    def __init__(self, source, file_path):
        self.source = source
        self.file_path = file_path
        self.segments = []
        self.chunks: list[FileChunk] = []
        self.class_stack = []
        self.current_class_methods = []
        self.imports = []
        self.classes = []
        self.functions = []
        self.constants = []
        
        
    @property
    def parent_class(self):
        return self.class_stack[-1] if self.class_stack else None
    
    def visit_Module(self, node):
        chunk = ModuleChunk(
            file_path=self.file_path,
            type="module",
            docstring=ast.get_docstring(node),
            imports=[],
            classes=[],
            functions=[],
            constants=[]
        )
        self.generic_visit(node)
        chunk.imports = self.imports
        chunk.classes = self.classes
        chunk.functions = self.functions
        chunk.constants = self.constants
        self.chunks.append(chunk)
        
    def visit_Import(self, node):
        import_code = ast.get_source_segment(self.source, node)
        self.imports.append(import_code)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        self.visit_Import(node)
        
    def visit_Assign(self, node):
        assignment_code = ast.get_source_segment(self.source, node)
        self.constants.append(assignment_code)
        self.generic_visit(node)
        
    def visit_AnnAssign(self, node):
        self.visit_Assign(node)
    
    def visit_ClassDef(self, node):
        segment = ast.get_source_segment(self.source, node)
        self.segments.append(segment)
        chunk = ClassChunk(
            file_path=self.file_path,
            type="class",
            docstring=ast.get_docstring(node),
            name=node.name,
            code=segment,
            parent_class=self.parent_class,
            decorators=[ast.get_source_segment(self.source, d) for d in node.decorator_list],
            start_line=node.lineno,
            end_line=node.end_lineno,
            methods=[]
        )
        self.class_stack.append(node.name)
        self.classes.append(node.name)
        
        self.generic_visit(node)
        chunk.methods = self.current_class_methods
        
        self.current_class_methods = []
        self.class_stack.pop()
        self.chunks.append(chunk)

    def visit_FunctionDef(self, node):
        segment = ast.get_source_segment(self.source, node)
        self.segments.append(segment)
        
        is_method = self.parent_class is not None
        chunk_class = MethodChunk if is_method else FunctionChunk
        
        chunk = chunk_class(
            file_path=self.file_path,
            type="method" if is_method else "function",
            docstring=ast.get_docstring(node),
            name=node.name,
            code=segment,
            args=[arg.arg for arg in node.args.args],
            returns=ast.get_source_segment(self.source, node.returns) if node.returns else None,
            decorators=[ast.get_source_segment(self.source, d) for d in node.decorator_list],
            start_line=node.lineno,
            end_line=node.end_lineno,
            async_func=isinstance(node, ast.AsyncFunctionDef),
            parent_class=self.parent_class
        )
        
        if is_method:
            self.current_class_methods.append(chunk.name)
            
        self.functions.append(chunk.name)
        self.generic_visit(node)
        self.chunks.append(chunk)
        
    def visit_AsyncFunctionDef(self, node): 
        self.visit_FunctionDef(node)