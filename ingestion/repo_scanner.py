import os

import pathspec


class RepositoryScanner:
    IGNORED_NAMES = {'.git', '.gitignore', '.venv', 'venv', '__pycache__', '.DS_Store'}

    @staticmethod
    def scan(project_path):
        """Recursively scan a repository and return a flat list of file paths."""
        if not os.path.isdir(project_path):
            raise ValueError(f"Project path does not exist or is not a directory: {project_path}")

        spec = RepositoryScanner._load_gitignore(project_path)

        files = []
        for root, dirnames, filenames in os.walk(project_path):
            kept_dirs = []
            for d in dirnames:
                if d in RepositoryScanner.IGNORED_NAMES or d.startswith('.'):
                    continue
                if spec is not None:
                    rel_dir = os.path.relpath(os.path.join(root, d), project_path)
                    if spec.match_file(rel_dir.replace(os.sep, '/') + '/'):
                        continue
                kept_dirs.append(d)
            dirnames[:] = kept_dirs

            for filename in filenames:
                if filename in RepositoryScanner.IGNORED_NAMES or filename.startswith('.') or not filename.endswith('.py'):
                    continue
                full_path = os.path.join(root, filename)
                if os.path.getsize(full_path) == 0:
                    continue
                rel_path = os.path.relpath(full_path, project_path)
                if spec is not None and spec.match_file(rel_path.replace(os.sep, '/')):
                    continue
                files.append(rel_path)
        return files

    @staticmethod
    def _load_gitignore(project_path: str) -> pathspec.PathSpec | None:
        gi_path = os.path.join(project_path, '.gitignore')
        if not os.path.isfile(gi_path):
            return None
        with open(gi_path, encoding='utf-8', errors='ignore') as f:
            return pathspec.PathSpec.from_lines('gitwildmatch', f)
