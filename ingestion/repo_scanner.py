
import os

class RepositoryScanner:
    IGNORED_NAMES = {'.git', '.gitignore', '.venv', 'venv', '__pycache__', '.DS_Store'}

    @staticmethod
    def scan(project_path):
        """Recursively scan a repository and return a flat list of file paths."""
        if not os.path.isdir(project_path):
            raise ValueError(f"Project path does not exist or is not a directory: {project_path}")

        files = []
        for root, dirnames, filenames in os.walk(project_path):
            dirnames[:] = [d for d in dirnames if d not in RepositoryScanner.IGNORED_NAMES and not d.startswith('.')]
            for filename in filenames:
                if filename in RepositoryScanner.IGNORED_NAMES or filename.startswith('.') or not filename.endswith('.py'):
                    continue
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, project_path)
                files.append(rel_path)
        return files