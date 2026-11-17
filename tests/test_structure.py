"""Evita reintroducir módulos duplicados fuera del paquete principal."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODULES = {"bootstrap", "core", "evaluation", "report", "segmentation", "ui", "vision"}


class StructureTests(unittest.TestCase):
    def test_modules_only_live_in_package(self):
        for name in sorted(LEGACY_MODULES):
            with self.subTest(module=name):
                self.assertFalse((ROOT / f"{name}.py").exists())
                self.assertTrue((ROOT / "valor_a_la_vista" / f"{name}.py").is_file())

    def test_active_python_files_do_not_import_root_modules(self):
        files = [ROOT / "app.py"]
        for folder in ("valor_a_la_vista", "scripts", "tests"):
            files.extend((ROOT / folder).rglob("*.py"))
        for path in sorted(files):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for name in names:
                    with self.subTest(file=str(path.relative_to(ROOT)), line=node.lineno, module=name):
                        self.assertNotIn(name.split(".")[0], LEGACY_MODULES)


if __name__ == "__main__":
    unittest.main()
