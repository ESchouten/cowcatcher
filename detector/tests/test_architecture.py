import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "src" / "aidetector"


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    return modules


def test_core_dependency_direction():
    allowed_internal = {
        "domain": ("aidetector.domain",),
        "pipeline": ("aidetector.domain", "aidetector.pipeline"),
    }
    allowed_external = {"numpy"}
    violations = []

    for layer, prefixes in allowed_internal.items():
        for path in (PACKAGE / layer).rglob("*.py"):
            for imported in imported_modules(path):
                if imported.startswith("aidetector"):
                    if not any(
                        imported == prefix or imported.startswith(f"{prefix}.")
                        for prefix in prefixes
                    ):
                        violations.append(f"{path.name} imports {imported}")
                    continue

                root = imported.split(".", 1)[0]
                if root not in sys.stdlib_module_names and root not in allowed_external:
                    violations.append(f"{path.name} imports third-party {imported}")

    assert violations == []
