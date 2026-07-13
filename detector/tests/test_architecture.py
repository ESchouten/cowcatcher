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


def test_identity_runtime_does_not_depend_on_application_wiring():
    forbidden = (
        "aidetector.adapters",
        "aidetector.application",
        "aidetector.utils.config",
    )
    violations = []

    excluded = {"enrollment_cli.py", "enrollment_benchmark.py", "video_benchmark.py"}
    for path in (PACKAGE / "dazzlecow").glob("*.py"):
        if path.name in excluded:
            continue
        for imported in imported_modules(path):
            if imported.startswith(forbidden):
                violations.append(f"{path.name} imports {imported}")

    assert violations == []
