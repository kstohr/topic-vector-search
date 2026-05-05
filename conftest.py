"""
Root conftest.py — pytest configuration for the workshop.

Run tests against src/ (exercise stubs, default):
    uv run pytest

Run tests against solutions/ (full implementations):
    uv run pytest --solutions

When --solutions is active, sys.modules is patched before test files are
imported so that all `from src.X import ...` statements and all
`monkeypatch.setattr("src.X.Y", ...)` calls transparently target the
solutions module instead.  No test files need to change.
"""

import sys

# Exercise modules that have a solutions/ counterpart.
_EXERCISE_MODULES = [
    "ai_labeler",
    "evaluation",
    "preprocess",
    "run_pipeline",
    "topic_model",
]


def pytest_addoption(parser):
    parser.addoption(
        "--solutions",
        action="store_true",
        default=False,
        help="Run tests against solutions/ implementations instead of src/ stubs.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "exercise: marks tests that exercise the attendee-implemented functions",
    )

    if not config.getoption("--solutions", default=False):
        return

    # Swap src.X → solutions.X before any test file is imported so that
    # `from src.X import ...` and string-based monkeypatches both resolve
    # to the solutions module.
    import src as src_pkg

    for mod_name in _EXERCISE_MODULES:
        src_key = f"src.{mod_name}"
        solutions_key = f"solutions.{mod_name}"

        # Import solutions module, then register it under both keys.
        __import__(solutions_key)
        solutions_mod = sys.modules[solutions_key]
        # Patch sys.modules so `from src.X import ...` gets the solutions module.
        sys.modules[src_key] = solutions_mod
        # Patch the package attribute so `monkeypatch.setattr("src.X.Y", ...)` works.
        setattr(src_pkg, mod_name, solutions_mod)
