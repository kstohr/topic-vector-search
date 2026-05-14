"""
Reset workshop state: delete Elasticsearch index and all generated output files.

Options:
    --restore-py   Also restore uncommitted changes to .py files (git restore '*.py')

Run:
    uv run python reset.py
    uv run python reset.py --restore-py
"""

import argparse
import logging
import shutil
import subprocess

from elasticsearch import ConnectionError as ESConnectionError
from elasticsearch import ConnectionTimeout

from src.config import ELASTICSEARCH_URL, OUTPUT
from src.es_index import delete_index, get_es_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Files and directories inside output/ to remove.
# __init__.py is kept so the package import still works.
_OUTPUT_GLOBS = ["*.json", "*.csv", "*.html", "bertopic_model/"]
_PY_PATHSPECS = ["*.py", ":(glob)**/*.py"]


def reset_elasticsearch() -> None:
    """Delete the Elasticsearch index if reachable."""
    try:
        client = get_es_client()
        client.info()
        delete_index(client)
    except ESConnectionError as e:
        logger.warning(
            f"Cannot reach Elasticsearch at {ELASTICSEARCH_URL}: {e}.\n"
            f"  Is Docker running? Start the stack with:  docker compose up -d\n"
            f"  Check container status with:               docker ps"
        )
    except ConnectionTimeout as e:
        logger.warning(
            f"Elasticsearch at {ELASTICSEARCH_URL} timed out: {e}. "
            f"The container may still be starting — retry in a few seconds."
        )
    except Exception as e:
        logger.warning(f"Elasticsearch index deletion skipped — {type(e).__name__}: {e}")


def reset_output() -> None:
    """Remove generated files from output/."""
    removed = 0
    for pattern in _OUTPUT_GLOBS:
        for path in OUTPUT.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                logger.info(f"Removed directory: {path.relative_to(OUTPUT.parent)}")
            else:
                path.unlink()
                logger.info(f"Removed file:      {path.relative_to(OUTPUT.parent)}")
            removed += 1
    if removed == 0:
        logger.info("output/ already clean — nothing to remove.")
    else:
        logger.info(f"Removed {removed} item(s) from output/.")


def _git_stdout_lines(args: list[str]) -> list[str]:
    """Run a git command and return non-empty stdout lines."""
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def list_uncommitted_python_files() -> list[str]:
    """Return unstaged, staged, and untracked .py files in the repo."""
    unstaged = _git_stdout_lines(["git", "diff", "--name-only", "--", *_PY_PATHSPECS])
    staged = _git_stdout_lines(["git", "diff", "--cached", "--name-only", "--", *_PY_PATHSPECS])
    untracked = _git_stdout_lines(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *_PY_PATHSPECS]
    )
    return sorted(set(unstaged + staged + untracked))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset workshop state.")
    parser.add_argument(
        "--restore-py",
        action="store_true",
        help="Restore uncommitted changes to .py files via git restore.",
    )
    args = parser.parse_args()

    reset_elasticsearch()
    reset_output()

    if args.restore_py:
        try:
            changed = list_uncommitted_python_files()
            if not changed:
                logger.info("No uncommitted .py changes to restore.")
            else:
                logger.info("Uncommitted .py files that will be restored:")
                for f in changed:
                    logger.info(f"  {f}")
                answer = input("Restore these files? This cannot be undone. [y/N] ").strip().lower()
                if answer == "y":
                    subprocess.run(["git", "restore", "--"] + changed, check=True)
                    logger.info(f"Restored {len(changed)} file(s).")
                else:
                    logger.info("Skipped .py restore.")
        except FileNotFoundError:
            logger.error("Git executable not found. Install Git to use --restore-py.")
        except subprocess.CalledProcessError as error:
            logger.error(f"Git command failed while restoring .py files: {error}")

    logger.info("Reset complete.")
