"""Execute notebooks as a reproducibility smoke test.

Usage
-----
Run every notebook::

    python scripts/run_all_notebooks.py

Run a single notebook (used by CI for a fast smoke check)::

    python scripts/run_all_notebooks.py --only 00_project_overview.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(
    notebook_path: Path,
    root: Path,
    timeout: int = 300,
    save: bool = False,
) -> None:
    """Execute a single notebook from the repository root.

    When ``save`` is true the executed notebook (including cell outputs) is
    written back to disk so the rendered outputs are stored in version control.
    """
    print(f"Executing {notebook_path.name}")
    with notebook_path.open(encoding="utf-8") as file:
        notebook = nbformat.read(file, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    client.execute()

    if save:
        with notebook_path.open("w", encoding="utf-8") as file:
            nbformat.write(notebook, file)


def main() -> None:
    """Execute notebooks under the notebooks directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=None,
        help="Execute only the notebook with this file name.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write executed notebooks (with outputs) back to disk.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks"

    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    if args.only is not None:
        notebooks = [path for path in notebooks if path.name == args.only]
        if not notebooks:
            raise SystemExit(f"No notebook named {args.only!r} found in {notebook_dir}.")

    for notebook_path in notebooks:
        execute_notebook(notebook_path, root, save=args.save)


if __name__ == "__main__":
    main()
