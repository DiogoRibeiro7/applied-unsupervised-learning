"""Execute all notebooks as a reproducibility smoke test."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    """Execute every notebook under the notebooks directory."""
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks"

    for notebook_path in sorted(notebook_dir.glob("*.ipynb")):
        print(f"Executing {notebook_path.name}")
        with notebook_path.open(encoding="utf-8") as file:
            notebook = nbformat.read(file, as_version=4)

        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()


if __name__ == "__main__":
    main()
