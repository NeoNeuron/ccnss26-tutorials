"""Derive a student notebook from a solutions notebook.

Strips exercise cells (marked by a leading `# EXERCISE: ...` comment) and
replaces their body with a `raise NotImplementedError("Exercise: ...")` stub.
CUTTABLE cells are also stubbed; CHECKPOINT cells are kept verbatim.
"""
import argparse, re, nbformat as nbf
from pathlib import Path

EXERCISE_RE = re.compile(r"^\s*#\s*EXERCISE:\s*(.+)$", re.MULTILINE)
CUTTABLE_RE = re.compile(r"^\s*#\s*CUTTABLE:?\s*(.+)$", re.MULTILINE)


def stub_cell(source: str, prompt: str) -> str:
    return f"# EXERCISE: {prompt}\n# YOUR CODE HERE\nraise NotImplementedError('{prompt}')\n"


def derive(src: Path, dst: Path):
    nb = nbf.read(src, as_version=4)
    out_cells = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            out_cells.append(cell); continue
        m_ex = EXERCISE_RE.search(cell.source)
        m_ct = CUTTABLE_RE.search(cell.source)
        if m_ex:
            cell.source = stub_cell(cell.source, m_ex.group(1).strip())
            cell.outputs = []
            cell.execution_count = None
        elif m_ct:
            cell.source = stub_cell(cell.source, f"(challenge) {m_ct.group(1).strip()}")
            cell.outputs = []; cell.execution_count = None
        else:
            cell.outputs = []; cell.execution_count = None
        out_cells.append(cell)
    nb.cells = out_cells
    nbf.write(nb, dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src"); p.add_argument("dst")
    a = p.parse_args()
    derive(Path(a.src), Path(a.dst))
