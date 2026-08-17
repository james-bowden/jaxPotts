"""Thin wrappers to run CCMpred / CCMpredPy and parse their raw coupling output.

Parsers reconstruct ``(h, J)`` in jaxPotts conventions (``docs/decisions.md`` D-009):
``h`` shape ``(L, 21)`` with the gap field 0; ``J`` shape ``(L, 21, L, 21)``,
symmetric, ``J[i,a,j,b]`` = coupling of state ``a`` at site ``i`` with state ``b``
at site ``j``.
"""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import numpy as np

Q = 21


def parse_ccmpred_raw(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse CCMpred's text raw file (``-r``): L lines of 20 single fields, then a
    ``# i j`` header + 21x21 block per pair, where ``block[a, b] = J[i, a, j, b]``.
    """
    lines = Path(path).read_text().splitlines()
    singles: list[list[float]] = []
    idx = 0
    # Single fields: consecutive non-'#' lines at the top (one per column).
    while idx < len(lines) and not lines[idx].startswith("#"):
        parts = lines[idx].split()
        if parts:
            singles.append([float(x) for x in parts])
        idx += 1
    L = len(singles)
    h = np.zeros((L, Q), dtype=np.float64)
    h[:, : Q - 1] = np.array(singles)  # 20 values; gap field stays 0

    J = np.zeros((L, Q, L, Q), dtype=np.float64)
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if not line.startswith("#"):
            continue
        if line.startswith("#>META>"):
            break
        _, i, j = line.split()[:3]
        i, j = int(i), int(j)
        block = np.array([[float(x) for x in lines[idx + a].split()] for a in range(Q)])
        idx += Q
        J[i, :, j, :] = block
        J[j, :, i, :] = block.T
    return h, J


def parse_msgpack_braw(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a CCMpredPy/CCMpred msgpack raw file (``-b``, gzip if ``.gz``).

    Format ``ccm-1``: ``x_single`` flat ``L*20`` and ``x_pair`` a dict keyed
    ``"i/j"`` (upper triangle) of flattened 21x21 blocks ``[a*21+b]`` (D-009).
    Uses our own reader so we do not depend on CCMpredPy's msgpack ``encoding=``
    kwarg (removed in msgpack>=1.0).
    """
    import msgpack

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as fh:
        raw = msgpack.unpackb(fh.read(), raw=False, strict_map_key=False)
    ncol = int(raw["ncol"])
    h = np.zeros((ncol, Q), dtype=np.float64)
    h[:, : Q - 1] = np.array(raw["x_single"], dtype=np.float64).reshape(ncol, Q - 1)
    J = np.zeros((ncol, Q, ncol, Q), dtype=np.float64)
    for entry in raw["x_pair"].values():
        i, j = int(entry["i"]), int(entry["j"])
        block = np.array(entry["x"], dtype=np.float64).reshape(Q, Q)
        J[i, :, j, :] = block
        J[j, :, i, :] = block.T
    return h, J


def read_score_matrix(path: str | Path) -> np.ndarray:
    """Read a whitespace-delimited ``L x L`` score matrix (CCMpred ``.mat`` output)."""
    return np.loadtxt(path)


def run_ccmpred(
    aln: str | Path,
    binary: str | Path,
    out_mat: str | Path,
    out_raw: str | Path | None = None,
    num_iter: int = 250,
    threads: int = 8,
    reweight_threshold: float = 0.8,
    lambda_pair_factor: float = 0.2,
    apc: bool = True,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run the CCMpred (C) binary. Returns the completed process (stdout captured)."""
    cmd = [str(binary), "-t", str(threads), "-n", str(num_iter),
           "-w", str(reweight_threshold), "-l", str(lambda_pair_factor)]
    if not apc:
        cmd.append("-A")
    if out_raw is not None:
        cmd += ["-r", str(out_raw)]
    if extra:
        cmd += extra
    cmd += [str(aln), str(out_mat)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def run_ccmpredpy(
    fasta: str | Path,
    out_braw: str | Path,
    out_mat: str | Path,
    objective: str = "cd",
    persistent: bool = True,
    maxit: int = 500,
    threads: int = 16,
    env: str = "ccmpredpy",
    micromamba: str = "/home/jcbowden/.local/bin/micromamba",
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run CCMpredPy (in its own conda env via ``micromamba run``).

    ``objective='cd'`` uses ``--ofn-cd`` (Boltzmann/PCD, with ``--persistent``);
    anything else uses the default pseudo-likelihood path. Writes the msgpack raw
    couplings (``-b``) and contact matrix (``-m``).
    """
    cmd = [micromamba, "run", "-n", env, "ccmpred", str(fasta),
           "--num-threads", str(threads), "--maxit", str(maxit)]
    if objective == "cd":
        cmd += ["--ofn-cd"]
        if persistent:
            cmd += ["--persistent"]
    if extra:
        cmd += extra
    cmd += ["-b", str(out_braw), "-m", str(out_mat)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)
