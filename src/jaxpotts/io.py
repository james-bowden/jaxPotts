"""MSA parsing, the amino-acid alphabet, and one-hot encoding.

Conventions (see ``docs/conventions.md`` §1-2):

- ``q = 21`` states in the fixed order ``ARNDCQEGHILKMFPSTWYV-``, so ``A = 0`` and
  the **gap ``-`` is state 20** (matching ``CCMpred/src/sequence.c`` and
  ``CCMgen/ccmpred/io/alignment.py``). Any character that is not one of the 20
  standard amino acids -- including ``B Z J X U O``, ``.`` and ``*`` -- maps to
  the gap state 20.
- Integer MSA ``A`` has shape ``(N, L)`` dtype ``int8``; one-hot ``X`` has shape
  ``(N, L, q)`` dtype ``float32``.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

ALPHABET: str = "ARNDCQEGHILKMFPSTWYV-"
"""Amino-acid alphabet in CCMpred order; index 20 is the gap ``-``."""

Q: int = 21
GAP: int = 20

# char -> index lookup over the full byte range. Everything unmapped defaults to GAP,
# so gap symbols ('-' is in ALPHABET at 20; '.', '*') and unknown letters all land there.
_CHAR_TO_INDEX = np.full(256, GAP, dtype=np.int8)
for _i, _c in enumerate(ALPHABET):
    _CHAR_TO_INDEX[ord(_c)] = _i
    _CHAR_TO_INDEX[ord(_c.lower())] = _i

_INDEX_TO_CHAR = np.frombuffer(ALPHABET.encode("ascii"), dtype=np.uint8)


def encode_sequence(seq: str) -> np.ndarray:
    """Map an amino-acid string to an ``int8`` array of state indices, shape ``(L,)``.

    Unknown/non-standard characters map to the gap state (20), matching CCMpred's
    ``aatoi``.
    """
    b = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return _CHAR_TO_INDEX[b].copy()


def decode_sequence(codes: np.ndarray) -> str:
    """Inverse of :func:`encode_sequence`: state indices ``(L,)`` -> string."""
    codes = np.asarray(codes, dtype=np.intp)
    return _INDEX_TO_CHAR[codes].tobytes().decode("ascii")


def _open(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def read_fasta(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Parse a FASTA MSA into an integer array ``(N, L)`` and the list of headers.

    All sequences must share one aligned length ``L``; a ``ValueError`` is raised
    otherwise. Returns ``(A, headers)`` with ``A`` dtype ``int8``.
    """
    headers: list[str] = []
    seqs: list[str] = []
    cur: list[str] = []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                headers.append(line[1:].strip())
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    if len(seqs) != len(headers):
        raise ValueError(
            f"FASTA parse mismatch: {len(headers)} headers but {len(seqs)} sequences"
        )
    return _stack(seqs, path), headers


def read_aln(path: str | Path) -> np.ndarray:
    """Parse a CCMpred/PSICOV ``.aln`` MSA (one sequence per line, no headers).

    Returns an integer array ``(N, L)`` dtype ``int8``.
    """
    seqs: list[str] = []
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                seqs.append(line)
    return _stack(seqs, path)


def _stack(seqs: list[str], path) -> np.ndarray:
    if not seqs:
        raise ValueError(f"no sequences found in {path}")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"sequences in {path} have differing lengths: {sorted(lengths)}")
    L = lengths.pop()
    A = np.empty((len(seqs), L), dtype=np.int8)
    for n, s in enumerate(seqs):
        A[n] = encode_sequence(s)
    return A


def read_stockholm(path: str | Path) -> np.ndarray:
    """Parse a Stockholm alignment (e.g. a Pfam full alignment) into a match-state
    integer MSA ``(N, L)`` dtype ``int8``.

    Only **match columns** are kept: insert states (lowercase letters and ``.``) are
    dropped, using the ``#=GC RF`` reference-annotation line when present (a column is
    a match column where ``RF`` is not ``.``/``-``), else falling back to "column has
    no lowercase and no ``.``". This is the standard Potts/CCMpred input.
    """
    seqs: dict[str, list[str]] = {}
    order: list[str] = []
    rf: list[str] = []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line == "//":
                continue
            if line.startswith("#=GC RF"):
                rf.append(line.split(None, 2)[2])
                continue
            if line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, seq = parts
            if name not in seqs:
                seqs[name] = []
                order.append(name)
            seqs[name].append(seq)
    if not order:
        raise ValueError(f"no sequences found in {path}")
    full = {name: "".join(chunks) for name, chunks in seqs.items()}
    rf_str = "".join(rf) if rf else None

    L_aln = len(next(iter(full.values())))
    if rf_str is not None and len(rf_str) == L_aln:
        match_cols = [k for k, c in enumerate(rf_str) if c not in ".-"]
    else:
        # Fallback: a match column has no lowercase letter and no '.' across all rows.
        cols = np.array([list(s) for s in full.values()])
        match_cols = [
            k for k in range(L_aln)
            if not any(ch.islower() or ch == "." for ch in cols[:, k])
        ]
    rows = []
    for name in order:
        s = full[name]
        rows.append("".join(s[k] for k in match_cols).upper())
    return _stack(rows, path)


def read_a3m(path: str | Path) -> np.ndarray:
    """Parse an A3M MSA (HHblits/AlphaFold format) into a match-state integer MSA
    ``(N, L)`` dtype ``int8``.

    A3M encodes match columns as uppercase letters / ``-`` and insertions as lowercase
    letters / ``.``. Match states are recovered per sequence by dropping lowercase
    letters and ``.`` (the standard a3m -> aligned-match conversion), which makes every
    row the same match length ``L``.
    """
    headers: list[str] = []
    chunks: list[list[str]] = []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                headers.append(line[1:].strip())
                chunks.append([])
            elif chunks:
                chunks[-1].append(line.strip())
    if not chunks:
        raise ValueError(f"no sequences found in {path}")
    rows = ["".join(c for c in "".join(parts) if not c.islower() and c != ".")
            for parts in chunks]
    return _stack(rows, path)


def read_msa(path: str | Path) -> np.ndarray:
    """Read an MSA, dispatching on extension: ``.aln`` -> PSICOV, ``.sto``/``.stk`` ->
    Stockholm (match states), ``.a3m`` -> A3M (match states), else FASTA. Returns
    integer array ``(N, L)`` int8.
    """
    path = Path(path)
    suffixes = path.suffixes
    stem_suffix = suffixes[0] if suffixes else path.suffix
    if stem_suffix == ".aln":
        return read_aln(path)
    if stem_suffix in (".sto", ".stk", ".stockholm"):
        return read_stockholm(path)
    if stem_suffix == ".a3m":
        return read_a3m(path)
    return read_fasta(path)[0]


def save_couplings(path: str | Path, h: np.ndarray, J: np.ndarray) -> None:
    """Save fields ``h`` ``(L, q)`` and couplings ``J`` ``(L, q, L, q)`` compactly.

    Only the strict upper triangle of ``J`` (the ``i<j`` blocks) is stored, as stacked
    ``(K, q, q)`` blocks plus their ``(i, j)`` indices, in a compressed ``.npz`` -- the
    symmetric lower triangle and the zero diagonal are reconstructed on load. This
    halves the on-disk coupling size (cf. evorca's ``sparse_J.npz``; D-023).
    """
    h = np.asarray(h)
    J = np.asarray(J)
    L, q = h.shape
    iu, ju = np.triu_indices(L, k=1)
    blocks = J[iu, :, ju, :]                       # (K, q, q); block[k] = J[i,:,j,:]
    np.savez_compressed(path, h=h.astype(np.float32), idx_i=iu.astype(np.int32),
                        idx_j=ju.astype(np.int32), blocks=blocks.astype(np.float32),
                        L=np.int64(L), q=np.int64(q))


def load_couplings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`save_couplings`: returns ``(h, J)`` with ``J`` symmetric and
    zero-diagonal, shape ``(L, q, L, q)``.
    """
    d = np.load(path)
    L, q = int(d["L"]), int(d["q"])
    h = d["h"]
    J = np.zeros((L, q, L, q), dtype=np.float32)
    ii, jj, blocks = d["idx_i"], d["idx_j"], d["blocks"]
    J[ii, :, jj, :] = blocks
    J[jj, :, ii, :] = blocks.transpose(0, 2, 1)    # mirror: J[j,b,i,a] = J[i,a,j,b]
    return h, J


def write_fasta(A: np.ndarray, path: str | Path, prefix: str = "seq") -> None:
    """Write an integer MSA ``(N, L)`` as FASTA (headers ``>seq{n}``)."""
    with open(path, "w") as fh:
        for n, row in enumerate(np.asarray(A)):
            fh.write(f">{prefix}{n}\n{decode_sequence(row)}\n")


def write_aln(A: np.ndarray, path: str | Path) -> None:
    """Write an integer MSA ``(N, L)`` as a CCMpred/PSICOV ``.aln`` (one seq per line)."""
    with open(path, "w") as fh:
        for row in np.asarray(A):
            fh.write(decode_sequence(row) + "\n")


def one_hot(A: np.ndarray, dtype=np.float32, num_classes: int = Q) -> np.ndarray:
    """One-hot encode an integer MSA ``(N, L)`` -> ``(N, L, num_classes)``.

    The gap state (20) gets its own channel like any other state. ``num_classes``
    defaults to the full alphabet size (21); small-alphabet test models pass their
    own ``q``.
    """
    A = np.asarray(A)
    X = np.zeros((*A.shape, num_classes), dtype=dtype)
    np.put_along_axis(X, A[..., None].astype(np.intp), 1.0, axis=-1)
    return X
