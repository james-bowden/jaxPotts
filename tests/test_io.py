"""Tests for alphabet encoding and MSA parsers (docs/conventions.md §1-2)."""

from __future__ import annotations

import numpy as np

from jaxpotts import io


def test_alphabet_order_gap_last():
    # Gap is state 20, A is 0 -- matches CCMpred/CCMpredPy (D-001).
    assert io.ALPHABET == "ARNDCQEGHILKMFPSTWYV-"
    assert io.GAP == 20
    assert io.encode_sequence("A")[0] == 0
    assert io.encode_sequence("-")[0] == 20
    assert io.encode_sequence("V")[0] == 19


def test_nonstandard_chars_map_to_gap():
    # B Z J X U O and '.' '*' all collapse to gap (20).
    codes = io.encode_sequence("BZJXUO.*-")
    assert np.all(codes == 20)


def test_alphabet_roundtrip():
    seq = "ARNDCQEGHILKMFPSTWYV-"
    codes = io.encode_sequence(seq)
    assert np.array_equal(codes, np.arange(21))
    assert io.decode_sequence(codes) == seq


def test_fasta_and_aln_agree(tmp_path):
    seqs = ["ARND-", "CQEGH", "IL-KM"]
    aln = tmp_path / "x.aln"
    aln.write_text("\n".join(seqs) + "\n")
    fasta = tmp_path / "x.fasta"
    fasta.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))

    A_aln = io.read_aln(aln)
    A_fasta, headers = io.read_fasta(fasta)
    assert np.array_equal(A_aln, A_fasta)
    assert headers == ["s0", "s1", "s2"]
    assert A_aln.shape == (3, 5)


def test_fasta_multiline_sequence(tmp_path):
    fasta = tmp_path / "wrap.fasta"
    fasta.write_text(">s0\nARND\nCQEG\n>s1\nHILK\nMFPS\n")
    A, headers = io.read_fasta(fasta)
    assert A.shape == (2, 8)
    assert io.decode_sequence(A[0]) == "ARNDCQEG"


def test_ragged_raises(tmp_path):
    aln = tmp_path / "ragged.aln"
    aln.write_text("ARND\nCQE\n")
    import pytest

    with pytest.raises(ValueError):
        io.read_aln(aln)


def test_stockholm_match_states(tmp_path):
    # Match columns are RF='x'; insert columns (RF='.') are dropped, along with the
    # lowercase insert residues. Two sequences, one with an insertion.
    sto = tmp_path / "t.sto"
    sto.write_text(
        "# STOCKHOLM 1.0\n"
        "seq1   AC.DE\n"
        "seq2   ARgKM\n"
        "#=GC RF xx.xx\n"
        "//\n"
    )
    A = io.read_stockholm(sto)
    # match columns are 0,1,3,4 -> seq1 'ACDE', seq2 'ARKM'
    assert A.shape == (2, 4)
    assert io.decode_sequence(A[0]) == "ACDE"
    assert io.decode_sequence(A[1]) == "ARKM"


def test_stockholm_fallback_no_rf(tmp_path):
    # Without an RF line, a match column is one with no lowercase and no '.'.
    sto = tmp_path / "t2.sto"
    sto.write_text("# STOCKHOLM 1.0\nseq1 ACdDE\nseq2 ARkKM\n//\n")
    A = io.read_stockholm(sto)
    assert A.shape == (2, 4)  # column 2 (lowercase insert) dropped
    assert io.decode_sequence(A[0]) == "ACDE"


def test_a3m_match_states(tmp_path):
    # A3M: lowercase letters and '.' are insertions and are dropped; uppercase and '-'
    # are match columns. Two sequences, one with an insertion, become the same length.
    a3m = tmp_path / "x.a3m"
    a3m.write_text(">s1\nACDkE\n>s2\nAR-.M\n")
    A = io.read_a3m(a3m)
    # s1: drop 'k' -> ACDE ; s2: drop '.' -> AR-M
    assert A.shape == (2, 4)
    assert io.decode_sequence(A[0]) == "ACDE"
    assert io.decode_sequence(A[1]) == "AR-M"


def test_a3m_via_read_msa(tmp_path):
    a3m = tmp_path / "y.a3m"
    # s1 has an insert 'a'; s2 lacks it, marked '.' at that insert column (valid a3m).
    a3m.write_text(">s1\nAaCD\n>s2\nA.CD\n")
    A = io.read_msa(a3m)
    assert A.shape == (2, 3)
    assert io.decode_sequence(A[0]) == "ACD"


def test_save_load_couplings_roundtrip(tmp_path):
    import jax.numpy as jnp

    from jaxpotts import energy

    rng = np.random.default_rng(0)
    L, q = 6, 21
    h = rng.normal(size=(L, q)).astype(np.float32)
    W = jnp.asarray(rng.normal(size=(L, q, L, q)))
    J = np.asarray(energy.couplings(W), dtype=np.float32)  # symmetric, zero-diagonal
    p = tmp_path / "cpl.npz"
    io.save_couplings(p, h, J)
    h2, J2 = io.load_couplings(p)
    assert np.allclose(h, h2, atol=1e-6)
    assert np.allclose(J, J2, atol=1e-6)
    # reconstructed J keeps symmetry and zero diagonal
    assert np.allclose(J2, J2.transpose(2, 3, 0, 1), atol=1e-6)
    for i in range(L):
        assert np.allclose(J2[i, :, i, :], 0.0)


def test_one_hot():
    A = np.array([[0, 20, 5]], dtype=np.int8)
    X = io.one_hot(A)
    assert X.shape == (1, 3, 21)
    assert X.sum() == 3
    assert X[0, 0, 0] == 1
    assert X[0, 1, 20] == 1
    assert X[0, 2, 5] == 1


def test_read_reference_1atzA():
    # The shipped reference MSA: 3068 sequences x 75 columns (D-015).
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    aln = root / ".refs/CCMpred/example/1atzA.aln"
    fas = root / ".refs/CCMgen/example/1atzA.fas"
    if not aln.exists() or not fas.exists():
        import pytest

        pytest.skip("reference repos not cloned")
    A_aln = io.read_aln(aln)
    A_fas = io.read_msa(fas)
    assert A_aln.shape == (3068, 75)
    assert A_fas.shape == (3068, 75)
    # Same protein, same alignment -> identical integer encodings.
    assert np.array_equal(A_aln, A_fas)
