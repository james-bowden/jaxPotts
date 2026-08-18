# Tutorial data

`1atzA.aln` — a multiple sequence alignment (3068 sequences × 75 columns) for the
protein domain **1atzA**, in PSICOV `.aln` format (one sequence per line). It is
the example alignment shipped with [CCMpred](https://github.com/soedinglab/CCMpred)
(AGPL-3.0), vendored here so `notebooks/tutorial.ipynb` is runnable out of the box.

`1atzA.pdb` — the reference crystal structure for the 1atzA domain (also shipped
with CCMgen, AGPL-3.0), used to compute ground-truth residue contacts. MSA column
`i` corresponds to PDB residue `i+1`.
