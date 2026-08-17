"""jaxPotts: Potts-model (21-state MRF) inference over protein MSAs in JAX.

Two objectives share one energy function and one parameter array:

- ``plm`` -- pseudo-likelihood maximisation (reference: CCMpred).
- ``bm``  -- Boltzmann machine learning by persistent contrastive divergence
  (reference: CCMpredPy / CCMgen ``--ofn-cd --persistent``).

See ``docs/conventions.md`` for the numerical conventions.
"""

from __future__ import annotations

from . import bm, energy, gauge, io, optim, plm, profiling, reference, weights
from .energy import Params, conditional_logits, couplings, fields, init_params, sequence_energy
from .io import (
    ALPHABET,
    GAP,
    Q,
    load_couplings,
    one_hot,
    read_a3m,
    read_aln,
    read_fasta,
    read_msa,
    save_couplings,
)
from .weights import n_eff, sequence_weights

__all__ = [
    "io",
    "weights",
    "energy",
    "gauge",
    "plm",
    "bm",
    "optim",
    "reference",
    "profiling",
    "ALPHABET",
    "Q",
    "GAP",
    "one_hot",
    "read_aln",
    "read_fasta",
    "read_msa",
    "read_a3m",
    "save_couplings",
    "load_couplings",
    "sequence_weights",
    "n_eff",
    "Params",
    "init_params",
    "couplings",
    "fields",
    "conditional_logits",
    "sequence_energy",
]

__version__ = "0.1.0"
