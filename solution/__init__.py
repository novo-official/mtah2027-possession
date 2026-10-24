"""Possession inference for More Than A Hack 2027: tracking -> evidence -> decoder (release candidate: preset `rc1`).

Pipeline (see README.md in this folder):

    data.py      ATD -> aligned NumPy arrays in metres, NaN for anything missing or implausible
    features.py  per-frame geometry (nearest player per team, velocities, availability masks)
    evidence.py  soft ownership evidence + a separate dead-ball channel
    decoder.py   three-state Viterbi decoder (H, A, D), halves decoded independently
    predict.py   command line entry point that writes the `frame,label` CSV
    config.py    every tunable number, documented
    model_registry.py  status, features, limitations, fingerprint and reference hash of every named preset
"""

__version__ = "1.0.0rc1"
