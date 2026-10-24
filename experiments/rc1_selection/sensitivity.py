"""Configuration sensitivity of the release candidates (label-free, full match, aggregates only).

Each candidate is perturbed on its own main policy parameters by a fixed, symmetric amount. This is
a stability check, NOT a search: no perturbed configuration is ever adopted.

    .venv/bin/python -m experiments.rc1_selection.sensitivity
"""

import numpy as np

from .diagnostics import D, pm, runs_of
from solution import data, features, predict
from solution.config import DEFAULT, PRESETS

# (preset, [(label, overrides)]) -- +/-25% of the decoder costs for every candidate, plus the one
# parameter that carries each dead-ball model's behaviour.
PLAN = {
    "corrected": [("switch_cost 9", dict(switch_cost=9.0)), ("switch_cost 15", dict(switch_cost=15.0)),
                  ("outside_dead_gain 1.5", dict(outside_dead_gain=1.5)), ("outside_dead_gain 2.5", dict(outside_dead_gain=2.5)),
                  ("enter/exit 7.5/4.5", dict(enter_dead_cost=7.5, exit_dead_cost=4.5)),
                  ("enter/exit 12.5/7.5", dict(enter_dead_cost=12.5, exit_dead_cost=7.5))],
    "corrected_dead": [("switch_cost 9", dict(switch_cost=9.0)), ("switch_cost 15", dict(switch_cost=15.0)),
                       ("relative_ref 0.6", dict(relative_ref=0.6)), ("relative_ref 0.8", dict(relative_ref=0.8)),
                       ("enter/exit 7.5/4.5", dict(enter_dead_cost=7.5, exit_dead_cost=4.5)),
                       ("enter/exit 12.5/7.5", dict(enter_dead_cost=12.5, exit_dead_cost=7.5))],
    "oa_only": [("switch_cost 9", dict(switch_cost=9.0)), ("switch_cost 15", dict(switch_cost=15.0)),
                ("activity_zero 0.2", dict(activity_zero=0.2)), ("activity_zero 0.4", dict(activity_zero=0.4)),
                ("enter/exit 7.5/4.5", dict(enter_dead_cost=7.5, exit_dead_cost=4.5)),
                ("enter/exit 12.5/7.5", dict(enter_dead_cost=12.5, exit_dead_cost=7.5))],
}


def summarize(codes, blocks):
    spells = pm.possession_spells(pm.fill_abstentions(codes, blocks), blocks)
    druns = sum([runs_of(codes[a:b] == D, a) for a, b in blocks], [])
    graded = sum(b - a for a, b in blocks)
    return len(spells), len(druns), sum(int((codes[a:b] == D).sum()) for a, b in blocks) / graded


def main():
    blocks = predict.block_ranges()
    tracking = data.load_tracking()
    feats = features.compute_features(tracking, blocks, DEFAULT)
    n = sum(b - a for a, b in blocks)
    print("{:<16s} {:<22s} {:>10s} {:>8s} {:>8s} {:>9s} {:>9s}".format(
        "preset", "perturbation", "agree", "poss", "D ints", "D share", "d share"))
    for name, perturbations in PLAN.items():
        base, _, _ = predict.predict_codes(tracking, PRESETS[name], "decoder", blocks, feats)
        p0, d0, s0 = summarize(base, blocks)
        print("{:<16s} {:<22s} {:>10s} {:>8d} {:>8d} {:>8.1%} {:>9s}".format(name, "(base)", "-", p0, d0, s0, "-"))
        for label, overrides in perturbations:
            codes, _, _ = predict.predict_codes(tracking, PRESETS[name].replace(**overrides), "decoder", blocks, feats)
            agree = sum(int((codes[a:b] == base[a:b]).sum()) for a, b in blocks) / n
            p, d, s = summarize(codes, blocks)
            print("{:<16s} {:<22s} {:>10.3f} {:>8d} {:>8d} {:>8.1%} {:>+8.1f}pt".format(name, label, agree, p, d, s, 100 * (s - s0)))


if __name__ == "__main__":
    main()
