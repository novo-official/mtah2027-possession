"""How do long ball-tracking gaps begin and end, and what does each candidate do inside them?

A dead state carried through a ball gap is physically coherent if the ball was last seen outside
the lines (it left play) and is untracked until it re-enters. This counts, label-free and as
aggregates, how the >= 5 s gaps of the graded halves begin / end and how the models label them.

    .venv/bin/python -m experiments.rc1_selection.gaps
"""

import numpy as np

from .diagnostics import D, H, A, runs_of
from solution import data, features, predict
from solution.config import DEFAULT, PRESETS

MIN_GAP = 125   # 5 s


def main():
    blocks = predict.block_ranges()
    tracking = data.load_tracking()
    feats = features.compute_features(tracking, blocks, DEFAULT)
    outside = np.nan_to_num(feats.ball_outside_m.astype(float), nan=0.0)
    gaps = []
    for a, b in blocks:
        for s, e in runs_of(~feats.ball_valid[a:b], a):
            if e - s >= MIN_GAP:
                gaps.append((s, e, a, b))
    n_before = sum(1 for s, e, a, b in gaps if s > a)
    last_out = [outside[s - 1] for s, e, a, b in gaps if s > a]
    first_out = [outside[e] for s, e, a, b in gaps if e < b]
    print("ball gaps >= 5 s in the graded halves: {} ({:.1%} of graded frames, longest {:.0f} s)".format(
        len(gaps), sum(e - s for s, e, _, _ in gaps) / sum(b - a for a, b in blocks), max(e - s for s, e, _, _ in gaps) / 25))
    print("  last seen OUTSIDE the lines (>1 m) before the gap: {}/{}   (>3 m: {})".format(
        sum(x > 1 for x in last_out), len(last_out), sum(x > 3 for x in last_out)))
    print("  first seen OUTSIDE the lines (>1 m) after the gap: {}/{}".format(
        sum(x > 1 for x in first_out), len(first_out)))
    print("  gap length quartiles (s): {}".format(np.round(np.percentile([(e - s) / 25 for s, e, _, _ in gaps], [25, 50, 75]), 1).tolist()))
    for name in ("legacy_p42", "corrected", "corrected_dead", "oa_only"):
        codes, _, ev = predict.predict_codes(tracking, PRESETS[name], "decoder", blocks, feats)
        d_share = [float((codes[s:e] == D).mean()) for s, e, _, _ in gaps]
        mostly_d = [i for i, x in enumerate(d_share) if x >= 0.5]
        starts_out = sum(1 for i in mostly_d if gaps[i][0] > gaps[i][2] and outside[gaps[i][0] - 1] > 1)
        print("{:<15s} gaps >=50% labelled D: {}/{} ({} of them began with the ball outside the lines); frames in gaps: D {:.1%} H/A {:.1%}".format(
            name, len(mostly_d), len(gaps), starts_out,
            sum((codes[s:e] == D).sum() for s, e, _, _ in gaps) / sum(e - s for s, e, _, _ in gaps),
            sum(((codes[s:e] == H) | (codes[s:e] == A)).sum() for s, e, _, _ in gaps) / sum(e - s for s, e, _, _ in gaps)))


if __name__ == "__main__":
    main()
