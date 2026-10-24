"""Is a dead state carried through an evidence-free stretch corroborated by independent evidence?

For the `corrected` decoder (no movement evidence in the decoder) every dead frame that has neither
ownership nor dead evidence is held only by the transition costs. This script asks, label-free,
whether the player movement ratio -- a cue the decoder did NOT see in that preset -- looks like a
stoppage inside those stretches, compared with all graded frames. It also reports ball visibility
on the public ground-truth alive / dead frames. Aggregates only.

    .venv/bin/python -m experiments.rc1_selection.persistence
"""

import numpy as np

from . import diagnostics as dg
from .diagnostics import D, W0, W1, pm, public_gt, runs_of
from solution import data, features, predict
from solution.config import DEFAULT, PRESETS
from solution.stoppage import activity_channel


def main():
    blocks = predict.block_ranges()
    tracking = data.load_tracking()
    feats = features.compute_features(tracking, blocks, DEFAULT)
    gt = public_gt()
    ratio = np.full(tracking.n_frames, np.nan)
    for a, b in blocks:
        ratio[a:b] = activity_channel(tracking, a, b, PRESETS["oa_only"])[2]
    observed = np.isfinite(ratio)
    slow = observed & (ratio < PRESETS["oa_only"].relative_ref)
    graded = feats.in_block
    print("movement ratio observed on {:.1%} of graded frames; 'slow' (ratio < {}) on {:.1%} of observed ones".format(
        observed[graded].mean(), PRESETS["oa_only"].relative_ref, slow[graded].sum() / max(observed[graded].sum(), 1)))

    inwin = np.zeros(tracking.n_frames, bool)
    inwin[W0:W1] = True
    vis = feats.ball_valid
    print("public window: ball visible on {:.1%} of GT-alive frames, {:.1%} of GT-dead frames".format(
        vis[inwin & (gt < 2)].mean(), vis[inwin & (gt == 2)].mean()))
    print("public window: movement ratio slow on {:.1%} of GT-alive, {:.1%} of GT-dead frames (observed only)".format(
        slow[inwin & (gt < 2) & observed].sum() / (inwin & (gt < 2) & observed).sum(),
        slow[inwin & (gt == 2) & observed].sum() / (inwin & (gt == 2) & observed).sum()))

    for name in ("legacy_p42", "corrected", "legacy_p43", "corrected_dead", "oa_only"):
        codes, _, ev = predict.predict_codes(tracking, PRESETS[name], "decoder", blocks, feats)
        free = (np.asarray(ev.own_llr) == 0) & (np.asarray(ev.dead_llr) == 0)
        sel_free = (codes == D) & free
        sel_all = codes == D
        sel_ball_gap = sel_all & (feats.gap_length >= 125)
        row = lambda sel: ("{:6.0f} frames, movement observed {:5.1%}, slow among observed {:5.1%}".format(
            sel.sum(), observed[sel].mean() if sel.any() else 0, slow[sel].sum() / max(observed[sel].sum(), 1)))
        print("{:<15s} all D: {} | D in >=5s ball gap: {} | evidence-free D: {}".format(
            name, row(sel_all), row(sel_ball_gap), row(sel_free)))
        # evidence-free D stretches: share whose own median movement ratio is slow
        stretches = runs_of(sel_free)
        long_s = [(a, b) for a, b in stretches if b - a >= 125]
        if long_s:
            med = [np.nanmedian(ratio[a:b]) if np.isfinite(ratio[a:b]).any() else np.nan for a, b in long_s]
            known = [m for m in med if np.isfinite(m)]
            print("                evidence-free D stretches >=5 s: {} (movement known for {}); median ratio < {} in {} of them; "
                  "median-of-medians {:.2f} vs all-graded median {:.2f}".format(
                      len(long_s), len(known), PRESETS["oa_only"].relative_ref,
                      sum(m < PRESETS["oa_only"].relative_ref for m in known),
                      float(np.median(known)) if known else float("nan"), float(np.nanmedian(ratio[graded]))))


if __name__ == "__main__":
    main()
