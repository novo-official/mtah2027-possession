"""Aggregate-only diagnostics used to select the Phase 4.8 release candidate.

Nothing here prints or stores a frame-level record: every function returns counts, shares and
distribution summaries. Public-window metrics use the official scorer unchanged. Full-match
statistics are LABEL-FREE descriptions of a prediction -- they say how it behaves, never how
accurate it is.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from solution import data, diagnostics, features, predict  # noqa: E402
from solution.config import DEFAULT  # noqa: E402

GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25
W0, W1 = 11185, 18685          # the public five-minute window (video frames)
H, A, D, X = 0, 1, 2, 3


def public_gt():
    return pm.label_codes(pm.read_label_csv(GT_PUBLIC))


def runs_of(mask, lo=0):
    """[(start, stop)] of the True-runs of a bool array (indices offset by lo)."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return [(lo + a, lo + b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def dead_runs(codes, blocks):
    out = []
    for a, b in blocks:
        out += runs_of(codes[a:b] == D, a)
    return out


def public_metrics(gt, codes, bootstrap_n=1000):
    """Official metrics plus the transition breakdown, for the public window."""
    r = pm.evaluate(gt, codes, bootstrap_n=bootstrap_n)
    b = diagnostics.transition_breakdown(gt, codes, taus=(12, 25))
    window = codes[W0:W1]
    gw = gt[W0:W1]
    alive = gw < 2
    d = r["per_class"]["D"]
    t12, t25 = r["transitions"][12], r["transitions"][25]
    dead_by_type = b["by_type_recall"][25]
    return {
        "macro_f1": r["macro_f1"], "spell_f1": r["spell_f1_mean"],
        "spell_ci": (r["spell_bootstrap_low"], r["spell_bootstrap_high"]),
        "macro_ci": (r["bootstrap_low"], r["bootstrap_high"]),
        "dead_p": d["precision"], "dead_r": d["recall"], "dead_f1": d["f1"],
        "trans_f1_12": t12["f1"], "trans_f1_25": t25["f1"],
        "direct_matched_25": b["direct"][25]["matched"], "direct_gt": b["direct"][25]["n_gt"],
        "direct_pred_25": b["direct"][25]["n_pred"],
        "dead_entry_25": dead_by_type["alive->D"], "dead_exit_25": dead_by_type["D->alive"],
        "n_pred_spells": r["n_pred_spells"], "n_gt_spells": r["n_gt_spells"],
        "mean_spell_s": r["mean_spell_duration_s"], "gt_mean_spell_s": r["gt_mean_spell_duration_s"],
        "fragmentation": r["fragmentation"],
        "team_acc_gt_alive": r["team_acc_gt_alive"],
        "alive_frame_acc": float(np.mean(np.isin(window[alive], (H, A)) & (window[alive] == gw[alive]))),
        "alive_called_dead": float(np.mean(window[alive] == D)),
    }


def fullmatch_diagnostics(codes, feats, evidence, blocks):
    """Label-free description of a prediction over both graded halves."""
    filled = pm.fill_abstentions(codes, blocks)
    spells = pm.possession_spells(filled, blocks)
    dur = np.array([b - a for a, b, _ in spells], dtype=float) / FPS
    graded = int(sum(b - a for a, b in blocks))
    druns = dead_runs(codes, blocks)
    dlen = np.array([b - a for a, b in druns], dtype=float) / FPS
    in_d = codes == D
    n_d = int(in_d.sum())

    own = np.asarray(evidence.own_llr)
    dead = np.asarray(evidence.dead_llr)
    free = (own == 0) & (dead == 0)                       # emission is exactly zero for all states
    gap = feats.gap_length
    d_free = in_d & free
    free_runs = [b - a for a, b in runs_of(d_free)]
    long_runs = dlen[dlen > 30] if len(dlen) else dlen
    # dead runs that fully contain a ball-missing gap of >= 5 s (the state was carried through it)
    carried = 0
    for a, b in druns:
        if (gap[a:b] >= 125).any():
            carried += 1

    tr = pm.transitions(filled, blocks)
    ha = [t for t in tr if t[1] < 2 and t[2] < 2]
    hd = [t for t in tr if (t[1] == 2) != (t[2] == 2)]
    seen = lambda t, w=25: bool(feats.ball_valid[max(0, t[0] - w): t[0] + w + 1].any())
    alive_codes = ((codes == H) | (codes == A)) & feats.in_block

    out = {
        "possessions": len(spells),
        "mean_poss_s": float(dur.mean()) if len(dur) else 0.0,
        "median_poss_s": float(np.median(dur)) if len(dur) else 0.0,
        "poss_lt3s": int((dur < 3).sum()), "poss_lt5s": int((dur < 5).sum()), "poss_gt60s": int((dur > 60).sum()),
        "dead_intervals": len(druns),
        "dead_share": n_d / graded,
        "dead_share_half": [float((codes[a:b] == D).mean()) for a, b in blocks],
        "dead_dur_p10_50_90": [float(x) for x in np.percentile(dlen, [10, 50, 90])] if len(dlen) else [0, 0, 0],
        "dead_lt3s": int((dlen < 3).sum()), "dead_lt5s": int((dlen < 5).sum()),
        "dead_gt30s": int((dlen > 30).sum()), "dead_gt60s": int((dlen > 60).sum()),
        "dead_longest_s": float(dlen.max()) if len(dlen) else 0.0,
        "dead_frames_in_gt30s_runs_pct": 100.0 * float(long_runs.sum() * FPS) / max(n_d, 1),
        "dead_frames_without_positive_dead_evidence_pct": 100.0 * float((in_d & (dead <= 0)).sum()) / max(n_d, 1),
        "dead_frames_evidence_free_pct": 100.0 * float(d_free.sum()) / max(n_d, 1),
        "dead_evidence_free_longest_s": (max(free_runs) / FPS) if free_runs else 0.0,
        "dead_frames_in_ball_gap5s_pct": 100.0 * float((in_d & (gap >= 125)).sum()) / max(n_d, 1),
        "dead_runs_carrying_ball_gap5s": carried,
        "changes_HA": len(ha), "changes_HD": len(hd),
        "HA_changes_no_ball_pm1s": sum(1 for t in ha if not seen(t)),
        "HD_changes_no_ball_pm1s": sum(1 for t in hd if not seen(t)),
        "changes_in_gap5s": sum(1 for t in tr if gap[t[0]] >= 125),
        "changes_in_any_ball_gap": sum(1 for t in tr if not feats.ball_valid[t[0]]),
        "home_share_of_alive": float((codes[alive_codes] == H).mean()) if alive_codes.any() else float("nan"),
        "abstentions_graded": int(sum((codes[a:b] == X).sum() for a, b in blocks)),
        "frames_outside_blocks_not_X": int(((codes != X) & ~feats.in_block).sum()),
    }
    return out
