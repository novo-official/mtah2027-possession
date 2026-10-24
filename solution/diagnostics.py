"""Diagnostics that sit next to, never inside, the official scorer.

The official transition metric (evaluation/possession_metrics.py) pools every label boundary. In
the public window the 20 boundaries are 12 direct home<->away changes and 8 that involve the dead
ball, so a missed dead-ball boundary must not be read as a missed handover. `transition_breakdown`
separates them using the official matching functions unchanged.

Types: "H->A", "A->H" (direct handovers), "alive->D" (a stoppage begins), "D->alive" (play restarts).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

H, A, D = pm.H, pm.A, pm.D
TYPES = ("H->A", "A->H", "alive->D", "D->alive")


def transition_type(from_code, to_code):
    if from_code in (H, A) and to_code in (H, A):
        return "H->A" if from_code == H else "A->H"
    return "alive->D" if to_code == D else "D->alive"


def _filter(transitions, kinds):
    return [t for t in transitions if transition_type(t[1], t[2]) in kinds]


def _prf(matched, n_gt, n_pred):
    precision = matched / n_pred if n_pred else 0.0
    recall = matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"matched": matched, "n_gt": n_gt, "n_pred": n_pred, "precision": precision,
            "recall": recall, "f1": f1}


def transition_breakdown(gt_codes, pred_codes, taus=(12, 25)):
    """Per-type transition diagnostics against a ground truth, using the official matcher.

    Returns {"counts": {type: n_gt}, "direct": {tau: prf}, "dead": {tau: prf},
             "by_type_recall": {tau: {type: (matched, n_gt)}}, "false_handovers": {tau: n},
             "missed_handovers": {tau: n}}.

    `direct` scores only home<->away changes (predicted and true), matching on the full
    (from, to) pair so an A->H cannot match an H->A. `dead` scores only boundaries that involve
    the dead ball, matching on the destination as the official metric does. `by_type_recall` uses
    the official (destination-typed) matching on all boundaries and then attributes each true
    boundary to its type, which is exactly how the pooled number is built.
    """
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gt_trans = pm.transitions(gt_codes, runs)
    pred_trans = pm.transitions(filled, runs)
    counts = {kind: sum(1 for t in gt_trans if transition_type(t[1], t[2]) == kind) for kind in TYPES}

    out = {"counts": counts, "direct": {}, "dead": {}, "by_type_recall": {},
           "false_handovers": {}, "missed_handovers": {}}
    for tau in taus:
        g_direct, p_direct = _filter(gt_trans, ("H->A", "A->H")), _filter(pred_trans, ("H->A", "A->H"))
        pairs, missed, spurious = pm.match_transitions(g_direct, p_direct, tau, typed="pair")
        out["direct"][tau] = _prf(len(pairs), len(g_direct), len(p_direct))
        out["false_handovers"][tau] = len(spurious)
        out["missed_handovers"][tau] = len(missed)

        g_dead, p_dead = _filter(gt_trans, ("alive->D", "D->alive")), _filter(pred_trans, ("alive->D", "D->alive"))
        pairs_d, _, _ = pm.match_transitions(g_dead, p_dead, tau, typed="to")
        out["dead"][tau] = _prf(len(pairs_d), len(g_dead), len(p_dead))

        pairs_all, _, _ = pm.match_transitions(gt_trans, pred_trans, tau, typed="to")
        matched_gt = {id(gt) for gt, _ in pairs_all}
        out["by_type_recall"][tau] = {
            kind: (sum(1 for t in gt_trans if transition_type(t[1], t[2]) == kind and id(t) in matched_gt),
                   counts[kind]) for kind in TYPES}
    return out


def ownership_disagreement(gt_codes, pred_codes):
    """Frame-level team disagreement on frames the ground truth calls alive.

    Returns {"alive_frames", "wrong_team", "called_dead", "abstained"}: of the GT-alive frames, how many
    got the other team, were called dead, or were left X.
    """
    alive = (gt_codes == H) | (gt_codes == A)
    pred = pred_codes[alive]
    gt = gt_codes[alive]
    return {"alive_frames": int(alive.sum()),
            "wrong_team": int((((pred == H) | (pred == A)) & (pred != gt)).sum()),
            "called_dead": int((pred == D).sum()),
            "abstained": int((pred == pm.X).sum())}
