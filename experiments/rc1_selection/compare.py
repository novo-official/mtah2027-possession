"""Phase 4.8 candidate comparison: public metrics, label-free full-match diagnostics, robustness.

    .venv/bin/python -m experiments.rc1_selection.compare                 # metrics + diagnostics
    .venv/bin/python -m experiments.rc1_selection.compare --robust        # + degradation tests (slow)

Prints aggregates only and writes them to outputs/rc1/selection/*.json (git-ignored). No thresholds
are tuned here: the candidate list is fixed, every configuration is an existing preset.
"""

import argparse
import json
import os
import time

import numpy as np

from . import diagnostics as dg
from .diagnostics import (D, ROOT, W0, W1, fullmatch_diagnostics, pm, public_gt, public_metrics, runs_of)
from solution import data, features, predict
from solution.config import DEFAULT, PRESETS
from solution.evidence import hard_nearest_evidence

CANDIDATES = ["legacy_p42", "corrected", "legacy_p43", "corrected_dead", "oa_only", "oa_restart"]
EXTERNAL = {"baseline_smooth50": "evaluation/results/baseline_smooth50.csv",
            "baseline_smooth125": "evaluation/results/baseline_smooth125.csv"}
OUT = os.path.join(ROOT, "outputs", "rc1", "selection")


def run_candidate(tracking, blocks, feats, config):
    t0 = time.time()
    codes, _, evidence = predict.predict_codes(tracking, config, "decoder", blocks, feats)
    return codes, evidence, time.time() - t0


def baseline_codes(path):
    return pm.label_codes(pm.read_label_csv(os.path.join(ROOT, path)))


def fmt_public(name, m):
    return ("{:<20s} macro {:.3f} [{:.3f}-{:.3f}]  spell {:.3f} [{:.3f}-{:.3f}]  D P/R/F1 {:.2f}/{:.2f}/{:.2f}  "
            "transF1@12/25 {:.2f}/{:.2f}  direct {}/{} (pred {})  Dentry {}/{} Dexit {}/{}  spells {} (gt {}) "
            "mean {:.1f}s (gt {:.1f}s) frag {:.2f}  teamAcc {:.3f}  aliveFrameAcc {:.3f}  alive->D {:.1%}").format(
        name, m["macro_f1"], m["macro_ci"][0] or 0, m["macro_ci"][1] or 0,
        m["spell_f1"], m["spell_ci"][0] or 0, m["spell_ci"][1] or 0, m["dead_p"], m["dead_r"], m["dead_f1"],
        m["trans_f1_12"], m["trans_f1_25"], m["direct_matched_25"], m["direct_gt"], m["direct_pred_25"],
        m["dead_entry_25"][0], m["dead_entry_25"][1], m["dead_exit_25"][0], m["dead_exit_25"][1],
        m["n_pred_spells"], m["n_gt_spells"], m["mean_spell_s"], m["gt_mean_spell_s"], m["fragmentation"],
        m["team_acc_gt_alive"], m["alive_frame_acc"], m["alive_called_dead"])


def fmt_full(name, f):
    return ("{:<20s} poss {:3d} mean {:5.1f}s med {:5.1f}s <3s {:3d} <5s {:3d} >60s {:3d} | D ints {:3d} share {:5.1%} "
            "(h1 {:5.1%} h2 {:5.1%}) dur p10/50/90 {} <3s {} <5s {} >30s {} >60s {} longest {:.0f}s | "
            "changes H<->A {} (no ball +-1s {}) D {} (no ball +-1s {}) in>=5s gap {} | home {:.1%} | X {}").format(
        name, f["possessions"], f["mean_poss_s"], f["median_poss_s"], f["poss_lt3s"], f["poss_lt5s"], f["poss_gt60s"],
        f["dead_intervals"], f["dead_share"], f["dead_share_half"][0], f["dead_share_half"][1],
        [round(x, 1) for x in f["dead_dur_p10_50_90"]], f["dead_lt3s"], f["dead_lt5s"], f["dead_gt30s"],
        f["dead_gt60s"], f["dead_longest_s"], f["changes_HA"], f["HA_changes_no_ball_pm1s"], f["changes_HD"],
        f["HD_changes_no_ball_pm1s"], f["changes_in_gap5s"], f["home_share_of_alive"], f["abstentions_graded"])


def fmt_persistence(name, f):
    return ("{:<20s} D frames: without positive dead evidence {:5.1f}%  evidence-free (carried) {:5.1f}%  in >=5s ball gap "
            "{:5.1f}%  in >30s runs {:5.1f}% | longest evidence-free D stretch {:.0f}s | D runs carrying a >=5s ball gap {}").format(
        name, f["dead_frames_without_positive_dead_evidence_pct"], f["dead_frames_evidence_free_pct"],
        f["dead_frames_in_ball_gap5s_pct"], f["dead_frames_in_gt30s_runs_pct"], f["dead_evidence_free_longest_s"],
        f["dead_runs_carrying_ball_gap5s"])


# ---------------------------------------------------------------------------------------------------
# robustness (protocols as in the Phase 4.3-4.6 analyses; seeds fixed)

def _gap_lengths(feats, blocks):
    gaps = []
    for a, b in blocks:
        gaps += [e - s for s, e in runs_of(~feats.ball_valid[a:b])]
    return np.minimum(np.array(gaps), 1000)


def window_ball_mask(seed, feats, gaps, target):
    """Hide ball frames in realistic gap lengths inside the public window until visibility drops to `target`."""
    rng = np.random.RandomState(seed)
    mask = np.zeros(len(feats.ball_valid), bool)
    for _ in range(200):
        if (feats.ball_valid & ~mask)[W0:W1].mean() <= target:
            break
        length = int(rng.choice(gaps))
        s0 = rng.randint(W0, W1 - 1)
        mask[s0:min(W1, s0 + length)] = True
    return mask


def match_ball_mask(seed, feats, blocks, fraction=0.2):
    """Hide ~fraction of the visible graded ball frames in random 10 s blocks."""
    rng = np.random.RandomState(100 + seed)
    mask = np.zeros(len(feats.ball_valid), bool)
    target = fraction * feats.ball_valid[feats.in_block].sum()
    lo, hi = blocks[0][0], blocks[-1][1] - 250
    while (mask & feats.ball_valid & feats.in_block).sum() < target:
        s0 = rng.randint(lo, hi)
        mask[s0:s0 + 250] = True
    return mask


def drop_players(tracking, seed, blocks):
    """Remove ~25% of every slot's observations in 10 s blocks (tracker dropout)."""
    rng = np.random.RandomState(1000 + seed)
    xy = tracking.player_xy.copy()
    lo, hi = blocks[0][0], blocks[-1][1]
    for slot in range(tracking.n_slots):
        for s0 in rng.randint(lo - 250, hi, size=int((hi - lo) / 250 * 0.25) + 1):
            xy[max(s0, 0): s0 + 250, slot] = np.nan
    return data.Tracking(xy, tracking.player_speed, tracking.ball_xy, tracking.slot_team,
                         tracking.slot_is_goalkeeper, tracking.pitch, tracking.fps)


def agreement(a, b, blocks):
    n = sum(y - x for x, y in blocks)
    return sum(int((a[x:y] == b[x:y]).sum()) for x, y in blocks) / n


def robustness(tracking, blocks, feats, gt, clean, names, seeds=5):
    """Per candidate: label-free full-match stability under extra ball / player dropout, and labelled
    false-stoppage rates when the ball is hidden over public-window alive frames."""
    gaps = _gap_lengths(feats, blocks)
    target = float(np.mean(np.concatenate([feats.ball_valid[a:b] for a, b in blocks])))
    alive_w = np.zeros(len(gt), bool)
    alive_w[W0:W1] = gt[W0:W1] < 2
    dead_w = np.zeros(len(gt), bool)
    dead_w[W0:W1] = gt[W0:W1] == 2
    share = lambda c: float(np.mean(np.concatenate([c[a:b] == D for a, b in blocks])))
    spells = lambda c: len(pm.possession_spells(pm.fill_abstentions(c, blocks), blocks))
    res = {n: {"ball_agree": [], "ball_dposs": [], "ball_dshare": [], "player_agree": [], "player_dposs": [],
               "player_dshare": [], "alive_to_D_visible": [], "alive_to_D_hidden": []} for n in names}
    for seed in range(seeds):
        m = match_ball_mask(seed, feats, blocks)
        tm = tracking.with_ball_masked(m)
        fm = features.compute_features(tm, blocks, DEFAULT)
        tp = drop_players(tracking, seed, blocks)
        fp = features.compute_features(tp, blocks, DEFAULT)
        mw = window_ball_mask(seed, feats, gaps, target)
        tw = tracking.with_ball_masked(mw)
        fw = features.compute_features(tw, blocks, DEFAULT)
        for n in names:
            cfg = PRESETS[n]
            c0 = clean[n]
            c1 = predict.predict_codes(tm, cfg, "decoder", blocks, fm)[0]
            c2 = predict.predict_codes(tp, cfg, "decoder", blocks, fp)[0]
            c3 = predict.predict_codes(tw, cfg, "decoder", blocks, fw)[0]
            r = res[n]
            r["ball_agree"].append(agreement(c0, c1, blocks)); r["ball_dposs"].append(spells(c1) - spells(c0))
            r["ball_dshare"].append(100 * (share(c1) - share(c0)))
            r["player_agree"].append(agreement(c0, c2, blocks)); r["player_dposs"].append(spells(c2) - spells(c0))
            r["player_dshare"].append(100 * (share(c2) - share(c0)))
            sel = alive_w & mw
            if sel.any():
                r["alive_to_D_visible"].append(float((c0[sel] == D).mean()))
                r["alive_to_D_hidden"].append(float((c3[sel] == D).mean()))
    # dead frames with the ball hidden over exactly the true stoppage frames
    tb = tracking.with_ball_masked(dead_w)
    fb = features.compute_features(tb, blocks, DEFAULT)
    for n in names:
        cb = predict.predict_codes(tb, PRESETS[n], "decoder", blocks, fb)[0]
        res[n]["dead_to_D_visible"] = float((clean[n][dead_w] == D).mean())
        res[n]["dead_to_D_hidden"] = float((cb[dead_w] == D).mean())
    return res


def fmt_robust(name, r):
    ms = lambda k: "{:.3f}+/-{:.3f}".format(np.mean(r[k]), np.std(r[k]))
    return ("{:<18s} +20% ball dropout: agree {} dPoss {:+.1f} dD% {:+.2f}pts | -25% player obs: agree {} dPoss {:+.1f} dD% {:+.2f}pts | "
            "public alive->D visible {:.1%} -> ball hidden {:.1%} | public stoppage->D visible {:.1%} -> ball hidden {:.1%}").format(
        name, ms("ball_agree"), np.mean(r["ball_dposs"]), np.mean(r["ball_dshare"]), ms("player_agree"),
        np.mean(r["player_dposs"]), np.mean(r["player_dshare"]), np.mean(r["alive_to_D_visible"]),
        np.mean(r["alive_to_D_hidden"]), r["dead_to_D_visible"], r["dead_to_D_hidden"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robust", action="store_true", help="also run the degradation tests (several minutes)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--candidates", nargs="+", default=CANDIDATES)
    args = parser.parse_args(argv)

    blocks = predict.block_ranges()
    tracking = data.load_tracking()
    feats = features.compute_features(tracking, blocks, DEFAULT)
    gt = public_gt()
    summary = {}
    clean = {}

    print("== PUBLIC FIVE-MINUTE WINDOW (weak diagnostic; reused many times, NOT a validation set)")
    for name, path in EXTERNAL.items():
        codes = baseline_codes(path)
        m = public_metrics(gt, codes, args.bootstrap)
        ev = hard_nearest_evidence(feats, DEFAULT)
        summary[name] = {"public": m, "full": fullmatch_diagnostics(codes, feats, ev, blocks)}
        print(fmt_public(name, m))
    for name in args.candidates:
        codes, evidence, secs = run_candidate(tracking, blocks, feats, PRESETS[name])
        clean[name] = codes
        m = public_metrics(gt, codes, args.bootstrap)
        summary[name] = {"public": m, "full": fullmatch_diagnostics(codes, feats, evidence, blocks), "decode_s": secs}
        print(fmt_public(name, m))

    print("\n== FULL MATCH, LABEL-FREE (behaviour descriptions, NOT accuracy)")
    for name, s in summary.items():
        print(fmt_full(name, s["full"]))
    print("\n== DEAD-STATE PERSISTENCE (full match)")
    for name in args.candidates:
        print(fmt_persistence(name, summary[name]["full"]))

    if args.robust:
        print("\n== ROBUSTNESS (5 paired seeds; label-free except the public-window columns)")
        res = robustness(tracking, blocks, feats, gt, clean, args.candidates)
        for name in args.candidates:
            summary[name]["robust"] = {k: v for k, v in res[name].items()}
            print(fmt_robust(name, res[name]))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "comparison{}.json".format("_robust" if args.robust else ""))
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\nwrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
