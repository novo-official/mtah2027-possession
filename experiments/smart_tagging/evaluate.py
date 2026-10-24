"""Phase 4.9 label-free evaluation of the Smart Tagging candidate against RC1 (aggregates only).

    .venv/bin/python -m experiments.smart_tagging.evaluate [--skip-robust]

Nothing here reads the public ground truth. Seeds are fixed (the documented RC1 protocols). No parameter is varied
to improve a number: the "controls" are perturbed COPIES of the pre-registered tags (shifted, flipped, removed) used
only to ask whether the candidate's behaviour is specific to the real tags.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rc1_selection import compare as rc1cmp  # noqa: E402
from experiments.rc1_selection import diagnostics as dg  # noqa: E402
from experiments.rc1_selection.diagnostics import D, H, A, pm, runs_of  # noqa: E402
from solution import data, features, predict, smart_tagging as st  # noqa: E402
from solution.config import DEFAULT, PRESETS  # noqa: E402

CFG = PRESETS["rc2_tag_experimental"]
TAGCFG = st.TAG_PRESETS["rc2_tag_experimental"]
BUDGET = TAGCFG.budget(CFG.switch_cost)
OUT = os.path.join(ROOT, "outputs", "rc2_tag_experimental")
TAG_XML = None


def votes_from_file():
    events, _ = st.parse_events(st.require_file(), TAGCFG)
    return st.build_votes(events, TAGCFG)[0]


def prior_of(votes, n, blocks):
    return st.ownership_prior(votes, n, blocks, BUDGET, TAGCFG)[0]


def shift(votes, frames):
    return [st.PairVote(v.start + frames, v.end + frames, v.half, v.side, v.pair) for v in votes]


def flip(votes, idx=None):
    return [st.PairVote(v.start, v.end, v.half, 1 - v.side if (idx is None or i in idx) else v.side, v.pair) for i, v in enumerate(votes)]


class Ctx:
    def __init__(self):
        self.blocks = predict.block_ranges()
        self.tracking = data.load_tracking()
        self.n = self.tracking.n_frames
        self.feats = features.compute_features(self.tracking, self.blocks, DEFAULT)
        self.votes = votes_from_file()
        self.prior = prior_of(self.votes, self.n, self.blocks)
        self.graded = self.feats.in_block

    def decode(self, prior=None, tracking=None, feats=None):
        t = self.tracking if tracking is None else tracking
        f = self.feats if feats is None else feats
        codes, _, ev = predict.predict_codes(t, CFG, "decoder", self.blocks, f, external_own=prior)
        return codes, ev


def frames_differing(a, b, ctx):
    return int(((a != b) & ctx.graded).sum())


def structure(codes, ev, ctx):
    return dg.fullmatch_diagnostics(codes, ctx.feats, ev, ctx.blocks)


def row(label, f):
    return ("{:<26s} poss {:3d} mean {:5.1f}s med {:5.1f}s <3s {:3d} >60s {:3d} | D ints {:3d} share {:5.2%} | H<->A {:3d} (no ball +-1s {}) "
            "D-chg {:3d} in>=5s gap {} | home {:.1%}").format(
        label, f["possessions"], f["mean_poss_s"], f["median_poss_s"], f["poss_lt3s"], f["poss_gt60s"], f["dead_intervals"], f["dead_share"],
        f["changes_HA"], f["HA_changes_no_ball_pm1s"], f["changes_HD"], f["changes_in_gap5s"], f["home_share_of_alive"])


def diff_segments(a, b, ctx):
    d = (a != b) & ctx.graded
    segs = runs_of(d)
    return segs


def tag_edge_distance(frames, votes):
    edges = np.array(sorted({v.start for v in votes} | {v.end for v in votes}))
    return np.array([np.min(np.abs(edges - f)) for f in frames]) if len(frames) else np.array([])


def transitions_ha(codes, ctx):
    filled = pm.fill_abstentions(codes, ctx.blocks)
    return [t for t in pm.transitions(filled, ctx.blocks) if t[1] < 2 and t[2] < 2], pm.transitions(filled, ctx.blocks)


def full_match(ctx, rc1, cand, ev_rc1, ev_cand, report):
    print("\n== FULL MATCH, LABEL-FREE (behaviour, not accuracy)")
    f1, f2 = structure(rc1, ev_rc1, ctx), structure(cand, ev_cand, ctx)
    print(row("RC1", f1))
    print(row("rc2_tag_experimental", f2))
    for k, (a, b) in enumerate(ctx.blocks):
        s1 = len(pm.possession_spells(pm.fill_abstentions(rc1, [(a, b)]), [(a, b)]))
        s2 = len(pm.possession_spells(pm.fill_abstentions(cand, [(a, b)]), [(a, b)]))
        print("  half {}: possessions RC1 {} -> cand {}; dead share RC1 {:.2%} -> cand {:.2%}".format(
            k + 1, s1, s2, float((rc1[a:b] == D).mean()), float((cand[a:b] == D).mean())))
    n_diff = frames_differing(rc1, cand, ctx)
    segs = diff_segments(rc1, cand, ctx)
    lens = np.array([b - a for a, b in segs]) if segs else np.zeros(0)
    print("frames differing from RC1: {} ({:.3%} of graded) in {} segments (median {:.0f} frames, max {} frames = {:.1f} s)".format(
        n_diff, n_diff / ctx.graded.sum(), len(segs), np.median(lens) if len(lens) else 0, int(lens.max()) if len(lens) else 0, (lens.max() / 25) if len(lens) else 0))
    report["diff_frames"] = n_diff
    report["diff_segments"] = len(segs)
    # what the differing frames are
    both_alive = ((rc1 < 2) & (cand < 2) & ctx.graded & (rc1 != cand)).sum()
    print("  of which H<->A relabelled: {}, involving D: {}".format(int(both_alive), n_diff - int(both_alive)))
    report["diff_HA"] = int(both_alive)
    prior = ctx.prior
    inside_tag = int(((prior != 0) & (rc1 != cand) & ctx.graded).sum())
    print("  differing frames lying inside a tag interval: {} of {}".format(inside_tag, n_diff))
    gapf = int(((ctx.feats.gap_length >= 125) & (rc1 != cand) & ctx.graded).sum())
    print("  differing frames inside ball gaps >= 5 s: {} of {}".format(gapf, n_diff))
    # transitions: new / removed, support
    ha1, all1 = transitions_ha(rc1, ctx)
    ha2, all2 = transitions_ha(cand, ctx)
    f1s = np.array([t[0] for t in all1]); f2s = np.array([t[0] for t in all2])
    new = [t for t in ha2 if not len(f1s) or np.min(np.abs(f1s - t[0])) > 25]
    gone = [t for t in ha1 if not len(f2s) or np.min(np.abs(f2s - t[0])) > 25]
    moved = [t for t in ha2 if len(f1s) and np.min(np.abs(f1s - t[0])) <= 25 and (t[0] not in set(x[0] for x in ha1))]
    print("H<->A changes: RC1 {} cand {}; new (no RC1 change within +-1 s) {}; removed {}; moved by <= 1 s {}".format(len(ha1), len(ha2), len(new), len(gone), len(moved)))
    report.update(ha_rc1=len(ha1), ha_cand=len(ha2), ha_new=len(new), ha_removed=len(gone), ha_moved=len(moved))
    own = np.asarray(ev_rc1.own_llr)
    ok = 0
    tagdom = 0
    for t in new:
        d = 1.0 if t[2] == H else -1.0
        lo, hi = max(0, t[0] - 125), t[0] + 125
        trk, tag = d * own[lo:hi].sum(), d * prior[lo:hi].sum()
        ok += trk > 0
        tagdom += tag > max(trk, 0)
    print("  of the new changes: tracking evidence points the same way in +-5 s for {}; tag mass exceeds tracking mass for {}".format(ok, tagdom))
    report.update(new_tracking_supported=int(ok), new_tag_dominated=int(tagdom))
    # concentration near tag boundaries
    near = lambda ts: int((tag_edge_distance([t[0] for t in ts], ctx.votes) <= 75).sum()) if ts else 0
    edges = np.array(sorted({v.start for v in ctx.votes} | {v.end for v in ctx.votes}))
    cover = np.zeros(ctx.n, bool)
    for e in edges:
        cover[max(0, e - 75): e + 76] = True
    expected = float(cover[ctx.graded].mean())
    print("share of the timeline within +-3 s of an eligible tag boundary: {:.1%} (chance level for a change); H<->A changes there: RC1 {}/{} ({:.1%})  cand {}/{} ({:.1%})  new {}/{}".format(
        expected, near(ha1), len(ha1), near(ha1) / max(len(ha1), 1), near(ha2), len(ha2), near(ha2) / max(len(ha2), 1), near(new), len(new)))
    report.update(expected_edge_share=expected, rc1_near_edge=near(ha1), cand_near_edge=near(ha2), new_near_edge=near(new))


def negative_controls(ctx, rc1, cand, report):
    print("\n== NEGATIVE CONTROLS (frames differing from RC1 / from the candidate; label-free)")
    out = {}
    def go(label, votes):
        c, ev = ctx.decode(prior_of(votes, ctx.n, ctx.blocks))
        f = structure(c, ev, ctx)
        out[label] = dict(vs_rc1=frames_differing(c, rc1, ctx), vs_cand=frames_differing(c, cand, ctx), noball=f["HA_changes_no_ball_pm1s"], gap=f["changes_in_gap5s"], dead=f["dead_share"], poss=f["possessions"])
        print("  {:<28s} differs from RC1 {:6d}  from cand {:6d} | possessions {} dead share {:.3%} | H<->A no ball {} in-gap {}".format(
            label, out[label]["vs_rc1"], out[label]["vs_cand"], f["possessions"], f["dead_share"], f["HA_changes_no_ball_pm1s"], f["changes_in_gap5s"]))
    go("real tags", ctx.votes)
    for k in (30, 90, 180):
        go("shifted +{} s".format(k), shift(ctx.votes, 25 * k))
        go("shifted -{} s".format(k), shift(ctx.votes, -25 * k))
    go("polarity flipped (all)", flip(ctx.votes))
    for seed in range(3):
        rng = np.random.RandomState(900 + seed)
        idx = set(rng.choice(len(ctx.votes), len(ctx.votes) // 5, replace=False).tolist())
        go("20% polarity flipped s{}".format(seed), flip(ctx.votes, idx))
    report["controls"] = out


def removal_tests(ctx, cand, report):
    print("\n== TAG DROPOUT AND INDIVIDUAL DEPENDENCE")
    rows = {}
    for frac in (0.1, 0.3, 0.5):
        diffs, dp = [], []
        for seed in range(5):
            rng = np.random.RandomState(300 + seed)
            keep = rng.rand(len(ctx.votes)) >= frac
            c, ev = ctx.decode(prior_of([v for v, k in zip(ctx.votes, keep) if k], ctx.n, ctx.blocks))
            diffs.append(frames_differing(c, cand, ctx))
            dp.append(100 * (float((c[ctx.graded] == D).mean()) - float((cand[ctx.graded] == D).mean())))
        rows[frac] = (float(np.mean(diffs)), float(np.max(diffs)))
        print("  remove {:.0%} of pairs (5 seeds): frames differing from the candidate mean {:.0f} (max {:.0f}) = {:.3%}; dead share change {:+.3f} pt".format(
            frac, np.mean(diffs), np.max(diffs), np.mean(diffs) / ctx.graded.sum(), np.mean(dp)))
    leave = []
    for i in range(len(ctx.votes)):
        c, _ = ctx.decode(prior_of(ctx.votes[:i] + ctx.votes[i + 1:], ctx.n, ctx.blocks))
        leave.append(frames_differing(c, cand, ctx))
    leave = np.array(leave)
    print("  leave-one-pair-out ({} pairs): frames changed — median {:.0f}, p90 {:.0f}, max {} ({:.1f} s); pairs whose removal changes nothing: {}".format(
        len(leave), np.median(leave), np.percentile(leave, 90), leave.max(), leave.max() / 25, int((leave == 0).sum())))
    report["dropout"] = {str(k): v for k, v in rows.items()}
    report["leave_one_out"] = dict(median=float(np.median(leave)), p90=float(np.percentile(leave, 90)), max=int(leave.max()), zero=int((leave == 0).sum()), n=len(leave))
    # missing tag interval: a contiguous 10 minute block without tags in each half
    for k, (a, b) in enumerate(ctx.blocks):
        mid = a + (b - a) // 2
        kept = [v for v in ctx.votes if not (mid <= v.start < mid + 25 * 600)]
        c, _ = ctx.decode(prior_of(kept, ctx.n, ctx.blocks))
        print("  10-minute tag outage in half {}: {} pairs removed, {} frames differ from the candidate".format(k + 1, len(ctx.votes) - len(kept), frames_differing(c, cand, ctx)))


def masking_test(ctx, rc1, report, fractions=(0.2, 0.4), seeds=5):
    print("\n== INDEPENDENT-VALUE TEST: hide the ball, compare with the UNMASKED RC1 on the hidden frames")
    out = {}
    controls = {"real tags": ctx.votes, "shifted +90 s": shift(ctx.votes, 2250), "polarity flipped": flip(ctx.votes)}
    ref_alive = (rc1 < 2)
    for fraction in fractions:
        res = {k: [] for k in ["RC1"] + list(controls)}
        for seed in range(seeds):
            m = rc1cmp.match_ball_mask(seed, ctx.feats, ctx.blocks, fraction)
            tm = ctx.tracking.with_ball_masked(m)
            fm = features.compute_features(tm, ctx.blocks, DEFAULT)
            sel = m & ctx.graded & ref_alive
            r, _ = ctx.decode(None, tm, fm)
            res["RC1"].append(float((r[sel] == rc1[sel]).mean()))
            for name, votes in controls.items():
                c, _ = ctx.decode(prior_of(votes, ctx.n, ctx.blocks), tm, fm)
                res[name].append(float((c[sel] == rc1[sel]).mean()))
        print("  hidden fraction {:.0%}{}: agreement with the unmasked RC1 on hidden alive frames".format(fraction, "  [pre-registered]" if fraction == 0.2 else "  [supplementary]"))
        for k, v in res.items():
            print("     {:<18s} {} mean {:.4f}".format(k, " ".join("{:.4f}".format(x) for x in v), np.mean(v)))
        for k in controls:
            d = np.array(res[k]) - np.array(res["RC1"])
            print("     paired gain over RC1, {:<17s}: {} mean {:+.4f}  better in {}/{} seeds".format(k, " ".join("{:+.4f}".format(x) for x in d), d.mean(), int((d > 0).sum()), seeds))
        out[str(fraction)] = {k: [float(x) for x in v] for k, v in res.items()}
    report["masking"] = out


def stress(ctx, rc1, cand, report, seeds=5):
    print("\n== PAIRED STRESS TESTS (RC1 and candidate on the same degraded input; agreement with each model's own clean output)")
    gaps = rc1cmp._gap_lengths(ctx.feats, ctx.blocks)
    def one(label, make):
        a, b, na, nb, da, db = [], [], [], [], [], []
        for seed in range(seeds):
            tm = make(seed)
            fm = features.compute_features(tm, ctx.blocks, DEFAULT)
            r, ev1 = ctx.decode(None, tm, fm)
            c, ev2 = ctx.decode(ctx.prior, tm, fm)
            a.append(dg_agree(r, rc1, ctx)); b.append(dg_agree(c, cand, ctx))
            f1, f2 = structure(r, ev1, ctx), structure(c, ev2, ctx)
            na.append(f1["HA_changes_no_ball_pm1s"] + f1["changes_in_gap5s"]); nb.append(f2["HA_changes_no_ball_pm1s"] + f2["changes_in_gap5s"])
            da.append(100 * (f1["dead_share"] - structure(rc1, ev1, ctx)["dead_share"] if False else 100 * 0)); db.append(0)
            report.setdefault("stress", {}).setdefault(label, []).append(dict(rc1_agree=a[-1], cand_agree=b[-1], rc1_unsupported=na[-1], cand_unsupported=nb[-1], rc1_dead=f1["dead_share"], cand_dead=f2["dead_share"], rc1_poss=f1["possessions"], cand_poss=f2["possessions"]))
        s = report["stress"][label]
        print("  {:<22s} agreement RC1 {:.4f} cand {:.4f} | unsupported (H<->A no ball + changes in gaps) RC1 {:.1f} cand {:.1f} | dead share RC1 {:.2%} cand {:.2%} | possessions RC1 {:.0f} cand {:.0f}".format(
            label, np.mean([x["rc1_agree"] for x in s]), np.mean([x["cand_agree"] for x in s]), np.mean([x["rc1_unsupported"] for x in s]), np.mean([x["cand_unsupported"] for x in s]),
            np.mean([x["rc1_dead"] for x in s]), np.mean([x["cand_dead"] for x in s]), np.mean([x["rc1_poss"] for x in s]), np.mean([x["cand_poss"] for x in s])))
    one("ball masking 20%", lambda s: ctx.tracking.with_ball_masked(rc1cmp.match_ball_mask(s, ctx.feats, ctx.blocks, 0.2)))
    one("player dropout 25%", lambda s: rc1cmp.drop_players(ctx.tracking, s, ctx.blocks))
    one("identity jumps", lambda s: identity_jumps(ctx, s))
    # long ball-missing gaps: hide the ball in 30 s blocks until 20% of visible frames are removed
    def long_gaps(s):
        rng = np.random.RandomState(700 + s)
        m = np.zeros(ctx.n, bool)
        tgt = 0.2 * ctx.feats.ball_valid[ctx.graded].sum()
        while (m & ctx.feats.ball_valid & ctx.graded).sum() < tgt:
            a0 = rng.randint(ctx.blocks[0][0], ctx.blocks[-1][1] - 750)
            m[a0:a0 + 750] = True
        return ctx.tracking.with_ball_masked(m)
    one("long ball gaps (30 s)", long_gaps)


def dg_agree(a, b, ctx):
    return float((a[ctx.graded] == b[ctx.graded]).mean())


def identity_jumps(ctx, seed):
    rng = np.random.RandomState(2000 + seed)
    xy = ctx.tracking.player_xy.copy()
    lo, hi = ctx.blocks[0][0], ctx.blocks[-1][1]
    for _ in range(int((hi - lo) / 25 / 60 * 30)):
        slot = rng.randint(ctx.tracking.n_slots)
        s0 = rng.randint(lo - 100, hi)
        length = rng.randint(75, 250)
        ang = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(5, 15)
        xy[max(s0, 0): s0 + length, slot] += np.array([r * np.cos(ang), r * np.sin(ang)], dtype=np.float32)
    return data.Tracking(xy, ctx.tracking.player_speed, ctx.tracking.ball_xy, ctx.tracking.slot_team, ctx.tracking.slot_is_goalkeeper, ctx.tracking.pitch, ctx.tracking.fps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-robust", action="store_true")
    args = parser.parse_args(argv)
    ctx = Ctx()
    report = {}
    print("eligible pairs: {}; frames with a prior: {:.1%} of graded; max |prior| per frame {:.3f} nats; budget {:.1f} nats per pair".format(
        len(ctx.votes), float((ctx.prior[ctx.graded] != 0).mean()), float(np.abs(ctx.prior).max()), BUDGET))
    t0 = time.time()
    rc1, ev1 = ctx.decode(None)
    t1 = time.time()
    cand, ev2 = ctx.decode(ctx.prior)
    t2 = time.time()
    print("decode time RC1 {:.2f}s candidate {:.2f}s".format(t1 - t0, t2 - t1))
    full_match(ctx, rc1, cand, ev1, ev2, report)
    negative_controls(ctx, rc1, cand, report)
    if not args.skip_robust:
        removal_tests(ctx, cand, report)
        masking_test(ctx, rc1, report)
        stress(ctx, rc1, cand, report)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "label_free_report.json"), "w") as handle:
        json.dump(report, handle, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
