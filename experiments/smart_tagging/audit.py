"""Phase 4.9 aggregate audit of the organiser's automatic Smart Tagging (no instance is printed).

    .venv/bin/python -m experiments.smart_tagging.audit

Prints counts, distributions and label-free agreement with the tracking-based RC1 evidence. The public
ground truth is NOT touched here. Nothing is tuned from this output except the choice of which codes are
eligible, and that choice is made from semantics and from tracking-only agreement.
"""

import collections
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

import load_smart_tagging as lst  # noqa: E402  (the organisers' parser)
from solution import data, features, predict  # noqa: E402
from solution.config import DEFAULT, PRESETS  # noqa: E402
from solution.evidence import ownership_evidence  # noqa: E402

ATTACK = {"build up", "progression", "creating chances", "attacking transition", "long ball"}
DEFEND = {"defending in defensive third", "defending in middle third", "defending in attacking third", "defensive transition"}
ZONE = {"ball in final third", "players in the box", "ball in the box"}


def norm(x):
    return (x or "").strip().lower()


def team_side(name):
    n = norm(name)
    if n.startswith("barcelona"):
        return 0
    if n.startswith("at") and "club" in n:
        return 1
    return None


def main():
    events = lst.load()
    blocks = predict.block_ranges()
    graded = np.zeros(199319, bool)
    for a, b in blocks:
        graded[a:b] = True
    print("instances:", len(events))
    bad = [e for e in events if not (np.isfinite(e["start_s"]) and np.isfinite(e["end_s"]) and e["end_s"] > e["start_s"])]
    print("invalid/non-positive intervals:", len(bad))
    print("tag groups:", sorted({g for e in events for g in e["tags"]}))
    for g in ("Team", "Half", "Type", "Side"):
        print("  ", g, dict(collections.Counter(e["tags"].get(g) for e in events)))
    rep = lst.check_time_base(events)
    print("time base (official helper): agrees =", rep["agrees"], [r["contains_kickoff"] for r in rep["halves"]])

    print("\ncode: n, median/p90 duration s, team side counts (H / A / N/A), half counts, frames inside graded halves")
    for code in sorted({norm(e["code"]) for e in events}):
        ev = [e for e in events if norm(e["code"]) == code]
        d = np.array([e["end_s"] - e["start_s"] for e in ev])
        t = collections.Counter(team_side(e["tags"].get("Team")) for e in ev)
        ing = sum(1 for e in ev if graded[min(e["start_frame"], 199318)] or graded[min(max(e["end_frame"] - 1, 0), 199318)])
        print("  {:<30s} n={:3d} dur med {:5.1f} p90 {:5.1f} | H {:3d} A {:3d} NA {:3d} | in graded {:3d}".format(
            code, len(ev), np.median(d), np.percentile(d, 90), t[0], t[1], t[None], ing))

    # pairing: attack/defend instances sharing the same interval
    by_interval = collections.defaultdict(list)
    for e in events:
        by_interval[(e["start_frame"], e["end_frame"])].append(e)
    pair_stats = collections.Counter()
    for iv, group in by_interval.items():
        att = [e for e in group if norm(e["code"]) in ATTACK]
        dfn = [e for e in group if norm(e["code"]) in DEFEND]
        if att and dfn:
            sa = {team_side(e["tags"].get("Team")) for e in att}
            sd = {team_side(e["tags"].get("Team")) for e in dfn}
            pair_stats["attack+defend same interval"] += 1
            if len(sa) == 1 and len(sd) == 1 and None not in sa | sd:
                pair_stats["partner team opposite" if sa != sd else "partner team SAME"] += 1
        elif att:
            pair_stats["attack alone"] += 1
        elif dfn:
            pair_stats["defend alone"] += 1
    n_att = sum(1 for e in events if norm(e["code"]) in ATTACK)
    n_def = sum(1 for e in events if norm(e["code"]) in DEFEND)
    print("\nresolvable polarity instances: attack", n_att, "defend", n_def, "| same-interval pairing:", dict(pair_stats))

    # coverage and overlap in graded halves
    def cover(codes, resolved_only=True):
        c = np.zeros(199319, int)
        for e in events:
            if norm(e["code"]) in codes and (not resolved_only or team_side(e["tags"].get("Team")) is not None):
                c[e["start_frame"]:e["end_frame"]] += 1
        return c
    cres = cover(ATTACK | DEFEND)
    print("coverage of graded frames by team-resolved attack/defend intervals: >=1: {:.1%}  >=2: {:.1%}  >=3: {:.1%}".format(
        (cres[graded] >= 1).mean(), (cres[graded] >= 2).mean(), (cres[graded] >= 3).mean()))
    sp = cover({"set pieces"}, False)
    print("coverage of graded frames by SET PIECES intervals: {:.1%}".format((sp[graded] >= 1).mean()))

    # tracking-only agreement (label-free): tag polarity vs RC1 ownership evidence on confident frames
    tracking = data.load_tracking()
    feats = features.compute_features(tracking, blocks, DEFAULT)
    own, w, _ = ownership_evidence(feats, PRESETS["rc1"])
    conf = (w >= 0.6) & graded                     # ball visible, close, controlled: automatic, label-free
    print("\nconfident-tracking frames: {:.1%} of graded".format(conf.mean() * 1.0 * graded.size / graded.sum()))
    print("agreement of tag team with the team the TRACKING names on confident frames (per code; polarity as hypothesised):")
    for code in sorted({norm(e["code"]) for e in events} - {"set pieces", "goals"}):
        agree = tot = 0
        for e in events:
            if norm(e["code"]) != code:
                continue
            s = team_side(e["tags"].get("Team"))
            if s is None:
                continue
            sl = slice(e["start_frame"], e["end_frame"])
            m = conf[sl]
            if not m.any():
                continue
            names_home = own[sl][m] > 0
            agree += int((names_home == (s == 0)).sum())
            tot += int(m.sum())
        print("  {:<30s} frames {:6d}  tag-team == tracking-team: {:.3f}".format(code, tot, agree / tot if tot else float("nan")))

    # new information: how much of the tag coverage lies where tracking has no ownership evidence
    in_tag = cres >= 1
    noev = (w == 0) & graded
    print("\nof graded frames with a team-resolved tag: {:.1%} have NO ownership evidence from tracking (ball absent/unreliable); "
          "of frames without a tag: {:.1%}".format(noev[in_tag & graded].mean(), noev[~in_tag & graded].mean()))
    # set-piece tags near RC1 dead-state starts
    codes_rc1, _, ev = predict.predict_codes(tracking, PRESETS["rc1"], "decoder", blocks, feats)
    d_starts = []
    for a, b in blocks:
        m = (codes_rc1[a:b] == 2).astype(int)
        d_starts += [a + i for i in np.flatnonzero(np.diff(np.concatenate(([0], m))) == 1)]
    print("RC1 dead intervals:", len(d_starts))
    sp_events = [e for e in events if norm(e["code"]) == "set pieces"]
    inside = sum(1 for s in d_starts if any(e["start_frame"] <= s <= e["end_frame"] for e in sp_events))
    print("RC1 dead-interval starts lying inside a SET PIECES interval: {}/{}".format(inside, len(d_starts)))
    # lead: set-piece interval start vs nearest RC1 dead-start / end
    ds = np.array(d_starts)
    leads = [(e["start_frame"] - ds[np.argmin(np.abs(ds - e["start_frame"]))]) / 25 for e in sp_events if graded[min(e["start_frame"], 199318)]]
    print("SET PIECES start minus nearest RC1 dead start (s): quartiles", np.round(np.percentile(leads, [10, 25, 50, 75, 90]), 1).tolist())
    ends = [(e["end_frame"] - ds[np.argmin(np.abs(ds - e["end_frame"]))]) / 25 for e in sp_events if graded[min(e["start_frame"], 199318)]]
    print("SET PIECES end   minus nearest RC1 dead start (s): quartiles", np.round(np.percentile(ends, [10, 25, 50, 75, 90]), 1).tolist())


if __name__ == "__main__":
    main()
