"""Optional, bounded team-phase prior from the organisers' automatic Smart Tagging (Phase 4.9, EXPERIMENTAL).

Smart Tagging is inferred, not ground truth: its intervals are tactical phases, not possessions, and they
carry false positives and negatives. A tag is therefore only ever *evidence*, added to the existing
ownership channel with a hard bound; it never writes a label. Specification (frozen before evaluation):
docs/PHASE_4_9_PREREGISTRATION.md.

    eligible pair   an attack-side code and its defend-side partner on the IDENTICAL interval:
                    (BUILD UP, DEFENDING IN ATTACKING THIRD), (PROGRESSION, DEFENDING IN MIDDLE THIRD),
                    (ATTACKING TRANSITION, DEFENSIVE TRANSITION)
    polarity        attack `Team` = team in control; defend `Team` = the DEFENDING team, so the other one
    one vote        a pair is one fact stated twice: disagreement or `N/A` on both sides -> zero
    bounded         each pair spreads a TOTAL of `budget_fraction * switch_cost` nats over its interval, so a long
                    interval cannot accumulate evidence, and the bound (6.0) stays below the hand-over cost (12.0):
                    a tag cannot pay for a switch on its own
    dead neutrality predict_codes adds |prior|/2 to the dead channel, so "favoured team versus dead ball" is exactly
                    as without tags: the prior moves the home/away decision only and cannot shorten a stoppage
    overlap         per frame the largest same-side contribution counts (agreeing pairs are not summed); opposite
                    sides cancel
    zero when       missing/invalid time stamps, interval outside 2..60 s, unknown half or no overlap with its half,
                    `N/A` / unknown team, ineligible code, frame outside the graded halves, no covering tag

Everything is deterministic, symmetric in the teams, and off unless a preset in `TAG_PRESETS` is selected.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields

import numpy as np

import games

ELIGIBLE_PAIRS = (
    ("build up", "defending in attacking third"),
    ("progression", "defending in middle third"),
    ("attacking transition", "defensive transition"),
)


@dataclass(frozen=True)
class TagConfig:
    fps: int = 25
    # Total nats a single pair may contribute, as a fraction of the home<->away switch cost (strictly < 1).
    budget_fraction: float = 0.5
    min_duration_s: float = 2.0
    max_duration_s: float = 60.0
    pairs: tuple = ELIGIBLE_PAIRS
    # Club names as they appear in the `Team` tag, matched case-insensitively. These identify the sides of this
    # fixture (home is listed first by the ATD metadata); they are not labels for any frame.
    home_names: tuple = ("barcelona",)
    away_names: tuple = ("atletic club", "athletic club")

    def budget(self, switch_cost):
        if not 0.0 < self.budget_fraction < 1.0:
            raise ValueError("budget_fraction must be in (0, 1): a tag may never pay a whole switch")
        return self.budget_fraction * switch_cost

    def describe(self):
        return "\n".join("  {:<22s} {}".format(f.name, getattr(self, f.name)) for f in fields(self))


# Opt-in experimental presets; their `Config` is the RC1 one (see config.PRESETS).
TAG_PRESETS = {"rc2_tag_experimental": TagConfig()}


@dataclass(frozen=True)
class Event:
    code: str
    start_frame: int
    end_frame: int
    team: object      # 0 home, 1 away, None unknown / N/A
    half: object      # 0 first, 1 second, None unknown


@dataclass(frozen=True)
class PairVote:
    start: int
    end: int
    half: int
    side: int         # 0 home in control, 1 away
    pair: int


def _norm(text):
    return (text or "").strip().casefold()


def team_side(name, config=TagConfig()):
    n = _norm(name)
    if n in config.home_names:
        return 0
    if n in config.away_names:
        return 1
    return None            # includes "N/A": never guessed


def half_index(text):
    n = _norm(text)
    if n.startswith("1"):
        return 0
    if n.startswith("2"):
        return 1
    return None


def to_frame(seconds, fps=25):
    """Same conversion as the organisers' helper: round(seconds * fps) to the nearest frame."""
    return int(round(seconds * fps))


def parse_events(path, config=TagConfig()):
    """([Event], stats) from a SportsCode XML file. Instances with a missing, non-finite or non-positive interval
    are dropped and counted, never repaired. Raises IOError for a missing file and ValueError for unparsable XML."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError("{}: not valid XML ({})".format(path, error))
    events, stats = [], {"instances": 0, "invalid_interval": 0}
    for instance in root.iter("instance"):
        stats["instances"] += 1
        try:
            start_s, end_s = float(instance.findtext("start")), float(instance.findtext("end"))
        except (TypeError, ValueError):
            stats["invalid_interval"] += 1
            continue
        if not (np.isfinite(start_s) and np.isfinite(end_s)) or end_s <= start_s or start_s < 0:
            stats["invalid_interval"] += 1
            continue
        tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
        events.append(Event(_norm(instance.findtext("code")), to_frame(start_s, config.fps), to_frame(end_s, config.fps),
                            team_side(tags.get("Team"), config), half_index(tags.get("Half"))))
    return events, stats


def build_votes(events, config=TagConfig()):
    """([PairVote], stats): one vote per eligible attack/defend pair on an identical interval and half."""
    votes, stats = [], {"pairs_seen": 0, "unpaired": 0, "conflict_or_unknown": 0, "eligible": 0}
    seen = set()
    for index, (attack_code, defend_code) in enumerate(config.pairs):
        groups = {}
        for e in events:
            if e.code in (attack_code, defend_code):
                groups.setdefault((e.start_frame, e.end_frame, e.half), []).append(e)
        for (start, end, half), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -1 if kv[0][2] is None else kv[0][2])):
            attackers = [e for e in group if e.code == attack_code]
            defenders = [e for e in group if e.code == defend_code]
            if not attackers or not defenders:
                stats["unpaired"] += 1
                continue
            stats["pairs_seen"] += 1
            sides = {e.team for e in attackers if e.team is not None} | {1 - e.team for e in defenders if e.team is not None}
            if len(sides) != 1 or half is None:
                stats["conflict_or_unknown"] += 1
                continue
            key = (index, start, end, half)
            if key in seen:                                 # a duplicated pair is the same fact
                continue
            seen.add(key)
            votes.append(PairVote(start, end, half, sides.pop(), index))
    stats["eligible"] = len(votes)
    return votes, stats


def ownership_prior(votes, n_frames, blocks, budget, config=TagConfig()):
    """(n_frames,) float64 prior on the home-minus-away evidence, > 0 favours home. 0 wherever no vote applies.

    Each vote's density is budget / (its original duration in frames), applied on the part of the interval inside
    the graded block of its half, so its total is at most `budget`. Per frame the largest same-side density counts and
    opposite sides cancel.
    """
    pos = np.zeros(n_frames)
    neg = np.zeros(n_frames)
    dropped = 0
    for v in votes:
        duration = (v.end - v.start) / float(config.fps)
        if not config.min_duration_s <= duration <= config.max_duration_s or v.half >= len(blocks):
            dropped += 1
            continue
        a, b = blocks[v.half]
        lo, hi = max(v.start, a), min(v.end, b)
        if hi <= lo:
            dropped += 1
            continue
        density = budget / float(v.end - v.start)
        target = pos if v.side == 0 else neg
        np.maximum(target[lo:hi], density, out=target[lo:hi])
    return pos - neg, {"dropped": dropped}


def tag_prior(path, n_frames, blocks, switch_cost, config=TagConfig()):
    """(prior, report) from a Smart Tagging file. The one entry point used by predict.py."""
    events, parse_stats = parse_events(path, config)
    votes, vote_stats = build_votes(events, config)
    prior, prior_stats = ownership_prior(votes, n_frames, blocks, config.budget(switch_cost), config)
    report = dict(parse_stats, **vote_stats, **prior_stats)
    report["frames_with_prior"] = int((prior != 0).sum())
    report["max_abs_per_frame"] = float(np.abs(prior).max()) if n_frames else 0.0
    return prior, report


def require_file(data_dir=None):
    """Path of the Smart Tagging XML, with the repository's readable 'gated data' error if it is absent."""
    return games.require(["smart_tagging"], games.GAME, data_dir)[0]
