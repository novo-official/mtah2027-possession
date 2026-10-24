"""Stoppage-entry channels for the observation-aware dead-ball model (Phase 4.6).

Two channels, each independently switchable and testable:

**activity** (player-based). The speed of the players, as a ratio to the local live-play level (the
same tempo-relative reference as Phase 4.3, so the two are directly comparable), scored
`tanh((relative_ref - ratio) / relative_scale)`. `oa_activity_stat` chooses the statistic:
"pooled" is the median over all players (Phase 4.3); "both_teams" is the larger of the two teams'
median speeds, which is dead-like only when BOTH teams are slow (one team holding its shape while
the other attacks is not a stoppage). Speeds are strict: a window must be fully observed with no
identity jump. A frame with too few valid speeds has no statistic and the channel is unavailable.

**outside** (ball-based). The ball tracked more than `outside_margin_m` outside the pitch for at
least `oa_outside_min_run` consecutive frames. The persistence requirement rejects jitter at the
line. It is unavailable (reliability 0) while the ball is unobserved: ball absence is never the cue.

A stationary ball is deliberately not a channel (it did not separate stopped from live play) and ball
absence never is.
"""

import numpy as np

from .config import DEFAULT
from .dead_ball import _collective_median, _player_speeds, _signed_score, local_quantile, rolling_median
from .observation import Channel


def _team_medians(speeds, slot_team, min_players):
    out = []
    for team in (0, 1):
        sub = np.where((slot_team == team)[None, :], speeds, np.nan)
        out.append(_collective_median(sub, min_players))
    return out


def activity_channel(tracking, a, b, config=DEFAULT):
    """(score, coverage, ratio) of the movement channel for frames [a, b)."""
    speeds = _player_speeds(tracking, a, b, config)
    if config.oa_activity_stat == "pooled":
        per_frame = _collective_median(speeds, config.min_speed_players)
    elif config.oa_activity_stat == "both_teams":
        home, away = _team_medians(speeds, tracking.slot_team, max(3, config.min_speed_players // 2))
        per_frame = np.fmax(home, away)          # larger median: dead-like only if both teams are slow
    else:
        raise ValueError("oa_activity_stat must be 'pooled' or 'both_teams'")
    agg, coverage = rolling_median(per_frame, config.dead_window_halfwidth, config.dead_min_coverage)
    level = local_quantile(agg, config.dead_level_halfwidth, config.dead_level_quantile,
                           config.dead_level_min_frames)
    with np.errstate(all="ignore"):
        ratio = agg / level
    score = _signed_score(ratio, config.relative_ref, config.relative_scale, config)
    return score, coverage, ratio


def _run_lengths(flag):
    """Length of the True-run each True element belongs to (0 elsewhere)."""
    out = np.zeros(len(flag), dtype=np.int32)
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[s:e] = e - s
    return out


def outside_channel(features, a, b, config=DEFAULT):
    """(score, reliability) of the out-of-play channel for frames [a, b)."""
    outside = np.nan_to_num(features.ball_outside_m[a:b].astype(np.float64), nan=0.0) > config.outside_margin_m
    outside &= features.ball_valid[a:b]
    persistent = outside & (_run_lengths(outside) >= config.oa_outside_min_run)
    score = np.where(features.ball_valid[a:b], np.where(persistent, 1.0, 0.0), np.nan)
    return score, features.ball_valid[a:b].astype(np.float64)


def stoppage_channels(tracking, features, blocks, config=DEFAULT):
    """({name: Channel}, ratio) over the whole recording; frames outside `blocks` are unavailable.

    `ratio` is the movement ratio (activity relative to the local live level), kept for the restart
    detector; NaN where unavailable.
    """
    n = tracking.n_frames
    nan = lambda: np.full(n, np.nan)
    act_score, act_rel, ratio = nan(), np.zeros(n), nan()
    out_score, out_rel = nan(), np.zeros(n)
    for a, b in blocks:
        if config.dead_use_activity:
            act_score[a:b], act_rel[a:b], ratio[a:b] = activity_channel(tracking, a, b, config)
        if config.use_outside_cue:
            out_score[a:b], out_rel[a:b] = outside_channel(features, a, b, config)
    channels = {}
    if config.dead_use_activity:
        channels["activity"] = Channel("activity", act_score, act_rel, config.activity_gain,
                                       zero=config.activity_zero, source="player")
    if config.use_outside_cue:
        channels["outside"] = Channel("outside", out_score, out_rel, config.outside_dead_gain,
                                      zero=0.0, source="ball")
    return channels, ratio
