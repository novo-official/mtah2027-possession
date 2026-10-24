"""Restart evidence: when does play actually resume? (Phase 4.6)

A separate signal from stoppage entry, testable on its own. It produces *negative* dead-ball
evidence (live-play evidence) over a short pulse after a detected restart, so that a stoppage can end
at the supported moment instead of persisting or ending on an unrelated dip.

A restart is detected in two steps:

1. **Sustained rise of play-level movement.** The movement ratio (activity relative to the local
   live level, from stoppage.py) has a median over the `restart_window` frames before a frame at most
   `restart_low` and over the `restart_window` frames after at least `restart_high`. The first frame
   of each such run is the candidate. Players moving before a restart do not satisfy this on their
   own, because...
2. **...the ball must confirm it.** Within [-restart_search_before, +restart_search_after] frames of the
   candidate, either (a) the ball moves at >= `restart_ball_speed` m/s for `restart_ball_frames`
   consecutive strictly valid frames (renewed ball motion), or (b) a contact episode of one team
   becomes established (the first controlled interaction). If the ball is visible but neither
   happens, the players are moving and the ball is not in play: no restart. If the ball is not
   observed anywhere in the window, a strong sustained movement rise alone counts at
   `restart_movement_only_scale` of the gain: a stationary or hidden ball must not prevent detecting
   a restart that the players clearly show.

The pulse carries no team: which team has the ball afterwards is left to the ownership evidence.
"""

import numpy as np

from .config import DEFAULT
from .dead_ball import rolling_median
from .features import strict_velocity
from .handover import frames_to_establish
from .observation import Channel
from .ownership import control_signals


def rise_candidates(ratio, config):
    """Indices (within the series) of the first frame of each run where the ratio rises from
    play-slow to play-level: median before <= restart_low and median after >= restart_high."""
    w = config.restart_window
    med, _ = rolling_median(ratio, w, 0.5)
    n = len(med)
    pre = np.full(n, np.nan)
    post = np.full(n, np.nan)
    if n > 2 * w + 2:
        pre[w + 1:] = med[:-(w + 1)]
        post[:-(w + 1)] = med[w + 1:]
    flag = np.nan_to_num(pre, nan=np.inf) <= config.restart_low
    flag &= np.nan_to_num(post, nan=-np.inf) >= config.restart_high
    edges = np.diff(np.concatenate(([0], flag.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1)


def _ball_motion_confirms(ball_speed, lo, hi, speed, frames):
    """True if the ball moves at >= speed for `frames` consecutive valid frames inside [lo, hi)."""
    seg = np.nan_to_num(ball_speed[lo:hi], nan=0.0) >= speed
    if seg.sum() < frames:
        return False
    edges = np.diff(np.concatenate(([0], seg.astype(np.int8), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return bool(len(runs) and runs.max() >= frames)


def restart_channel(tracking, features, blocks, ratio, config=DEFAULT):
    """(Channel, events). events: [(frame, kind)] with kind 'ball' (confirmed) or 'movement' (ball unobserved)."""
    n = tracking.n_frames
    score = np.full(n, np.nan)
    reliability = np.zeros(n)
    events = []
    signals = control_signals(tracking, features, blocks, config)
    need = frames_to_establish(config)
    established = [int(frames[need - 1]) for episodes in (signals.episodes_home, signals.episodes_away)
                   for _, _, length, frames in episodes if length >= need]
    established = np.array(sorted(established))
    for a, b in blocks:
        ball_vel = strict_velocity(tracking.ball_xy[a:b], config.velocity_halfwidth, config.fps,
                                   config.ball_max_step_m, config.ball_max_speed)
        ball_speed = np.hypot(ball_vel[:, 0], ball_vel[:, 1])
        valid = features.ball_valid[a:b]
        score[a:b] = 0.0
        for c in rise_candidates(ratio[a:b], config):
            lo = max(0, c - config.restart_search_before)
            hi = min(b - a, c + config.restart_search_after)
            moves = _ball_motion_confirms(ball_speed, lo, hi, config.restart_ball_speed, config.restart_ball_frames)
            contact = bool(len(established) and np.any((established >= a + lo) & (established < a + hi)))
            if moves or contact:
                strength, kind = 1.0, "ball"
            elif valid[lo:hi].mean() < 0.2:
                strength, kind = config.restart_movement_only_scale, "movement"
            else:
                continue                                  # ball visible and not in play: players are just moving
            end = min(b, a + c + config.restart_pulse)
            score[a + c:end] = -1.0
            reliability[a + c:end] = np.maximum(reliability[a + c:end], strength)
            events.append((int(a + c), kind))
    return Channel("restart", score, reliability, config.restart_gain, zero=0.0, source="ball"), events
