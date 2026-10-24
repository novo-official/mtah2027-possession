"""Per-frame geometric features from a Tracking. Pure functions, NumPy only, no labels.

Every feature carries its own availability: a quantity that cannot be computed is NaN (or -1 for
an index) and the boolean masks say which frames can be trusted. Absence of the ball produces
*unavailable* features, never a value that looks like evidence.

Features are computed block by block (`blocks` = [(start, stop), ...], stop exclusive) so that a
velocity, a gap length or a consistency window never reaches across the halftime gap. Frames
outside every block stay unavailable.
"""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT
from .data import AWAY, HOME


@dataclass
class Features:
    ball_valid: np.ndarray        # (N,) bool: ball position usable
    ball_outside_m: np.ndarray    # (N,) float32: metres outside the pitch rectangle (0 inside), NaN if no ball
    ball_speed: np.ndarray        # (N,) float32 m/s from position differences, NaN if unavailable
    ball_vel: np.ndarray          # (N, 2) float32 m/s
    n_valid_home: np.ndarray      # (N,) int16 valid tracked home players
    n_valid_away: np.ndarray      # (N,) int16
    d_home: np.ndarray            # (N,) float32 nearest home player to the ball (m), NaN if undefined
    d_away: np.ndarray            # (N,) float32
    nearest_team: np.ndarray      # (N,) int8: 0 home, 1 away, -1 none -- the baseline's hard rule
    slot_home: np.ndarray         # (N,) int16 slot index of the nearest home player (-1 if none)
    slot_away: np.ndarray         # (N,) int16
    rel_speed: np.ndarray         # (N,) float32 |v_ball - v_nearest_player| m/s, NaN if unavailable
    gap_length: np.ndarray        # (N,) int32 length of the ball-missing run containing the frame (0 if valid)
    in_block: np.ndarray          # (N,) bool: frame lies inside a processed block

    @property
    def n_frames(self):
        return self.ball_valid.shape[0]


def _central_velocity(xy, halfwidth, fps, max_speed):
    """Velocity (m/s) by central difference over +/- halfwidth frames; NaN where either end is
    missing, near the edges, or above `max_speed` (a jump, not motion). Works on (T, 2) or (T, S, 2)."""
    velocity = np.full(xy.shape, np.nan, dtype=np.float32)
    k = halfwidth
    if k <= 0 or xy.shape[0] <= 2 * k:
        return velocity
    velocity[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2.0 * k))
    speed = np.hypot(velocity[..., 0], velocity[..., 1])
    velocity[speed > max_speed] = np.nan
    return velocity


central_velocity = _central_velocity   # public name; the leading-underscore one is kept for compatibility


def strict_window_valid(xy, halfwidth, max_step):
    """Bool array: True where the central-difference window [t-k, t+k] is trustworthy.

    A window is trustworthy when every frame in it is observed and no frame-to-frame step in it is
    larger than `max_step` metres. A larger step is an identity switch or a re-acquisition, not
    motion, and a NaN gap inside the window could hide one. `xy` is (T, 2) or (T, S, 2).
    """
    k = halfwidth
    T = xy.shape[0]
    if k <= 0 or T <= 2 * k:
        return np.zeros(xy.shape[:-1], dtype=bool)
    seen = np.isfinite(xy).all(axis=-1)
    step = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    step[1:] = np.hypot(xy[1:, ..., 0] - xy[:-1, ..., 0], xy[1:, ..., 1] - xy[:-1, ..., 1])
    jump = np.nan_to_num(step, nan=0.0) > max_step

    def window_sum(flag, lo, hi):
        cs = np.concatenate([np.zeros((1,) + flag.shape[1:], dtype=np.int32),
                             np.cumsum(flag, axis=0, dtype=np.int32)], axis=0)
        idx = np.arange(T)
        return cs[np.clip(idx + hi + 1, 0, T)] - cs[np.clip(idx + lo, 0, T)]

    return (window_sum(~seen, -k, k) == 0) & (window_sum(jump, -k + 1, k) == 0)


def strict_velocity(xy, halfwidth, fps, max_step, max_speed):
    """Velocity (m/s, same shape as `xy`) by central difference, NaN unless the window is
    trustworthy (see strict_window_valid) and the speed is at most `max_speed`."""
    k = halfwidth
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    if k <= 0 or xy.shape[0] <= 2 * k:
        return out
    ok = strict_window_valid(xy, k, max_step)
    out[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2.0 * k))
    out[~ok] = np.nan
    speed = np.hypot(out[..., 0], out[..., 1])
    out[speed > max_speed] = np.nan
    return out


def track_age(xy, max_step):
    """Frames each position has been continuously observed without a jump (0 = just (re)acquired).

    `xy` is (T, S, 2). A frame breaks continuity when it is missing or farther than `max_step`
    from the previous frame; the age counts frames since the last break.
    """
    T = xy.shape[0]
    seen = np.isfinite(xy).all(axis=-1)
    step = np.full(seen.shape, np.nan, dtype=np.float32)
    step[1:] = np.hypot(xy[1:, ..., 0] - xy[:-1, ..., 0], xy[1:, ..., 1] - xy[:-1, ..., 1])
    broken = ~seen | (np.nan_to_num(step, nan=0.0) > max_step)
    broken[0] = True
    broken[1:] |= ~seen[:-1]          # the first frame after a gap is a (re)acquisition
    idx = np.arange(T).reshape((T,) + (1,) * (xy.ndim - 2))
    last_break = np.maximum.accumulate(np.where(broken, idx, -1), axis=0)
    return np.where(seen, idx - last_break, 0).astype(np.int32)


def _run_lengths(missing):
    """For a bool array, the length of the True-run each True element belongs to (0 elsewhere)."""
    out = np.zeros(missing.shape[0], dtype=np.int32)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for a, b in zip(starts, stops):
        out[a:b] = b - a
    return out


def _block_features(tracking, a, b, config):
    """Features for frames [a, b) as a dict of arrays of length b - a."""
    fps = config.fps
    player = tracking.player_xy[a:b]                 # (T, S, 2)
    ball = tracking.ball_xy[a:b]                     # (T, 2)
    team = tracking.slot_team
    length, width = tracking.pitch
    T = b - a

    ball_valid = np.isfinite(ball).all(axis=1)
    player_valid = np.isfinite(player).all(axis=2)   # (T, S)

    # Distance outside the pitch rectangle (0 when inside); NaN without a ball.
    ox = np.maximum(np.maximum(-ball[:, 0], ball[:, 0] - length), 0.0)
    oy = np.maximum(np.maximum(-ball[:, 1], ball[:, 1] - width), 0.0)
    outside = np.hypot(ox, oy).astype(np.float32)
    outside[~ball_valid] = np.nan

    # Distances to every slot; missing ball or missing player -> inf (never nearest).
    with np.errstate(invalid="ignore"):
        dist = np.hypot(player[..., 0] - ball[:, None, 0], player[..., 1] - ball[:, None, 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)

    is_home = (team == HOME)[None, :]
    is_away = (team == AWAY)[None, :]
    d_home = np.where(is_home, dist, np.inf).min(axis=1)
    d_away = np.where(is_away, dist, np.inf).min(axis=1)
    slot_home = np.where(np.isfinite(d_home), np.argmin(np.where(is_home, dist, np.inf), axis=1), -1).astype(np.int16)
    slot_away = np.where(np.isfinite(d_away), np.argmin(np.where(is_away, dist, np.inf), axis=1), -1).astype(np.int16)
    d_home = np.where(np.isfinite(d_home), d_home, np.nan).astype(np.float32)
    d_away = np.where(np.isfinite(d_away), d_away, np.nan).astype(np.float32)

    # The baseline's hard rule, kept for reproduction and ablation: first minimum over all slots.
    any_valid = np.isfinite(dist).any(axis=1)
    nearest_slot = np.argmin(dist, axis=1)
    nearest_team = np.where(any_valid & ball_valid, team[nearest_slot], -1).astype(np.int8)

    # Velocities and the nearest player's relative motion.
    ball_vel = _central_velocity(ball, config.velocity_halfwidth, fps, config.ball_max_speed)
    player_vel = _central_velocity(player, config.velocity_halfwidth, fps, config.player_max_speed)
    rows = np.arange(T)
    near_vel = player_vel[rows, nearest_slot]                       # (T, 2)
    rel = np.hypot(ball_vel[:, 0] - near_vel[:, 0], ball_vel[:, 1] - near_vel[:, 1])
    rel[~(any_valid & ball_valid)] = np.nan
    rel = rel.astype(np.float32)

    return dict(
        ball_valid=ball_valid,
        ball_outside_m=outside,
        ball_speed=np.hypot(ball_vel[:, 0], ball_vel[:, 1]).astype(np.float32),
        ball_vel=ball_vel,
        n_valid_home=(player_valid & is_home).sum(axis=1).astype(np.int16),
        n_valid_away=(player_valid & is_away).sum(axis=1).astype(np.int16),
        d_home=d_home, d_away=d_away, nearest_team=nearest_team, rel_speed=rel,
        slot_home=slot_home, slot_away=slot_away,
        gap_length=_run_lengths(~ball_valid),
    )


def compute_features(tracking, blocks=None, config=DEFAULT):
    """Features for the whole recording, computed independently inside each block.

    `blocks` is a list of (start, stop) video-frame ranges (stop exclusive); None means one block
    covering everything. Frames outside all blocks are unavailable (ball_valid False, in_block False).
    """
    n = tracking.n_frames
    blocks = [(0, n)] if blocks is None else [(int(a), int(b)) for a, b in blocks]
    out = dict(
        ball_valid=np.zeros(n, bool),
        ball_outside_m=np.full(n, np.nan, np.float32),
        ball_speed=np.full(n, np.nan, np.float32),
        ball_vel=np.full((n, 2), np.nan, np.float32),
        n_valid_home=np.zeros(n, np.int16), n_valid_away=np.zeros(n, np.int16),
        d_home=np.full(n, np.nan, np.float32), d_away=np.full(n, np.nan, np.float32),
        nearest_team=np.full(n, -1, np.int8), rel_speed=np.full(n, np.nan, np.float32),
        slot_home=np.full(n, -1, np.int16), slot_away=np.full(n, -1, np.int16),
        gap_length=np.zeros(n, np.int32), in_block=np.zeros(n, bool),
    )
    for a, b in blocks:
        if not 0 <= a < b <= n:
            raise ValueError("block ({}, {}) outside 0..{}".format(a, b, n))
        part = _block_features(tracking, a, b, config)
        for key, value in part.items():
            out[key][a:b] = value
        out["in_block"][a:b] = True
    return Features(**out)
