"""Dead-ball evidence from player and ball movement. Produces a signed score plus a reliability.

    llr          > 0 favours a dead ball, < 0 favours live play, exactly 0 means "no usable evidence"
    reliability  in [0, 1]: how much of the evidence budget was actually observed at that frame

The output plugs into the decoder through `evidence.build_evidence(..., external_dead=llr)`; the
decoder is unchanged. Scores are in the same uncalibrated evidence units as the ownership
evidence (see config.py) and are never presented as probabilities.

Invariants (each has a test):

* **An unobserved ball is not a stoppage.** Nothing here is derived from ball *absence*. The
  ball-based cues are undefined (score 0, reliability 0) wherever the ball, or the players around
  it, were not observed.
* **Missing is not slow.** A player speed exists only when every frame of its difference window
  was observed and no single step looks like a teleport (an identity switch or a re-acquisition).
  Missing players never count as standing still, and a frame with too few valid speeds has no
  statistic at all.
* **Halves are isolated.** Every temporal aggregate is computed inside one block, so nothing leaks
  across the halftime gap.
* **Cues are independent and switchable** (`dead_use_*`), so each can be ablated on its own.

Cues (all rolling medians over +/- `dead_window_halfwidth` frames, inside a block):

    activity    median speed over all tracked players. Walking => dead, jogging => live.
    near_ball   median speed of the players within `near_ball_radius_m` of the ball (>= 3 of them
                with a valid speed); defined only while the ball is observed.
    ball_motion (optional, off by default) strict ball speed; slow => dead. Weak and inconsistent
                in the measurements behind this phase, kept for the ablation.

A stationary ball is deliberately *not* a cue: in the measurements it did not separate dead from
live frames at all.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import DEFAULT
from .features import strict_window_valid


@dataclass
class DeadBallEvidence:
    llr: np.ndarray            # (N,) float64, >0 dead, <0 live, 0 unknown
    reliability: np.ndarray    # (N,) float64 in [0, 1]
    parts: dict = field(default_factory=dict)   # name -> (signed score in [-1, 1], reliability)


def strict_speed(xy, halfwidth, fps, max_step, max_speed):
    """Speed (m/s) by central difference over +/- halfwidth frames, with strict validity.

    `xy` is (T, 2) or (T, S, 2) in metres, NaN = missing. A speed is returned only if every frame
    in the window [t-k, t+k] is observed and no frame-to-frame step in it exceeds `max_step` (m).
    Otherwise NaN. Speeds above `max_speed` are also discarded.
    """
    k = halfwidth
    T = xy.shape[0]
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if k <= 0 or T <= 2 * k:
        return out
    ok = strict_window_valid(xy, k, max_step)
    speed = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    disp = np.hypot(xy[2 * k:, ..., 0] - xy[:-2 * k, ..., 0], xy[2 * k:, ..., 1] - xy[:-2 * k, ..., 1])
    speed[k:-k] = disp * (fps / (2.0 * k))
    speed[~ok] = np.nan
    speed[speed > max_speed] = np.nan
    return speed


def rolling_median(values, halfwidth, min_coverage):
    """(median, coverage) of a 1-D series over +/- halfwidth, ignoring NaN.

    coverage = fraction of the window that is observed; where it is below `min_coverage` the median
    is NaN (not enough to say anything). Edges are padded with NaN, so they lose coverage.
    """
    n = len(values)
    if halfwidth <= 0:
        cov = np.isfinite(values).astype(np.float64)
        return values.astype(np.float64), cov
    padded = np.pad(values.astype(np.float32), (halfwidth, halfwidth), constant_values=np.nan)
    window = sliding_window_view(padded, 2 * halfwidth + 1)
    count = np.isfinite(window).sum(axis=1)
    coverage = count / float(2 * halfwidth + 1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN windows are expected and handled
        median = np.nanmedian(window, axis=1)
    median = np.where(coverage >= min_coverage, median, np.nan)
    return median.astype(np.float64), coverage


def local_quantile(values, halfwidth, q, min_count, stride=125):
    """Rolling q-quantile of a 1-D series over +/- halfwidth, ignoring NaN.

    Evaluated every `stride` frames and linearly interpolated in between (the level changes over
    minutes, so this loses nothing); NaN where a window holds fewer than `min_count` observations.
    """
    n = len(values)
    centres = np.unique(np.concatenate([np.arange(0, n, stride), [n - 1]]))
    level = np.full(len(centres), np.nan)
    for i, c in enumerate(centres):
        window = values[max(0, c - halfwidth): c + halfwidth + 1]
        window = window[np.isfinite(window)]
        if len(window) >= min_count:
            level[i] = np.quantile(window, q)
    good = np.isfinite(level)
    if not good.any():
        return np.full(n, np.nan)
    good_centres = centres[good]
    position = np.arange(n)
    out = np.interp(position, good_centres, level[good])
    # Do not bridge stretches whose level windows were unobserved: beyond `max_gap` frames from the
    # nearest evaluated centre the level is unknown rather than extrapolated.
    right = np.clip(np.searchsorted(good_centres, position), 0, len(good_centres) - 1)
    left = np.clip(right - 1, 0, len(good_centres) - 1)
    nearest = np.minimum(np.abs(position - good_centres[left]), np.abs(position - good_centres[right]))
    out[nearest > 2 * stride + halfwidth // 2] = np.nan
    return out


def _signed_score(speed, reference, scale, config):
    """tanh((reference - speed)/scale): + when slower than the reference. NaN stays NaN."""
    score = np.tanh((reference - speed) / scale)
    if not config.dead_allow_negative:
        score = np.where(score > 0, score, np.where(np.isfinite(score), 0.0, np.nan))
    else:
        score = np.where(score < 0, score * config.alive_evidence_scale, score)
    return score


def _slow_score(aggregate, config):
    """Signed slow-vs-fast score of a rolling aggregate under the configured reference."""
    if config.dead_speed_reference == "absolute":
        return _signed_score(aggregate, config.walk_speed_ref, config.walk_speed_scale, config)
    if config.dead_speed_reference != "relative":
        raise ValueError("dead_speed_reference must be 'absolute' or 'relative'")
    level = local_quantile(aggregate, config.dead_level_halfwidth, config.dead_level_quantile,
                           config.dead_level_min_frames)
    with np.errstate(all="ignore"):
        ratio = aggregate / level
    return _signed_score(ratio, config.relative_ref, config.relative_scale, config)


def _collective_median(speeds, min_players):
    """Per-frame median over the slots with a valid speed; NaN if fewer than `min_players`."""
    count = np.isfinite(speeds).sum(axis=1)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)       # frames with no valid speed are expected
        median = np.nanmedian(speeds, axis=1)
    return np.where(count >= min_players, median, np.nan)


def _player_speeds(tracking, a, b, config):
    if config.dead_speed_source == "native":
        speed = tracking.player_speed[a:b].astype(np.float32)
        speed = np.where(np.isfinite(tracking.player_xy[a:b]).all(axis=-1), speed, np.nan)
        speed[speed > config.player_max_speed] = np.nan
        return speed
    if config.dead_speed_source != "positions":
        raise ValueError("dead_speed_source must be 'positions' or 'native'")
    return strict_speed(tracking.player_xy[a:b], config.velocity_halfwidth, config.fps,
                        config.max_step_m, config.player_max_speed)


def _block(tracking, a, b, config):
    """{cue: (signed score, reliability)} for frames [a, b)."""
    hw = config.dead_window_halfwidth
    speeds = _player_speeds(tracking, a, b, config)                 # (T, S)
    out = {}

    if config.dead_use_activity:
        per_frame = _collective_median(speeds, config.min_speed_players)
        agg, cov = rolling_median(per_frame, hw, config.dead_min_coverage)
        out["activity"] = (_slow_score(agg, config), cov)

    if config.dead_use_near_ball:
        ball = tracking.ball_xy[a:b]
        player = tracking.player_xy[a:b]
        with np.errstate(invalid="ignore"):
            dist = np.hypot(player[..., 0] - ball[:, None, 0], player[..., 1] - ball[:, None, 1])
        near = np.where(dist < config.near_ball_radius_m, speeds, np.nan)
        per_frame = _collective_median(near, config.near_ball_min_players)
        agg, cov = rolling_median(per_frame, hw, config.near_ball_min_coverage)
        out["near_ball"] = (_slow_score(agg, config), cov)

    if config.dead_use_ball_motion:
        speed = strict_speed(tracking.ball_xy[a:b], config.velocity_halfwidth, config.fps,
                             config.ball_max_step_m, config.ball_max_speed)
        agg, cov = rolling_median(speed, hw, config.near_ball_min_coverage)
        out["ball_motion"] = (_signed_score(agg, config.ball_speed_ref, config.ball_speed_scale, config), cov)
    return out


def dead_ball_evidence(tracking, blocks, config=DEFAULT):
    """DeadBallEvidence for the whole recording; frames outside `blocks` get 0 / 0.

    `blocks` are (start, stop) video-frame ranges (stop exclusive), processed independently.
    """
    n = tracking.n_frames
    gains = {"activity": config.activity_gain, "near_ball": config.near_ball_gain,
             "ball_motion": config.ball_motion_gain}
    llr = np.zeros(n)
    budget = np.zeros(n)
    seen = np.zeros(n)
    parts = {}
    for a, b in blocks:
        if not 0 <= a < b <= n:
            raise ValueError("block ({}, {}) outside 0..{}".format(a, b, n))
        for name, (score, reliability) in _block(tracking, a, b, config).items():
            usable = np.isfinite(score)
            rel = np.where(usable, np.clip(reliability, 0.0, 1.0), 0.0)
            contribution = np.where(usable, gains[name] * rel * np.nan_to_num(score), 0.0)
            llr[a:b] += contribution
            budget[a:b] += gains[name]
            seen[a:b] += gains[name] * rel
            full = parts.setdefault(name, (np.full(n, np.nan), np.zeros(n)))
            full[0][a:b] = score
            full[1][a:b] = rel
    reliability = np.divide(seen, budget, out=np.zeros(n), where=budget > 0)
    return DeadBallEvidence(llr=llr, reliability=reliability, parts=parts)


def ownership_compensation(own_llr, own_weight, blocks, config=DEFAULT):
    """Dead-score offset (>= 0, to be subtracted) for frames with no ownership evidence.

    The decoder gives the winning alive state |own|/2 whenever the ball is observed and in
    control; where it is not, that bonus is absent. The offset is the mean of |own|/2 over the
    frames of the same block that do have evidence, applied to the frames that do not. Zero when
    compensation is disabled or a block has no observed frame. Label-free and deterministic.
    """
    offset = np.zeros(len(own_llr))
    if not config.dead_compensate_missing_ownership:
        return offset
    available = own_weight > 0
    for a, b in blocks:
        seen = available[a:b]
        if seen.any():
            offset[a:b] = np.where(seen, 0.0, 0.5 * np.abs(own_llr[a:b][seen]).mean())
    return offset
