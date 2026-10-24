"""Controlled-ownership evidence (Phase 4.4): which team has *established control* of the ball.

The Phase 4.2 evidence asks which team's player is nearest. This module keeps that evidence as its
starting point (the same claim, proximity, observation and consistency terms, reused from
`evidence.ownership_evidence`, so there is one definition of each) and adds three separately
switchable refinements, each with its own measurable effect:

    own_strict_motion    relative motion of the claiming team's nearest player, from strictly valid
                         velocities (never through a gap or an identity switch)
    own_track_stability  a just-(re)acquired track is trusted less
    own_reception        sustained contact (the ball within reach and moving with the player)
                         becomes an "episode"; its strength grows with its length, so a single touch
                         or deflection is weak and a second of control is strong

A *contested ball* penalty (a second team's player close to the ball) was tested and refuted on the
public window: the nearest-team rule was *more* accurate there, not less, because a ball with two
players on it is usually at someone's feet. It is deliberately not implemented.

Missing observations stay missing evidence: without a ball, or without a trustworthy velocity, the
corresponding term is neutral or zero and nothing is interpolated across a gap. Home and away are
processed by identical code on mirrored inputs; no direction of play or possession share is assumed.
"""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT
from .evidence import Evidence, ownership_evidence
from .features import strict_velocity, track_age


@dataclass
class ControlSignals:
    rel_home: np.ndarray        # (N,) relative speed ball vs the nearest home player, NaN if untrustworthy
    rel_away: np.ndarray        # (N,)
    age_home: np.ndarray        # (N,) track age of the nearest home player (frames)
    age_away: np.ndarray        # (N,)
    contact_home: np.ndarray    # (N,) bool: contact this frame (within reach and moving with the ball)
    contact_away: np.ndarray
    strength_home: np.ndarray   # (N,) strength of the contact episode containing the frame (0 if none)
    strength_away: np.ndarray
    episodes_home: list         # [(start, stop, L, contact_frames)] in absolute frame numbers
    episodes_away: list


def contact_episodes(contact, bridge, scale):
    """(strength, episodes) for one team from its per-frame contact flags.

    Runs of contact separated by at most `bridge` non-contact frames form one episode. Every frame
    of an episode (including bridged ones) gets strength 1 - exp(-L / scale), L being the number of
    contact frames in the episode; frames outside any episode get 0. `episodes` lists
    (start, stop, L, contact_frames) with `stop` exclusive and the indices of the contact frames,
    so a caller can ask when the L-th contact frame happened (the moment control became established).
    """
    n = len(contact)
    strength = np.zeros(n, dtype=np.float64)
    index = np.flatnonzero(contact)
    if len(index) == 0:
        return strength, []
    breaks = np.flatnonzero(np.diff(index) - 1 > bridge)
    first = np.concatenate(([0], breaks + 1))
    last = np.concatenate((breaks, [len(index) - 1]))
    episodes = []
    for i, j in zip(first, last):
        frames = index[i:j + 1]
        length = len(frames)
        strength[frames[0]:frames[-1] + 1] = 1.0 - np.exp(-length / scale)
        episodes.append((int(frames[0]), int(frames[-1]) + 1, length, frames))
    return strength, episodes


def _relative_speed(ball_vel, player_vel, slot):
    """|v_ball - v_player| for the player at `slot` per frame; NaN where either velocity is unavailable."""
    n = len(slot)
    safe = np.where(slot >= 0, slot, 0)
    pv = player_vel[np.arange(n), safe]
    rel = np.hypot(ball_vel[:, 0] - pv[:, 0], ball_vel[:, 1] - pv[:, 1])
    rel[slot < 0] = np.nan
    return rel


def control_signals(tracking, features, blocks, config=DEFAULT):
    """ControlSignals for the whole recording, computed independently inside each block."""
    n = tracking.n_frames
    nan = lambda: np.full(n, np.nan, dtype=np.float32)
    rel_h, rel_a = nan(), nan()
    age_h, age_a = np.zeros(n, np.int32), np.zeros(n, np.int32)
    contact_h, contact_a = np.zeros(n, bool), np.zeros(n, bool)
    str_h, str_a = np.zeros(n), np.zeros(n)
    eps_h, eps_a = [], []
    for a, b in blocks:
        ball = tracking.ball_xy[a:b]
        player = tracking.player_xy[a:b]
        ball_vel = strict_velocity(ball, config.velocity_halfwidth, config.fps,
                                   config.ball_max_step_m, config.ball_max_speed)
        player_vel = strict_velocity(player, config.velocity_halfwidth, config.fps,
                                     config.max_step_m, config.player_max_speed)
        age = track_age(player, config.max_step_m)
        for slot_arr, d_arr, rel_out, age_out, contact_out, strength_out, eps_out in (
                (features.slot_home[a:b], features.d_home[a:b], rel_h, age_h, contact_h, str_h, eps_h),
                (features.slot_away[a:b], features.d_away[a:b], rel_a, age_a, contact_a, str_a, eps_a)):
            rel = _relative_speed(ball_vel, player_vel, slot_arr)
            rel_out[a:b] = rel
            rows = np.arange(b - a)
            age_out[a:b] = age[rows, np.where(slot_arr >= 0, slot_arr, 0)] * (slot_arr >= 0)
            near = np.nan_to_num(d_arr, nan=np.inf) < config.contact_radius_m
            moving_with = np.nan_to_num(rel, nan=np.inf) < config.contact_rel_speed
            contact = near & moving_with & features.ball_valid[a:b]
            contact_out[a:b] = contact
            strength, episodes = contact_episodes(contact, config.episode_bridge, config.episode_scale)
            strength_out[a:b] = strength
            eps_out.extend((x + a, y + a, length, frames + a) for x, y, length, frames in episodes)
    return ControlSignals(rel_h, rel_a, age_h, age_a, contact_h, contact_a, str_h, str_a, eps_h, eps_a)


def control_evidence(tracking, features, blocks, config=DEFAULT, signals=None):
    """(Evidence-like own_llr, own_weight, parts, ControlSignals) with the Phase 4.4 refinements.

    With all three refinements off this returns the Phase 4.2 numbers unchanged.
    """
    own_llr, weight, parts = ownership_evidence(features, config)
    signals = control_signals(tracking, features, blocks, config) if signals is None else signals
    usable = features.ball_valid & np.isfinite(features.d_home) & np.isfinite(features.d_away)
    home_claims = np.nan_to_num(features.d_home, nan=np.inf) <= np.nan_to_num(features.d_away, nan=np.inf)
    parts = dict(parts)

    motion = parts["motion"]
    if config.own_strict_motion:
        rel = np.where(home_claims, signals.rel_home, signals.rel_away).astype(np.float64)
        motion = np.where(np.isfinite(rel), 1.0 / (1.0 + (rel / config.relative_speed_scale) ** 2),
                          config.motion_unknown)
        parts["motion"] = motion

    stability = np.ones(len(weight))
    if config.own_track_stability:
        age = np.where(home_claims, signals.age_home, signals.age_away).astype(np.float64)
        stability = config.stability_floor + (1.0 - config.stability_floor) * np.minimum(
            1.0, age / float(config.stability_age_full))
        parts["stability"] = stability

    if config.own_strict_motion or config.own_track_stability:
        weight = np.where(usable, parts["proximity"] * motion * parts["observation"]
                          * parts["consistency"] * stability, 0.0)
        own_llr = config.evidence_gain * weight * parts["claim"]

    if config.own_reception:
        reception = config.reception_gain * (signals.strength_home - signals.strength_away)
        parts["reception"] = reception
        own_llr = own_llr + reception
        weight = np.maximum(weight, np.maximum(signals.strength_home, signals.strength_away))
    return own_llr, weight, parts, signals


def ownership_evidence_v2(tracking, features, blocks, config, dead_llr):
    """An Evidence object using the Phase 4.4 ownership terms and the given dead-ball channel."""
    own_llr, weight, parts, signals = control_evidence(tracking, features, blocks, config)
    parts["signals"] = signals
    return Evidence(own_llr=own_llr, own_weight=weight, dead_llr=dead_llr, parts=parts)
