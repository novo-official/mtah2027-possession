"""Soft per-frame evidence for the three possession states. No labels are produced here.

Two independent channels, because "no evidence of possession" is not "evidence of no possession":

* **Ownership evidence** `own_llr` (evidence units, >0 favours home, <0 favours away, 0 = says
  nothing) with a trust `own_weight` in [0, 1]. Without a usable ball both are exactly 0.
* **Dead-ball evidence** `dead_llr` (>0 favours dead, 0 = says nothing). Ball absence contributes
  nothing. The one built-in cue is a ball tracked outside the pitch lines; every other future cue
  (collective player speed, restarts, tags) is supplied through the `external` argument.

Evidence units are log-likelihood-ratio-like numbers that add up over frames. They are NOT
calibrated probabilities (see config.py) and are not reported as such.

Ownership model, per frame with a ball:

    margin   = d_away - d_home                          nearer team claims the ball (m)
    claim    = tanh(margin / margin_scale)              in (-1, 1), 0 when the teams are equidistant
    prox     = exp(-(min(d_home, d_away) / d0)^2)       a loose ball says little about control
    motion   = 1 / (1 + (v_rel / v0)^2)                 a ball in flight says little about control
    observe  = min(1, min(n_home, n_away) / full)       few tracked players: margin is unreliable
    consist  = floor + (1 - floor) * agreement          flickering sign of the claim lowers trust

    own_weight = prox * motion * observe * consist
    own_llr    = gain * own_weight * claim
"""

from dataclasses import dataclass

import numpy as np

from .config import DEFAULT


@dataclass
class Evidence:
    own_llr: np.ndarray       # (N,) float64
    own_weight: np.ndarray    # (N,) float64 in [0, 1]
    dead_llr: np.ndarray      # (N,) float64
    parts: dict               # diagnostic components: claim, proximity, motion, observation, consistency

    @property
    def available(self):
        """Frames on which ownership evidence says something at all."""
        return self.own_weight > 0


def _consistency(claim, valid, halfwidth, floor):
    """Agreement of the sign of `claim` within +/- halfwidth frames, mapped to [floor, 1].

    agreement = |sum of signs| / count over the valid frames in the window (1 = unanimous). Frames
    with no valid neighbour (including themselves) get the floor... except that they carry no
    weight anyway because `claim` is unavailable there.
    """
    if halfwidth <= 0:
        return np.ones(claim.shape[0])
    sign = np.where(valid, np.sign(claim), 0.0)
    count = valid.astype(np.float64)
    n = sign.shape[0]
    cs_sign = np.concatenate(([0.0], np.cumsum(sign)))
    cs_count = np.concatenate(([0.0], np.cumsum(count)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - halfwidth)
    hi = np.minimum(n, idx + halfwidth + 1)
    total = cs_sign[hi] - cs_sign[lo]
    seen = cs_count[hi] - cs_count[lo]
    agreement = np.where(seen > 0, np.abs(total) / np.maximum(seen, 1.0), 0.0)
    return floor + (1.0 - floor) * agreement


def ownership_evidence(features, config=DEFAULT):
    """(own_llr, own_weight, parts) from Features. Zero everywhere the ball or a team is unobserved."""
    d_h = features.d_home.astype(np.float64)
    d_a = features.d_away.astype(np.float64)
    usable = features.ball_valid & np.isfinite(d_h) & np.isfinite(d_a)

    margin = np.where(usable, d_a - d_h, 0.0)
    claim = np.tanh(margin / config.margin_scale_m)

    d_min = np.where(usable, np.minimum(d_h, d_a), np.inf)
    proximity = np.exp(-(d_min / config.control_radius_m) ** 2)

    rel = features.rel_speed.astype(np.float64)
    motion = np.where(np.isfinite(rel), 1.0 / (1.0 + (rel / config.relative_speed_scale) ** 2),
                      config.motion_unknown)

    fewest = np.minimum(features.n_valid_home, features.n_valid_away).astype(np.float64)
    observation = np.clip(fewest / float(config.full_players), 0.0, 1.0)

    consistency = _consistency(claim, usable, config.consistency_halfwidth, config.consistency_floor)

    weight = np.where(usable, proximity * motion * observation * consistency, 0.0)
    own_llr = config.evidence_gain * weight * claim
    parts = dict(claim=claim, proximity=proximity, motion=motion, observation=observation,
                 consistency=consistency)
    return own_llr, weight, parts


def dead_evidence(features, config=DEFAULT, external=None):
    """Dead-ball log-likelihood-ratio per frame (>0 favours dead, 0 = no information).

    Absent ball => 0. Only a ball *tracked* outside the lines by more than `outside_margin_m`
    adds evidence. `external` (array of length N, evidence units) is added unchanged: this is the
    hook for collective player speed, restart detection, Smart Tagging, and so on.
    """
    dead = np.zeros(features.n_frames, dtype=np.float64)
    if config.use_outside_cue:
        outside = np.nan_to_num(features.ball_outside_m.astype(np.float64), nan=0.0)
        dead += np.where(features.ball_valid & (outside > config.outside_margin_m),
                         config.outside_dead_gain, 0.0)
    if external is not None:
        external = np.asarray(external, dtype=np.float64)
        if external.shape != dead.shape:
            raise ValueError("external dead evidence has shape {}, expected {}".format(
                external.shape, dead.shape))
        dead += np.nan_to_num(external, nan=0.0)
    return dead


def build_evidence(features, config=DEFAULT, external_dead=None):
    own_llr, weight, parts = ownership_evidence(features, config)
    return Evidence(own_llr=own_llr, own_weight=weight,
                    dead_llr=dead_evidence(features, config, external_dead), parts=parts)


def hard_nearest_evidence(features, config=DEFAULT):
    """Ablation: the baseline's hard rule expressed as evidence (+/- gain per frame, 0 without ball).

    Same units as `ownership_evidence`, so the same decoder can be run on it and the difference
    between the two is attributable to the evidence alone.
    """
    sign = np.where(features.nearest_team == 0, 1.0, np.where(features.nearest_team == 1, -1.0, 0.0))
    return Evidence(own_llr=config.evidence_gain * sign, own_weight=np.abs(sign),
                    dead_llr=dead_evidence(features, config), parts={})
