"""Observation-aware fusion of dead-ball evidence (Phase 4.6).

Three things are kept apart that earlier phases mixed:

* **Ownership evidence** (home vs away) lives in evidence.py and only decides H versus A.
* **Dead-ball evidence** is a set of *channels*, each with a signed score (> 0 stopped, < 0 live),
  a reliability in [0, 1] (was it observed, and plausibly), and a gain.
* **Observation reliability** is explicit: a channel that is not observed has reliability 0 and
  contributes exactly 0. There is no fixed bonus or penalty for a missing ball; a ball-based
  channel (out-of-play, restart confirmation) simply says nothing while the ball is unseen, and a
  player-based channel (movement) keeps its full meaning while the ball is unseen.

The fused score is

    dead_net(t) = sum over channels c of  reliability_c(t) * ( gain_c * score_c(t) - zero_c )

where `zero_c` is the channel's zero point (nats per frame at full reliability): the level of
evidence below which the channel's reading is just as typical of live play as of a stoppage.
Because `zero_c` is multiplied by the reliability, it disappears with the observation instead of
turning an unobserved stretch into evidence for live play. Units are uncalibrated nats; nothing
here is a calibrated probability.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Channel:
    name: str
    score: np.ndarray            # (N,) signed score per unit gain; NaN = unavailable
    reliability: np.ndarray      # (N,) in [0, 1]
    gain: float                  # nats per frame at reliability 1 and |score| 1
    zero: float = 0.0            # zero point, nats per frame at reliability 1
    source: str = "player"       # "player" or "ball": what had to be observed

    def contribution(self):
        usable = np.isfinite(self.score) & (self.reliability > 0)
        rel = np.where(usable, np.clip(self.reliability, 0.0, 1.0), 0.0)
        return np.where(usable, rel * (self.gain * np.nan_to_num(self.score) - self.zero), 0.0)


@dataclass
class ObservationEvidence:
    dead_net: np.ndarray                      # (N,) fused dead-ball score (> 0 favours dead)
    channels: dict = field(default_factory=dict)
    supported: np.ndarray = None              # (N,) bool: at least one channel observed
    player_supported: np.ndarray = None       # (N,) bool: a player-based channel observed
    ball_supported: np.ndarray = None         # (N,) bool: a ball-based channel observed


def fuse(channels, n):
    """ObservationEvidence from a list of Channel, over a recording of length n."""
    net = np.zeros(n)
    supported = np.zeros(n, bool)
    by_source = {"player": np.zeros(n, bool), "ball": np.zeros(n, bool)}
    named = {}
    for channel in channels:
        net += channel.contribution()
        usable = np.isfinite(channel.score) & (channel.reliability > 0)
        supported |= usable
        by_source[channel.source] |= usable
        named[channel.name] = channel
    return ObservationEvidence(dead_net=net, channels=named, supported=supported,
                               player_supported=by_source["player"], ball_supported=by_source["ball"])

