"""Evidence for team-to-team handovers, expressed as time-varying home<->away switch costs.

What the signal is. Established control is a contact episode (see ownership.py) of strength at
least `established_strength`: a team's player within reach of the ball and moving with it for
roughly half a second or more. Walking through the match in time order, two *consecutive*
established episodes belonging to different teams mark a handover: control moved from team a to
team b somewhere between the end of a's episode and the start of b's.

What it is not. It is not derived from the decoder's state and it does not look at the ownership
evidence of the frames in between: it comes from the episodes alone, so it adds information the
per-frame evidence does not (that a *sustained* contact of the other team follows), and is
independent of what the decoder has decided.

How it enters the decoder. Switching to team b at frame t costs `switch_cost * (1 + handover_extra *
(1 - support_b(t)))`, where support_b(t) in [0, 1] is

    ramp from 0 to 1 over the `handover_slack` frames before b's episode begins, then 1 for
    `handover_pad` frames, never earlier than the end of a's episode (minus the pad).

so (i) a switch with no established new controller behind it costs more than normal, by at most
`handover_extra * switch_cost` and never more than the cheapest detour through the dead state, (ii) a supported switch costs the normal amount, and (iii) among equally supported
switch times inside a stretch of no evidence the *latest* is cheapest: possession stays with the last
team until the new one has established control, which is how the challenge defines a change. In log
terms this is a prior over *when* a switch happens, added to the per-frame evidence rather than
replacing it. It never lowers any cost below the normal switch cost, a missing ball creates no
support, and home and away are treated by the same code.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT


@dataclass
class Handovers:
    extra_ha: np.ndarray                  # (N,) extra cost of switching home -> away at frame t
    extra_ah: np.ndarray                  # (N,) extra cost of switching away -> home at frame t
    events: list = field(default_factory=list)   # (frame, from_team, to_team, strength)

    @property
    def switch_extra(self):
        return self.extra_ha, self.extra_ah


def frames_to_establish(config):
    """Contact frames an episode needs before its strength reaches `established_strength`."""
    return int(np.ceil(-config.episode_scale * np.log(1.0 - config.established_strength) - 1e-9))


def established_episodes(episodes, other_strength, config):
    """[(established_at, end, team_peak)] for episodes that reach established control.

    `established_at` is the frame of the episode's L-th contact frame, L = frames_to_establish: the
    moment control became established, not the first touch (a change of possession happens when a
    player *gains controlled* possession). An episode counts only if the other team's concurrent
    episode, at that frame, is not stronger than half of this one's.
    """
    need = frames_to_establish(config)
    out = []
    for start, stop, length, frames in episodes:
        if length < need:
            continue
        at = int(frames[need - 1])
        own = 1.0 - np.exp(-length / config.episode_scale)
        if other_strength[at] <= 0.5 * own:
            out.append((at, stop, own))
    return out


def handover_events(signals, config):
    """[(established_at_new, end_prev, from_team, to_team, strength)] in time order (0 = home, 1 = away)."""
    episodes = [(at, end, 0, s) for at, end, s in established_episodes(signals.episodes_home, signals.strength_away, config)]
    episodes += [(at, end, 1, s) for at, end, s in established_episodes(signals.episodes_away, signals.strength_home, config)]
    episodes.sort(key=lambda e: (e[0], e[2]))
    events = []
    for previous, current in zip(episodes[:-1], episodes[1:]):
        if previous[2] != current[2]:
            events.append((current[0], previous[1], previous[2], current[2], min(previous[3], current[3])))
    return events


def max_extra_cost(config):
    """Largest extra cost an unsupported home<->away switch may carry.

    `handover_extra * switch_cost`, capped so that a direct switch never costs more than the
    cheapest detour through the dead state (enter + exit) minus a margin of 1 nat. Without the cap a
    costlier direct switch makes it cheaper to change team via an invented stoppage, which the
    ablation showed: hundreds of phantom dead intervals appeared to hide handovers.
    """
    detour_headroom = config.enter_dead_cost + config.exit_dead_cost - config.switch_cost - 1.0
    return max(0.0, min(config.handover_extra * config.switch_cost, detour_headroom))


def handover_costs(signals, blocks, config=DEFAULT):
    """Handovers (extra switch costs and events) for the whole recording, per block."""
    n = len(signals.strength_home)
    support = np.zeros((2, n))                      # support[team][t]: support for switching *to* team at t
    events = []
    block_events = handover_events(signals, config)
    for a, b in blocks:
        for start, prev_end, from_team, to_team, strength in block_events:
            if not (a <= start < b):
                continue
            lo = min(max(start - config.handover_slack, prev_end - config.handover_pad), start)
            hi = min(start + config.handover_pad, b - 1)
            ramp = np.ones(hi - lo + 1)
            if start > lo:
                ramp[: start - lo + 1] = np.linspace(0.0, 1.0, start - lo + 1)
            support[to_team, lo:hi + 1] = np.maximum(support[to_team, lo:hi + 1], strength * ramp)
            events.append((int(start), int(from_team), int(to_team), float(strength)))
    scale = max_extra_cost(config)
    in_block = np.zeros(n, bool)
    for a, b in blocks:
        in_block[a:b] = True
    extra_ha = np.where(in_block, scale * (1.0 - support[1]), 0.0)    # entering away
    extra_ah = np.where(in_block, scale * (1.0 - support[0]), 0.0)    # entering home
    return Handovers(extra_ha=extra_ha, extra_ah=extra_ah, events=events)
