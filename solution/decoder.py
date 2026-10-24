"""Three-state temporal decoder: H (home), A (away), D (dead ball).

A log-domain Viterbi over additive costs. Deterministic, O(T x 9) per half, float64 throughout.

Design points (see config.py for the numbers):

* **Emissions are separate from structure.** `build_emissions` turns the two evidence channels
  into a (T, 3) score matrix; `viterbi` knows nothing about football. A semi-Markov decoder can
  reuse the emissions unchanged and only replace the transition handling by duration terms.
* **Absence carries state.** A frame with no evidence has emission 0 for every state, so it adds
  the same to all paths and the decoder simply keeps its current state through the gap. It can
  still leave it when later evidence outweighs the transition cost. Missing never means dead.
* **Costs are penalties, not prohibitions.** Every transition is allowed. Home<->away hand-overs
  and alive<->dead changes have separate costs because they are different events.
* **Halves are independent.** Each block is decoded on its own with its own initial prior and a
  free final state; no score, state or velocity crosses the gap between them. Frames outside
  every block are labelled X (the scorer ignores them).
* **Ties** go to the lowest state index (H, then A, then D), at every step and at the end.
"""

import numpy as np

H, A, D, X = 0, 1, 2, 3   # the codes of evaluation/possession_metrics.py
LABELS = "HADX"
_TIE_TOLERANCE = 1e-9   # candidates within this of the best count as exactly tied (late_ties only)


def build_emissions(own_llr, dead_llr, emission_mode="legacy"):
    """(T, 3) per-state scores.

    legacy:        [own/2, -own/2, dead]. Equal, up to a per-frame constant that cannot change any
                   path, to [min(own,0), min(-own,0), dead - |own|/2]: the dead channel carries an
                   implicit live-play term of -|own|/2 that exists only while the ball is observed.
    mixture_null:  [min(own,0) + b, min(-own,0) + b, dead] with b = log 2 - log(1 + exp(-|own|)),
                   the log-likelihood advantage of the winning team over a dead state that explains
                   the ownership observation as an equal mixture of both teams. b is in [0, log 2)
                   and is 0 where there is no ownership evidence. H versus A is own in both modes.
    With no evidence at all every state scores 0.
    """
    own = np.asarray(own_llr, dtype=np.float64)
    dead = np.asarray(dead_llr, dtype=np.float64)
    if own.shape != dead.shape:
        raise ValueError("evidence channels differ in length: {} vs {}".format(own.shape, dead.shape))
    if emission_mode == "legacy":
        return np.stack([0.5 * own, -0.5 * own, dead], axis=1)
    if emission_mode != "mixture_null":
        raise ValueError("emission_mode must be 'legacy' or 'mixture_null'")
    bonus = np.log(2.0) - np.log1p(np.exp(-np.abs(own)))
    return np.stack([np.minimum(own, 0.0) + bonus, np.minimum(-own, 0.0) + bonus, dead], axis=1)


def transition_costs(config):
    """(3, 3) matrix, cost[i, j] = penalty for moving from state i to state j. Staying costs 0."""
    cost = np.zeros((3, 3), dtype=np.float64)
    cost[H, A] = cost[A, H] = config.switch_cost
    cost[H, D] = cost[A, D] = config.enter_dead_cost
    cost[D, H] = cost[D, A] = config.exit_dead_cost
    return cost


def _choose(candidate, late_ties):
    """Best predecessor per column. Default: the first maximum (lowest state index wins ties).
    With `late_ties`: among predecessors tied with the best, prefer a different state to staying,
    which reports a switch at the latest of equally good times, identically for both directions."""
    best_from = np.argmax(candidate, axis=0)
    if not late_ties:
        return best_from
    tied = candidate >= candidate.max(axis=0) - _TIE_TOLERANCE
    for j in range(3):
        for i in range(3):
            if i != j and tied[i, j]:
                best_from[j] = i
                break
        else:
            best_from[j] = j
    return best_from


def viterbi(emissions, cost, initial, late_ties=False):
    """Best state path (int8, length T) maximising sum(emission) - sum(transition cost) + initial prior.

    `emissions` (T, 3) float, `cost` (3, 3) >= 0, `initial` (3,) log-prior of the first state.
    `late_ties` breaks exact ties between "stay" and "switch" toward the switch, which places a
    switch at the latest of equally good times, identically for home->away and away->home.
    """
    emissions = np.asarray(emissions, dtype=np.float64)
    T = emissions.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    score = np.asarray(initial, dtype=np.float64) + emissions[0]
    back = np.zeros((T, 3), dtype=np.int8)
    for t in range(1, T):
        # candidate[i, j] = best score ending in i at t-1, then moving to j
        candidate = score[:, None] - cost
        best_from = _choose(candidate, late_ties)
        score = candidate[best_from, np.arange(3)] + emissions[t]
        back[t] = best_from
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_time_varying(emissions, cost, initial, extra_ha, extra_ah, late_ties=False):
    """Viterbi with per-frame extra cost for the two home<->away switches.

    Identical to `viterbi` except that entering away from home at step t costs
    `cost[H, A] + extra_ha[t]` and entering home from away costs `cost[A, H] + extra_ah[t]`
    (extra >= 0, indexed by the frame being entered). With all-zero extras the result equals
    `viterbi` exactly. Mathematically it is the same additive-log model with a time-dependent
    log-prior over when a hand-over occurs; the emissions are not touched, so no evidence is
    counted twice.
    """
    emissions = np.asarray(emissions, dtype=np.float64)
    T = emissions.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    score = np.asarray(initial, dtype=np.float64) + emissions[0]
    back = np.zeros((T, 3), dtype=np.int8)
    step_cost = np.array(cost, dtype=np.float64)
    base_ha, base_ah = cost[H, A], cost[A, H]
    columns = np.arange(3)
    for t in range(1, T):
        step_cost[H, A] = base_ha + extra_ha[t]
        step_cost[A, H] = base_ah + extra_ah[t]
        candidate = score[:, None] - step_cost
        best_from = _choose(candidate, late_ties)
        score = candidate[best_from, columns] + emissions[t]
        back[t] = best_from
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_blocks(own_llr, dead_llr, blocks, config, switch_extra=None):
    """Per-frame codes (int8, H=0 A=1 D=2 X=3) for the whole recording.

    Each block (start, stop) is decoded independently; frames outside all blocks are X.
    `switch_extra=(extra_ha, extra_ah)` (full-length arrays) selects the time-varying path used by
    the handover logic; None keeps the original decoding path exactly.

    `config.duration_model` in ("soft", "both") selects the age-chain decoder of duration.py.
    With `config.decoder_late_ties` (the corrected semantics) a block that carries no evidence at
    all is written as X: there is nothing to carry and no state to prefer, whereas the lowest-index
    rule would label it H.
    """
    n = len(own_llr)
    codes = np.full(n, X, dtype=np.int8)
    cost = transition_costs(config)
    initial = np.asarray(config.initial_logprior, dtype=np.float64)
    soft = config.duration_model in ("soft", "both")
    if soft and switch_extra is not None:
        raise ValueError("the soft duration decoder cannot be combined with time-varying switch costs")
    for start, stop in blocks:
        emissions = build_emissions(own_llr[start:stop], dead_llr[start:stop], config.emission_mode)
        if config.decoder_late_ties and not emissions.any():
            continue                                   # nothing observed anywhere in this block: X
        if soft:
            from .duration import viterbi_min_duration
            codes[start:stop] = viterbi_min_duration(emissions, cost, initial, config.soft_min_dead_frames,
                                                     config.short_dead_penalty, config.decoder_late_ties)
        elif switch_extra is None:
            codes[start:stop] = viterbi(emissions, cost, initial, config.decoder_late_ties)
        else:
            codes[start:stop] = viterbi_time_varying(emissions, cost, initial,
                                                     switch_extra[0][start:stop], switch_extra[1][start:stop],
                                                     config.decoder_late_ties)
    return codes
