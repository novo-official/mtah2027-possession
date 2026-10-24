"""Dead-ball duration modelling: two practical mechanisms, both soft, both optional.

The first-order decoder (decoder.py) has no notion of how long a stoppage lasts: leaving the dead
state costs the same after one second as after one minute, so brief dead intervals appear whenever
the dead-ball score flickers. Two ways to express "stoppages usually last a while" without ever
forbidding a short one:

**A. Evidence-aware hysteresis** (`hysteresis_dead`). A Schmitt trigger on the rolling mean of the
dead-ball score. A stretch is *armed* once the mean reaches `hysteresis_enter` and stays armed
until it falls below `hysteresis_exit` (< enter). Outside armed stretches positive dead evidence is
attenuated (a lone spike cannot start a stoppage); inside them live-play evidence is attenuated
(small dips do not end it). It conditions the *evidence* the decoder sees; the decoder is unchanged.

**B. Soft duration-aware decoding** (`viterbi_min_duration`). The dead state is expanded into an
age chain D_1 ... D_K (K = `soft_min_dead_frames`, D_K meaning "K frames or more"). Entering D goes to
D_1, each frame advances one age, and leaving from age k < K costs the normal exit cost plus
`penalty * (1 - k/K)`. This is an explicit-duration (semi-Markov) prior restricted to the one
state that needs it, solved exactly by Viterbi on 2 + K states in O(T K); it equals the original
decoder when the penalty is zero. Home and away are interchangeable throughout.

Why these cannot create the H -> D -> A loophole (see `check_transition_safety`): the penalty only
ever *adds* cost to leaving D, so the cheapest indirect route from H to A (enter + exit + penalty)
is never cheaper than it was before, and `check_transition_safety` insists the direct switch
costs less than that route. A route through D is therefore chosen only when dead-ball evidence
pays for it.
"""

import numpy as np

H, A, D = 0, 1, 2


# -- A. hysteresis ---------------------------------------------------------------------------------

def _rolling_mean(values, halfwidth):
    """Centred mean over +/- halfwidth, windows shrinking at the edges."""
    n = len(values)
    cs = np.concatenate(([0.0], np.cumsum(values)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - halfwidth)
    hi = np.minimum(n, idx + halfwidth + 1)
    return (cs[hi] - cs[lo]) / (hi - lo)


def armed_mask(dead_llr, halfwidth, enter, exit_):
    """Boolean mask of armed stretches: maximal runs where the rolling mean is at least `exit_`
    and that reach `enter` somewhere. Requires exit_ <= enter."""
    if exit_ > enter:
        raise ValueError("hysteresis_exit must not exceed hysteresis_enter")
    mean = _rolling_mean(np.asarray(dead_llr, dtype=np.float64), halfwidth)
    above = mean >= exit_
    edges = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    armed = np.zeros(len(mean), dtype=bool)
    for a, b in zip(starts, stops):
        if mean[a:b].max() >= enter:
            armed[a:b] = True
    return armed


def hysteresis_dead(dead_llr, blocks, config):
    """Dead-ball score after hysteresis conditioning, block by block (no leakage across halves)."""
    out = np.array(dead_llr, dtype=np.float64)
    for a, b in blocks:
        x = out[a:b].copy()
        armed = armed_mask(x, config.hysteresis_halfwidth, config.hysteresis_enter, config.hysteresis_exit)
        x[~armed & (x > 0)] *= config.unarmed_positive_scale
        x[armed & (x < 0)] *= config.armed_negative_scale
        out[a:b] = x
    return out


# -- B. soft duration prior --------------------------------------------------------------------------

def exit_costs(config):
    """Cost of leaving the dead state at age k = 1 ... K (index k - 1)."""
    K = config.soft_min_dead_frames
    age = np.arange(1, K + 1, dtype=np.float64)
    return config.exit_dead_cost + config.short_dead_penalty * np.maximum(0.0, 1.0 - age / K)


def check_transition_safety(config):
    """Report whether an indirect route through D can undercut a direct home<->away switch.

    The cheapest route H -> D -> A (or A -> D -> H) pays `enter_dead_cost` once and the smallest
    exit cost; if that is not larger than `switch_cost`, an unsupported switch is cheaper when
    disguised as a stoppage, and phantom dead intervals appear. Returns a dict.
    """
    min_exit = float(exit_costs(config).min()) if config.duration_model in ("soft", "both") \
        else float(config.exit_dead_cost)
    detour = config.enter_dead_cost + min_exit
    return {"direct": float(config.switch_cost), "cheapest_detour": float(detour),
            "margin": float(detour - config.switch_cost), "safe": bool(detour > config.switch_cost)}


def viterbi_min_duration(emissions, cost, initial, K, penalty, late_ties=False):
    """Viterbi over H, A and a dead-state age chain D_1..D_K with an exit penalty that decays with age.

    Equivalent to decoder.viterbi when `penalty` is 0 (tested). Returns an int8 path of H/A/D codes.
    `late_ties` has the same meaning as there: among exactly tied predecessors prefer a switch.
    A path that starts in D starts at age K (the interval began earlier, so no early-exit penalty);
    a path that ends in D at any age pays no exit penalty (it never exits).
    """
    if K < 2:
        raise ValueError("soft_min_dead_frames must be at least 2")
    emissions = np.asarray(emissions, dtype=np.float64)
    T = emissions.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int8)
    c_hd, c_ad, c_ha, c_ah = cost[H, D], cost[A, D], cost[H, A], cost[A, H]
    ages = np.arange(1, K + 1, dtype=np.float64)
    exit_vec = cost[D, H] + penalty * np.maximum(0.0, 1.0 - ages / K)        # exit cost from age 1..K
    tol = 1e-9

    sH = initial[H] + emissions[0, H]
    sA = initial[A] + emissions[0, A]
    d = np.full(K, -np.inf)
    d[K - 1] = initial[D] + emissions[0, D]

    pH = np.zeros(T, dtype=np.int8)
    pA = np.zeros(T, dtype=np.int8)
    kx = np.zeros(T, dtype=np.int16)          # best age to exit from (shared by H and A targets)
    p1 = np.zeros(T, dtype=np.int8)           # D_1 entered from H (0) or A (1)
    pK = np.zeros(T, dtype=np.int8)           # D_K reached from D_{K-1} (1) or from D_K (0)
    nd = np.empty(K)

    def pick(values, late, self_index):
        """Index of the best of the three predecessor candidates (H, A, D) under the tie policy:
        legacy = first maximum; late = a switch (any other state) preferred to staying on exact ties."""
        best = max(values)
        if not late:
            return values.index(best)
        tied = [i for i in range(3) if values[i] >= best - tol]
        return next((i for i in tied if i != self_index), self_index)

    for t in range(1, T):
        ex = d - exit_vec
        k = int(np.argmax(ex))
        mx = ex[k]
        kx[t] = k
        iH = pick((sH, sA - c_ah, mx), late_ties, H)
        iA = pick((sH - c_ha, sA, mx), late_ties, A)
        pH[t], pA[t] = iH, iA
        nH = (sH, sA - c_ah, mx)[iH] + emissions[t, H]
        nA = (sH - c_ha, sA, mx)[iA] + emissions[t, A]
        # entering the dead state from H or A (lowest index wins an exact tie, as in the original)
        from_h, from_a = sH - c_hd, sA - c_ad
        p1[t] = 0 if from_h >= from_a else 1
        nd[0] = max(from_h, from_a) + emissions[t, D]
        nd[1:] = d[:-1] + emissions[t, D]
        # D_K absorbs: reached from D_{K-1} (age grows) or by staying (exact ties: stay)
        pK[t] = 0 if d[K - 1] >= d[K - 2] else 1
        nd[K - 1] = max(d[K - 1], d[K - 2]) + emissions[t, D]
        sH, sA = nH, nA
        d, nd = nd, d

    # final state: best of H, A and every age of D (index order H, A, D_1, ..., D_K)
    final = np.concatenate(([sH, sA], d))
    s = int(np.argmax(final))
    if s == 0:
        state, age = H, -1
    elif s == 1:
        state, age = A, -1
    else:
        state, age = D, s - 2
    path = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = state
        if t == 0:
            break
        if state == H or state == A:
            p = pH[t] if state == H else pA[t]
            if p == D:
                state, age = D, int(kx[t])
            else:
                state, age = int(p), -1
        else:
            if age == 0:
                state, age = (H if p1[t] == 0 else A), -1
            elif age < K - 1:
                age -= 1
            elif pK[t] == 1:
                age = K - 2
            # else: stays in D_K
    return path
