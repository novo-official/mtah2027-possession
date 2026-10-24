# Phase 4.9 — pre-registration of the Smart Tagging hypothesis

Written **before** any candidate output was produced and before any public-window score of a tag-based model was computed. It is based on a
metadata / semantic audit (`python -m experiments.smart_tagging.audit`: counts, durations, polarity, coverage, and agreement with the
*tracking-only* RC1 ownership evidence on confident frames — no ground truth of any kind). RC1 (`a9be383b…37f1be`) is the sole control and
is not modified.

## 1. What the audit found (aggregates)

* 638 instances, all intervals valid (end > start), every attack, defend, set-piece and goal instance lies in a graded half (20 zone-code instances do not); the official helper confirms the
  time base (both kickoffs fall inside their tagged windows). Conversion is `round(seconds × 25)`; no offset is applied.
* **Polarity.** `Team` = the club of the *code's subject*. Attack-side codes (`BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `LONG BALL`,
  `CREATING CHANCES`) name the team in possession; defend-side codes (`DEFENDING IN ATTACKING THIRD`, `…MIDDLE THIRD`, `…DEFENSIVE THIRD`,
  `DEFENSIVE TRANSITION`) name the *defending* team. 142 attack instances have a defend instance on the **identical interval**; in 140 of
  them the two teams are opposite — a pair is one fact stated twice, not two votes. The only unpaired attack instances are the 29 `LONG BALL`.
* **Agreement with the team the tracking names** on confident tracking frames (23% of graded frames; label-free): `BUILD UP` 0.878,
  `PROGRESSION` 0.850, `ATTACKING TRANSITION` 0.747, `LONG BALL` 0.681, `CREATING CHANCES` 0.617; the paired defend codes mirror this
  (0.12–0.38 before polarity inversion). Zone codes (`BALL IN FINAL THIRD` 0.70, `BALL IN THE BOX` 0.63, `PLAYERS IN THE BOX` 0.59) have unclear team semantics → **not used**.
* **Coverage.** Team-resolved attack/defend intervals cover 44.5% of graded frames (40.1% by ≥ 2 overlapping instances, i.e. mostly the pair).
  Tagged frames have tracking ownership evidence far more often than untagged ones (82% vs 41%): the tags largely describe stretches the tracking
  already sees, so little complementary information in ball gaps is expected.
* **Set pieces (59).** Fixed ≈ 35 s windows; `Team` is `N/A` on 32 of 59 (every throw-in); the interval start lies a median of +2.8 s (quartiles −20 s … +36 s) and the end +21 s (−25 s … +49 s) from the
  nearest RC1 dead-interval start. **They do not support restart timing and 54% carry no team.** The restart
  mechanism is therefore **rejected before evaluation**.

## 2. The one hypothesis

> A bounded **team-phase prior** built from the three *eligible phase pairs* tells the decoder which team is in control during stretches where the
> tracking evidence is weak or absent, and can improve the placement of hand-overs and the resilience to hidden-ball intervals **without** creating
> hand-overs or stoppages the tracking does not support.

Prior-independence claim to be tested rather than assumed: the prior must carry information the tracking-only RC1 lacks. That is checked with a
masking experiment (section 5), not with the public window.

## 3. The prior (frozen)

| element | specification |
|---|---|
| eligible pairs | (`BUILD UP`, `DEFENDING IN ATTACKING THIRD`), (`PROGRESSION`, `DEFENDING IN MIDDLE THIRD`), (`ATTACKING TRANSITION`, `DEFENSIVE TRANSITION`). Rule: attack-side code with a defend-side partner on the identical interval, and tracking agreement ≥ 0.70 (audit). Excluded: `LONG BALL` (unpaired, 0.681), `CREATING CHANCES` / `DEFENDING IN DEFENSIVE THIRD` (0.617), zone codes, `SET PIECES`, `GOALS` |
| team polarity | attack instance `Team` → that club in control; defend instance `Team` → the **opposite** club in control. Club names map to H/A through configurable name lists (`barcelona` → home; `atletic club` / `athletic club` → away), case-insensitive |
| one vote per pair | a pair yields one side. Attack and implied-defend sides must agree if both are known; if they disagree, or neither is known (`N/A`), the pair contributes **zero**. Duplicate instances of the same (interval, pair) count once |
| interval | the tagged interval `[round(25·start), round(25·end))`, assigned to the half named by its `Half` tag and clipped to that half's graded block; invalid timestamps, intervals shorter than 2 s or longer than 60 s, or not overlapping their half → zero. No offset, no shifting |
| bounded contribution | each pair carries a **total budget of `0.5 × switch_cost` = 6.0 nats** spread uniformly over its clipped interval (`6.0 / L` per frame, `+` home, `−` away). Long intervals therefore cannot accumulate more than 6.0 nats |
| overlap / conflict | per frame, the largest same-side contribution counts (overlapping agreeing pairs are not summed); opposite-side contributions cancel |
| where it enters | `own_llr += prior` (the existing ownership channel; emissions `[own/2, −own/2, dead]`). The H↔A advantage a pair can supply is ≤ 6.0 nats, **strictly below the 12-nat hand-over cost**, so a tag alone can never pay for a switch; tracking evidence must supply at least the other half. Dead neutrality: see the amendment below |
| zero contribution when | file missing / unparsable (error, never a silent fallback inside the experimental preset), team `N/A` or unrecognised, pair conflicting, ineligible code, frame outside graded halves, no tag covering the frame |
| decoder | unchanged (`decode_blocks`, `viterbi`, costs). Only `predict_codes` gains an optional `external_own` argument (default `None` ⇒ byte-identical RC1) |
| why this strength | 6.0 = half a hand-over: the weakest rule that is not an arbitrary constant ("a tag supplies at most half of what a switch needs; the tracking must supply the rest"). It was **not** chosen from the public window and will not be changed after it is seen |
| effect of a false tag | bounded: ≤ 6.0 nats of misplaced preference per pair, ≤ 3.0 nats against the dead state; it can move a hand-over by a few frames or rescue/obstruct a weak-evidence decision but cannot flip a well-supported state (tested) |

Settings live in a separate dataclass (`TagConfig`), not in `Config`; the preset is `rc2_tag_experimental`; it uses the **RC1** `Config` unchanged.

## 4. Pre-registered promotion criteria (label-free first; the public window is last and non-decisive)

All must hold for PROMOTE; any structural failure means REJECT; otherwise EXPERIMENTAL.

1. All correctness tests pass; RC1 hash and every historical pin unchanged.
2. **No new unsupported changes:** H↔A changes with no ball within ±1 s = 0; changes inside ≥ 5 s ball gaps = 0 (RC1 values).
3. **No false stoppages / no dead drift:** |Δ dead share| ≤ 0.3 point and |Δ dead intervals| ≤ 2 versus RC1.
4. **Locality:** frames differing from RC1 ≤ 3% of graded frames; and tag-removal of a random 10% of eligible pairs changes ≤ 0.5% of graded frames (otherwise: fragile).
5. **Independent value (masking test):** hide the ball over random 10 s blocks (the documented RC1 protocol, 5 seeds); on the hidden frames compare agreement of
   [RC1-masked] and [candidate-masked] with the **unmasked RC1** output. The candidate must be better in ≥ 4 of 5 seeds, and the same test with the tags shifted by +90 s
   and with the team polarity flipped must **not** show that improvement (specificity control).
6. Robustness: paired stress tests (ball masking, player dropout, identity jumps, tag dropout/shift/conflict) show no more instability than RC1.
7. Determinism, official validator OK, identical SHA-256 on regeneration.
8. Public window: **one** evaluation of the frozen candidate; reported next to RC1; a gain alone never promotes, a loss beyond the bootstrap spread is a warning.

If the prior turns out to be negligible (criterion 5 shows no resilience gain), the decision is REJECT, with the negative result recorded.

## Amendment A1 (before any candidate evaluation): dead-state neutrality

While writing the tests, `test_tag_never_creates_a_stoppage_or_removes_one` failed on the first implementation: a bare `own_llr += prior` gives the
favoured team a bonus over the dead state (emissions `[own/2, −own/2, dead]`), and because a dead state carried through an evidence-free stretch is
held only by ties, even 3 nats pulled its exit forward (460 → 60 dead frames in the synthetic scenario). That contradicts the stated intent of section 3
("no dead evidence is added or removed") and would silently shorten stoppages. Correction, made before the candidate was run on the real match or scored
on any window: `predict_codes` adds `|prior|/2` to the dead channel whenever an ownership prior is supplied, so *favoured team versus dead ball* is exactly
unchanged and only the home/away decision moves. The disfavoured team versus dead ball loses at most 6.0 nats in total, below `enter_dead_cost` (10); criterion 3 (dead share /
dead intervals unchanged) is the check. No parameter changed; the budget (6.0 nats per pair) is untouched.
