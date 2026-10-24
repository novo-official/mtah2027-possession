# Final risk acceptance record — RC1-FINAL

*Engineering readiness is not competitive accuracy.* This record says what has been verified (the package is correct, reproducible and compliant) and what has **not** (how
well it scores on the private full match, which is unknown). Passing the validator proves the file has the right shape; it says nothing about whether the labels are right.

## 1. Why RC1 was selected

Applying seven pre-declared engineering gates and a fixed priority order (correctness → robustness to a missing ball → temporal integrity → dead-ball reliability → possession structure →
reproducibility → public window (weak) → simplicity — see [RC1_DECISION_RECORD.md](RC1_DECISION_RECORD.md)), RC1 failed one gate; the movement-based dead-ball variants failed two or three.
It has no state change that rests on a missing ball, the fewest short dead intervals, the smallest parameter sensitivity (≤ 0.8 point of dead share under ±25% cost changes), the same CPU cost,
and a single dead-ball cue that is physical (ball tracked outside the lines). It is **not** claimed to be the highest-accuracy model; on the public window, the dead-ball variants score higher (macroF1 0.69–0.71 vs 0.57).

## 2. Why it is reproducible and stable

Deterministic NumPy float64 code; no randomness, network, GPU, video or downloaded model. Pinned configuration fingerprint and output hash; every historical configuration's output is re-derived byte for byte by the
test suite. Two independent runs, and a run from a source-only copy with an empty environment, give the identical SHA-256 `a9be383b…37f1be`. Stable under ±25% cost changes (≥ 0.992 agreement), 25% player dropout (0.872) and 20% extra ball dropout (0.900).

## 3. Errors that remain likely

| # | risk | evidence | consequence |
|---|---|---|---|
| R1 | **Dead-ball under-detection** | public dead recall ≈ 6% (none of 4 stoppage starts or ends matched); 10% of graded frames labelled dead, whereas the scoring documentation says dead ball is a large share of frames | stoppages with the ball in view are labelled live; possessions spanning a stoppage merge or are mis-split; macroF1's dead term is low |
| R2 | **Dead-state persistence** | the corrected decoder carries a dead state, entered through the out-of-play cue, through ball gaps: up to ≈ 139 s; 75% of dead frames evidence-free; ≈ 5.8% of graded frames in dead runs > 30 s | if the carry is wrong, live play is labelled dead for up to minutes; bounded, partly corroborated by player movement and by where gaps begin (12 of the 14 gaps that begin with the ball outside the lines are labelled mostly dead), not verified |
| R3 | **Missed / mis-timed hand-overs** | 4 of 12 direct H↔A boundaries matched at ±25 frames (public window); transition F1 0.21 | hand-overs are placed where the ball is re-observed, often late; spells straddling a gap are mis-bounded |
| R4 | **Long ball-observation gaps** | ball untracked on ≈ 41% of graded frames, 132 gaps ≥ 5 s, up to 221 s | the last team keeps the ball through the gap; ownership is never re-estimated inside it |
| R5 | **Documented pre-kickoff period not special-cased** | the organisers state the 2,934 graded frames 110808–113741 are all `D`; RC1 labels 2,246 of them `H`/`A` (688 `D`) | ≈ 1.4% of graded frames are known to be wrong, and a spurious possession may be created before the second-half kickoff. **Avoidable but not fixed here:** doing so changes the prediction file and its hash and adds a rule derived from public documentation, which needs the team's decision (see §7) |
| R6 | No independent full-match ground truth | public window: 17 possessions, 4 stoppages, reused many times | generalisation cannot be quantified |
| R7 | Floating-point reproducibility | byte identity verified for Python 3.14.8 / NumPy 2.5.4 only | another NumPy build could in principle change a tie; the expected hash is stated for the pinned set |

## 4. Why the alternative dead-ball variants were not promoted

* `corrected_dead` / `corrected_dead_hysteresis` (Phase 4.3 movement module): removing 25% of player observations adds ≈ 9–10 points of dead share and drops agreement to 0.78; 41–46% of dead intervals shorter than 3 s;
  dead share moves −5.6/+8.4 points for a ±0.1 change of one reference; a stoppage with the ball hidden is detected 0%.
* `oa_only` / `oa_restart` (Phase 4.6): best public dead F1 (0.47) and stable under dropout, but 28% of dead-boundary changes have no ball within ±1 s, 51% of the frames in ball gaps are called dead, its single policy parameter sits
  on a cliff (±0.1 → +3.4/−6.6 points of dead share) and its 24% dead share cannot be checked.
* They remain selectable (`--preset corrected_dead`, `--preset oa_only`), documented, tested and byte-reproducible. Choosing one is a legitimate alternative **if the team accepts its documented gate failures**; that choice was reserved for the user and not made here.

## 5. The rejected Smart Tagging experiment

A bounded team-phase prior from three eligible attack/defend tag pairs (6 nats per pair, half a hand-over; pre-registered before any scoring) was implemented as an opt-in module and **rejected**
([PHASE_4_9_SMART_TAGGING_REPORT.md](PHASE_4_9_SMART_TAGGING_REPORT.md)): the tags largely restate what the tracking already shows (82% of tagged frames already have tracking evidence); set pieces cannot time restarts (54% have no team, ±30 s timing);
with the ball hidden the prior gave no gain (better in 1 of 5 seeds, mean −0.0005) and wrong tags cost 3–4 points; it created unsupported state changes (RC1: 0; candidate: 1 H↔A change without a nearby ball, 2 inside gaps, 3–5 under every
degradation); the public window was unchanged (macroF1 0.568, spellF1 0.457). RC1 is unaffected; `rc2_tag_experimental` is kept, off, for the record.

## 6. What is unknown

The private full-match macroF1 and spellF1 of RC1 (and of every alternative); whether the dead-state carry is net positive or negative; the true dead-ball share of the match; how the model behaves on another match. Everything reported on the
full match is a label-free *behaviour description*. The only known full-match scores are the organisers' for their baselines (`baseline_smooth50`: spellF1 0.2433, macroF1 0.6724); RC1 may score above or below them.

## 7. Decisions deliberately left to the team

1. **Accept RC1 as submitted** (this record), or consciously prefer a dead-ball variant despite its gate failures.
2. **R5 (pre-kickoff frames):** keep RC1 byte-identical, or authorise a new, separately versioned release that forces the documented all-`D` period (a documented rule taken from the public FAQ/DATA text — not private data, but a rule keyed to a frame range, which the
   project had so far avoided) and re-run the whole verification. Not done here: the instructions for this phase forbid changing RC1's prediction semantics.
3. The compliance wording about the use of an AI coding assistant (technical report §12) must be confirmed against the enrolment terms.

*Verified readiness is engineering readiness: correct format, deterministic regeneration, no restricted data in the package. It does not predict the leaderboard.*
