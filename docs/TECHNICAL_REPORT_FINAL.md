# Possession inference from automatic tracking — technical document

**Team:** [REGISTERED TEAM NAME — to be filled in before submission] · **Release:** `RC1-FINAL` (preset `rc1`) · More than a Hack 2027, Phase 1

This document describes the method that produces the submitted prediction file. Numbers on the public five-minute window are diagnostics, not an
estimate of the private full-match score, which is unknown to us.

## 1. Problem definition

For each of the 199,319 frames of the tactical video, output `H` (home in possession, ball alive), `A`, `D` (dead ball) or `X` (abstain). The ranking metric
(**spellF1**) matches predicted possessions — maximal runs of `H` or `A` — one-to-one with the true ones by temporal IoU and averages the F1 over IoU 0.3–0.7;
**macroF1** is the frame-level mean F1 of H, A, D. Because the score counts possessions, not frames, temporal structure matters: per-frame decisions flicker,
and a stoppage (`D`) separates two possessions. The method is therefore an evidence-accumulating *state* model, not a frame classifier.

## 2. Data sources

| asset | used by the final model | note |
|---|---|---|
| tactical tracking (ATD `*_atd.txt`) and metadata (`*_atd.xml`) | **yes** | 199,319 records; 46 slots (tracklets, not people) + ball; the first `<Team>` is home |
| public 5-minute ground truth | diagnostics only | never used to fit or select anything; 17 possessions, 4 stoppages |
| tactical video, homography | explored, **not used** | image motion and restart cues were redundant with tracking, dominated by camera motion, and 200–5000× more expensive |
| Smart Tagging | explored, **not used** | a bounded team-phase prior added no independent value (§7) |
| broadcast video / tracking / homography | not used | separate timeline, not aligned to the tactical video |

The private full-match ground truth was not available and not used. No label was produced or corrected by hand.

## 3. Preprocessing

* **Frame alignment.** Row *i* of the output is frame *i* of the tactical video. The ATD counts `frame_id` from 1 (`video_frame = frame_id − 1`); the loader rejects any
  file whose ids are not exactly 1…N. The graded halves are video frames 11185–90651 and 110808–190197 (79,467 + 79,390 = 158,857 frames); all other frames are `X`
  (they are never scored). Each half is processed independently — no velocity, window, score or state crosses halftime.
* **Coordinates.** Positions are stored as pitch fractions and converted axis by axis to metres on the 105.3 × 68.0 m pitch. A non-finite coordinate, or one more than
  50 m outside the pitch, is a tracker error and becomes `NaN` (both axes together). Missing is `NaN`, never 0.
* **Team mapping.** The first `<Team>` of the metadata is home (FIFA/EPTS convention); team 0 = home, 1 = away.
* **Velocities.** Central differences over ±4 frames (0.32 s): `v = (x[t+4] − x[t−4]) · 25 / 8`. A velocity above 45 m/s (ball) or 12 m/s (player) is a position jump (identity
  switch, re-acquisition) and is discarded, not clipped. The tracker's own speed column (91% empty) is not used.
* **Reliability.** Every derived quantity carries its availability; an unavailable quantity contributes exactly zero evidence. Windows are computed inside a half only.

## 4. Ownership inference

For a frame with the ball and both teams tracked, let `d_h`, `d_a` be the distances (m) from the ball to the nearest home and away player, `margin = d_a − d_h`
(> 0 favours home), `v_rel` the speed of the ball relative to the nearest player, and `n_h`, `n_a` the numbers of validly tracked players:

```
claim       = tanh(margin / 3.0)
proximity   = exp(−(min(d_h, d_a) / 4.0)²)
motion      = 1 / (1 + (v_rel / 6.0)²)                 (0.7 if v_rel is unavailable)
observation = min(1, min(n_h, n_a) / 8)
consistency = 0.5 + 0.5 · |Σ sign(claim)| / count         over ±12 frames
weight      = proximity · motion · observation · consistency
own_llr     = 3.0 · weight · claim                       evidence units, > 0 favours home
```

Without a ball, or without a tracked player in one team, `weight = own_llr = 0`: absence is not evidence. Evidence units are uncalibrated, additive log-likelihood-ratio-like
scores and are never interpreted as probabilities. None of the constants was fitted; each is a round, physically motivated value (reach of a defender, sprint speed, how long a team needs the ball),
listed with its reason in `solution/config.py`.

## 5. Temporal decoding

States H, A, D. Per frame the emissions are `[own_llr/2, −own_llr/2, dead_llr]`. A log-domain float64 Viterbi decoder maximises the summed emissions minus transition costs:
home↔away **12** nats (≈ 8 frames of strong, consistent evidence), alive→dead **10**, dead→alive **6**, staying is free, the initial prior is uniform. All costs are finite: an unsupported
change is penalised, never forbidden. A configuration is accepted only if a direct hand-over is cheaper than the detour through the dead state (checked at start-up and by a test).

* **Missing evidence.** A frame with no evidence has emission 0 for every state; it adds the same to all paths, so the decoder carries its current state through the gap and leaves it only
  when later evidence outweighs the transition cost. A missing ball never means dead and never favours a team.
* **Corrected tie-breaking.** Plain Viterbi resolves exact ties to the lowest state index (H < A < D), which makes switch placement across an evidence-free stretch asymmetric (A→H, D→H, D→A at the
  start; H→A, H→D, A→D at the end). The final model prefers the switch among equally good predecessors (`decoder_late_ties`), so every switch lands at the *latest* of the equally good times —
  after the stretch, when evidence returns. The rule is mirror-symmetric in the teams, changes nothing when there are no ties, and a half with no evidence at all is written `X` (not `H`). All six state pairs are tested.
* **Half independence.** Each half is decoded on its own; decoding half 2 alone gives exactly the same labels as decoding it with half 1 present (tested on synthetic and real data).

## 6. Dead-ball handling

Exactly one dead-ball cue is enabled: the ball **tracked** more than 1 m outside the pitch rectangle adds 2.0 evidence units per frame to D. Ball absence is never a cue. The movement-based
dead-ball models, restart evidence and duration models described in §7 are **disabled** in the final model. Consequently the model calls a stoppage dead essentially only when the ball visibly
leaves the pitch, and a dead state, once entered, is held through a following evidence-free stretch until ownership evidence returns (§10).

## 7. Model development (what was tried, and why it was not selected)

| investigated | finding | decision |
|---|---|---|
| player-movement dead-ball detector (median player speed relative to the local live-play level) | public dead F1 0.11 → 0.41, but removing 25% of player observations adds ≈ 10 points of dead share, 46% of dead intervals last < 3 s, dead share moves −5.6/+8.4 points for ±0.1 on one reference, and a hidden ball over a stoppage is detected 0% | experimental, not selected |
| observation-aware dead-ball fusion | best public dead F1 (0.47) and stable under dropout, but 28% of its dead-boundary changes have no ball within ±1 s, the zero-point policy sits on a cliff (±0.1 → +3.4/−6.6 points of dead share), 24% dead share unverifiable | experimental, not selected |
| contact episodes, relative-motion/track-stability ownership, time-varying handover costs | no defensible gain; added short possessions | rejected |
| soft dead-duration decoder, hysteresis | lower or neutral spellF1; hysteresis does not repair the dropout sensitivity | rejected / experimental |
| video motion and restart features | camera motion confounds everything; homography stabilisation insufficient; restart cues redundant with tracking | no-go |
| **Smart Tagging team-phase prior** (bounded, 6 nats per pair, three eligible attack/defend pairs) | no benefit when the ball is hidden (agreement with the unmasked output 0.7550 vs 0.7555, better in 1 of 5 seeds); introduces unsupported changes (0 → 1 H↔A change without a ball nearby, 2 inside gaps; 3–5 under every degradation); wrong tags move the dead share by up to +3.8 points | **rejected** |

The final model was chosen by seven pre-declared engineering gates (decoder correctness, neutrality to ball dropout, robustness to player dropout, no unsupported dead persistence, dead-interval structure,
parameter stability, transition support), not by the public score. It fails one (dead persistence); the alternatives fail two or three. The code of every tried component is in the repository, switched off.

## 8. Evaluation

Official scorer, public window (7,500 graded frames, 17 possessions, 4 stoppages), bootstrap 5–95 percentile, 1000 draws:

| | macroF1 | spellF1 | dead F1 | transition F1 (±25 frames) | possessions (truth 17) |
|---|---|---|---|---|---|
| this model | 0.569 [0.44, 0.65] | 0.457 [0.30, 0.62] | 0.11 | 0.21 | 18 |
| organisers' `baseline_smooth50` | 0.621 [0.46, 0.74] | 0.326 [0.21, 0.45] | 0.32 | 0.08 | 26 |

These are **not** estimates of the private score: the window has been analysed repeatedly, the intervals overlap, and the organisers show that the ordering of baselines on this window differs from the
full match. The only known full-match scores are the organisers' for their baselines (`baseline_smooth50`: macroF1 0.6724, spellF1 0.2433). We make no claim to exceed them.

## 9. Robustness

Label-free, full match, five fixed seeds, agreement of the perturbed output with the unperturbed output:

| perturbation | agreement | change in dead share | possessions |
|---|---|---|---|
| +20% of visible ball frames hidden (random 10 s blocks) | 0.900 ± 0.006 | −0.9 pt | −46.8 |
| 25% of every player slot's observations removed (10 s blocks) | 0.872 ± 0.021 | +0.5 pt | +22.6 |
| identity discontinuities (~30/min, 5–15 m for 3–10 s) | 0.966 | — | — |
| ±25% on the switch/enter/exit costs and the out-of-play gain | ≥ 0.992 | −0.3…+0.8 pt | 256–309 |

Hiding the ball over live public-window frames raised false stoppages from 1.5% to 0.0% (never up): a missing ball does not create a stoppage. State changes never rest on a missing ball (0 hand-overs without a
ball observation within ±1 s, 0 changes inside a ≥ 5 s gap). The drop in possessions under ball dropout reflects possessions merging while the state is carried.

## 10. Limitations

* **Dead ball is under-called.** Public-window dead recall ≈ 6% (none of the 4 stoppage starts and none of the 4 stoppage ends matched within ±25 frames): stoppages with the ball in view are labelled live. The model calls 10% of graded frames dead.
* **Dead-state persistence.** A dead state entered through the out-of-play cue is carried through ball-tracking gaps (longest evidence-free stretch ≈ 139 s; ≈ 5.8% of graded frames lie in dead runs longer than 30 s).
  Partly corroborated by independent player movement and by where gaps begin, not verified against labels.
* **Missed and mis-timed hand-overs.** 4 of 12 direct H↔A boundaries matched within ±25 frames on the public window; hand-overs are placed only where the ball is re-observed.
* **Long ball gaps.** The ball is untracked on ≈ 41% of graded frames (132 gaps of ≥ 5 s, up to 221 s); ownership is carried, never re-estimated, inside them.
* **Documented pre-kickoff period.** The organisers state that the 2,934 graded frames of the second half before its kickoff (video frames 110808–113741) are all `D`. The final model does not special-case this period and labels
  most of it as live play (2,246 of the 2,934 frames) — a known avoidable error of about 1.4% of graded frames.
* **No independent full-match ground truth;** potential generalisation failures cannot be excluded or quantified.

## 11. Reproducibility

Python 3.14.8, `numpy==2.5.4` (`pandas==3.0.6` only for the organisers' scorer); CPU only; no network. The expected output hash is `a9be383bfa867ae034432d7746894951c89f6f9cd56ad8c9264eb9a5dd37f1be`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# unpack the organiser data into ./data (or set MTAH_DATA_DIR)
.venv/bin/python -m solution.predict --preset rc1 --out outputs/possession.csv      # ≈ 3 s, ≈ 640 MB
.venv/bin/python evaluation/validate_submission.py outputs/possession.csv
sha256sum outputs/possession.csv
.venv/bin/python -m unittest discover -s tests
```

The preset is frozen: its configuration fingerprint (`1e2dbb95abbe735a…`) and its output hash are pinned in the test suite together with the output hashes of every historical configuration. Two independent runs, and a run from a
source-only copy with an empty environment, give identical bytes.

## 12. Compliance

All labels are produced by the algorithm above; none was written or corrected by hand, and no hard-coded answer for any frame exists in the code. The private ground truth was not used; the public five minutes served only as a
diagnostic. The match data (tracking, video, tagging, homography) is not redistributed: the public repository contains source, tests, documentation and the organisers' public files only. All processing of the restricted
data was done locally; no video, tracking record, tagging file or frame-level export was uploaded or sent to any third-party service, and the development tooling was shown aggregate statistics only. The code was written with the help of an
AI coding assistant. [TEAM TO CONFIRM this statement matches the terms accepted at enrolment before submission.]
