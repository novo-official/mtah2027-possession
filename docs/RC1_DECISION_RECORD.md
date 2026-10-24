# RC1 decision record (Phase 4.8)

Status: **RC1 selected — `rc1`, with documented warnings.** Nothing here was submitted. All numbers are aggregates; no
restricted record is reproduced. The evidence scripts are in [`experiments/rc1_selection/`](../experiments/rc1_selection/)
(they print aggregates only; their raw output stays in the git-ignored `outputs/rc1/selection/`).

## 1. Decision

| | |
|---|---|
| **Primary (release candidate)** | **`rc1`** — Phase 4.2 ownership evidence + the physical out-of-play cue + the corrected (symmetric tie) decoder. No movement-based dead-ball model. |
| **Conservative fallback** | **Official `baseline_smooth50`** (`evaluation/results/baseline_smooth50.csv`, SHA-256 `6d4e04b8…2ab4b9`). Not produced by `solution/`; it is the only candidate whose full-match score is known (macroF1 0.6724, spellF1 0.2433, as published in `EVALUATION.md` and the `starter/baseline.py` docstring). It needs no research to select: it is committed and byte-reproducible. |
| **In-pipeline alternates** | `corrected_dead`, `oa_only` — experimental, *not* endorsed (section 4). Selectable with `--preset` without rerunning research. |
| Rejected as release candidates | `legacy_p42`, `legacy_p43` (known tie defect), `corrected_dead`, `corrected_dead_hysteresis`, `oa_only`/`oa_restart` (gate failures below), all Phase 4.4–4.6 options already rejected. |

`rc1` produces the same output as the historical `corrected` preset (SHA-256 `a9be383b…37f1be` — see the manifest). It is a
separately named, separately pinned configuration so that the release candidate cannot change when an experimental preset
is edited.

## 2. What was weighed, in the order the task fixes

1. **Correctness.** The Phase 4.2 decoder has a demonstrated asymmetry (lowest-state-index tie-breaking places A→H, D→H
   and D→A switches at the *start* of an evidence-free stretch). A regression test reproduces it (`test_legacy_tie_rule_moved…`).
   All `corrected*`, `oa_*` presets carry the fix; the legacy ones are kept only as byte-exact references.
2. **Robustness to missing ball.** Ball gaps ≥ 5 s cover 40.6% of the graded frames (132 gaps, longest 221 s).
3. **Temporal integrity.** No state change may rest on an unobserved ball.
4. **Dead-ball reliability.** No extreme or unsupported dead calls.
5. **Possession structure.** No fragmentation / under-segmentation.
6. **Reproducibility, parameter stability.**
7. **Public window — a weak diagnostic only.** 17 possessions, 4 stoppages, investigated many times over; it is *not* a validation set.
8. **Simplicity.**

### Engineering gates

Fixed after the first comparison pass and before selection. They are round engineering thresholds, **not** tuned on the public
window and not on labels. (Sensitivity perturbations are a check, never a search: no perturbed configuration was adopted.)

| gate | requirement |
|---|---|
| G1 decoder correctness | corrected tie semantics |
| G2 ball-absence neutrality | +20% extra ball dropout: dead share moves ≤ 2 pts, frame agreement ≥ 0.85; hiding the ball over public alive frames raises false-D by ≤ 3 pts |
| G3 player-track robustness | −25% player observations: dead share moves ≤ 2 pts, agreement ≥ 0.85 |
| G4 no unsupported dead persistence | ≤ 25% of D frames evidence-free; no evidence-free D stretch > 60 s |
| G5 dead structure | ≤ 35% of dead intervals shorter than 3 s |
| G6 parameter stability | ±25% on the decoder costs and one step on the model's main dead-ball parameter: dead share moves ≤ 5 pts, agreement ≥ 0.90 |
| G7 transition support | ≤ 10% of dead-boundary changes lack a ball observation within ±1 s; no H↔A change without one |

### Results (full-match numbers are label-free *behaviour*, never accuracy)

| | `rc1` (= corrected) | `corrected_dead` | `corrected_dead_hysteresis` | `oa_only` | `legacy_p42` |
|---|---|---|---|---|---|
| **G1** ties | ✅ | ✅ | ✅ | ✅ | ❌ asymmetric |
| **G2** +20% ball dropout | ✅ ΔD −0.9 pt, agree 0.900; false-D hidden 0.0% | ✅ −1.1 pt, 0.908 | ✅ −1.6 pt, 0.897 | ✅ +0.4 pt, 0.944; false-D 3.1→5.8% (+2.7) | ✅ |
| **G3** −25% player obs | ✅ +0.5 pt, 0.872 | ❌ **+9.9 pt**, 0.775 | ❌ +8.7 pt, 0.783 | ✅ +0.6 pt, 0.868 | ✅ |
| **G4** dead persistence | ❌ **74.9%** evidence-free, longest **139 s** | ✅ 0% | ✅ 0% | ✅ 5.6%, 27 s | ❌ 62.8%, 39 s |
| **G5** dead intervals < 3 s | ✅ 10/42 (24%) | ❌ 46/101 (46%) | ❌ 34/82 (41%) | ✅ 24/107 (22%) | ✅ 14/42 |
| **G6** parameter stability | ✅ max 0.8 pt, agree ≥ 0.992 | ❌ −5.6/+8.4 pt (`relative_ref` 0.6/0.8) | not run | ❌ +3.4/−6.6 pt (`activity_zero` 0.2/0.4) | — |
| **G7** transition support | ✅ 0/83 dead, 0/227 H↔A | ✅ 13/201 (6%) | ✅ 8/163 (5%) | ❌ **59/212 (28%)** | ✅ |
| gates failed | **1 (G4)** | 3 | 2 (G6 not run) | 2 | 2 |

Public five-minute window (official scorer; 1000-draw bootstrap, 5–95 pct; overlapping intervals are **not** evidence of a difference):

| | macroF1 | spellF1 | dead P / R / F1 | transition F1 @25 | direct H↔A matched | dead entry / exit @25 | spells (GT 17) |
|---|---|---|---|---|---|---|---|
| baseline_smooth50 | 0.621 [0.46–0.74] | 0.326 [0.21–0.45] | 0.66 / 0.21 / 0.32 | 0.08 | 2/12 | 0/4 · 0/4 | 26 |
| baseline_smooth125 | 0.581 | 0.424 [0.27–0.59] | 1.00 / 0.08 / 0.15 | 0.17 | 3/12 | 0/4 · 0/4 | 16 |
| legacy_p42 | 0.544 [0.42–0.63] | 0.389 [0.24–0.54] | 0.65 / 0.06 / 0.11 | 0.16 | 3/12 | 0/4 · 0/4 | 18 |
| **rc1 / corrected** | **0.569** [0.44–0.65] | **0.457** [0.30–0.62] | 0.65 / 0.06 / 0.11 | 0.21 | 4/12 | 0/4 · 0/4 | 18 |
| corrected_dead | 0.685 [0.55–0.75] | 0.491 [0.34–0.64] | 0.66 / 0.30 / 0.41 | 0.23 | 3/12 | 2/4 · 0/4 | 18 |
| oa_only | 0.706 [0.50–0.84] | 0.485 [0.33–0.64] | 0.71 / 0.35 / 0.47 | 0.26 | 3/12 | 1/4 · 1/4 | 16 |

**Read this honestly:** on the public window `rc1` is *below* both dead-ball variants on macroF1 (0.57 vs 0.69–0.71) because it
recovers only 6% of the dead frames, and it is below `baseline_smooth50` on macroF1 (0.57 vs 0.62) while above it on spellF1
(0.46 vs 0.33). Those intervals overlap; no ranking is claimed. There is no private-match estimate for any of them.

Full-match label-free behaviour (graded halves, 158,857 frames):

| | possessions | mean / median (s) | D intervals | D share | D < 3 s | D > 30 s | home share of alive | H↔A changes |
|---|---|---|---|---|---|---|---|---|
| baseline_smooth50 | 594 | 7.2 / 4.6 | 133 | 45.0% | 29 | 15 | 57.0% | 461 |
| legacy_p42 | 270 | 21.9 / 12.8 | 42 | 6.7% | 14 | 4 | 54.3% | 227 |
| **rc1 / corrected** | **270** | 21.2 / 13.0 | 42 | 10.0% (h1 10.7%, h2 9.3%) | 10 | 6 | 51.5% | 227 |
| corrected_dead | 298 | 19.2 / 11.1 | 101 | 10.2% (6.0% / 14.3%) | 46 | 5 | 56.4% | 196 |
| corrected_dead_hysteresis | 287 | 20.4 / 11.6 | 82 | 8.0% | 34 | 2 | 56.6% | 204 |
| oa_only | 316 | 15.3 / 9.5 | 107 | 23.7% (22.3% / 25.2%) | 24 | 13 | 54.9% | 209 |

No possession count or dead share was targeted. (For scale only: the public window has 19.6% dead frames and a mean possession of
14.2 s; a five-minute sample says nothing reliable about a 106-minute match.)

Engineering cost (measured, same machine, one match): every preset 2.9–5.2 s wall, 638 MB peak; `rc1` 3.1 s, 628–643 MB. Dependencies
identical (NumPy). Complexity: `rc1` has no dead-ball tuning parameter beyond the out-of-play gain (2.0 nats) and margin (1 m).

## 3. Why `rc1`

Applying the fixed priority order: priority 3 (temporal integrity) separates the candidates — `rc1` has **zero** state changes
without a ball observation, `oa_only` has 59 of 212 dead-boundary changes with no ball within ±1 s. Priority 5 (structure) and 6/8
(stability, simplicity) also favour `rc1`: 270 possessions with the fewest short dead intervals, and a configuration that moves < 1 point
under ±25% cost changes, whereas the movement-based dead models move 3–8 points under a single ±0.1 policy step. Priority 7 (public
window) favours the dead-ball variants and is weak by construction. Priority 4 (dead-ball reliability) is mixed: `rc1`'s dead calls are
few and tied to the ball leaving the pitch, but mostly carried; `oa_only`'s are supported by movement but are 24% of the match and
87% inside ball gaps.

A second reason is experimental hygiene: Smart Tagging (Phase 4.9) is the first *independent* source of dead-ball evidence. It should
be tested against a reference that does not already contain a tuned movement-based stoppage model, so that its effect is attributable.

## 4. Rejected alternatives

* **`legacy_p42` / `legacy_p43`** — G1: unresolved tie defect. Kept byte-exact as references.
* **`corrected_dead` / `corrected_dead_hysteresis`** — G3: removing 25% of player observations adds ~9–10 points of dead share and drops
  agreement to 0.78; G5: 41–46% of dead intervals are < 3 s; G6: dead share moves −5.6/+8.4 points for `relative_ref` 0.6/0.8. With the ball
  hidden over a stoppage, detection is 0%. Hysteresis does not repair G3.
* **`oa_only` / `oa_restart`** — the strongest public-window dead-ball numbers and the most stable under degradation, but: G7 (28% of dead
  boundaries have no ball nearby), G6 (zero point 0.30 sits on a cliff: 0.2 → +3.4, 0.4 → −6.6 points), 51% of the frames inside ball gaps are
  called dead, and the 23.7% dead share cannot be checked. Not "automatically" promoted, as instructed; kept selectable.
* **Disabling the out-of-play cue** — evaluated as an ablation (not a candidate): macroF1 0.526, spellF1 0.446, D never predicted.
  The cue is worth keeping (public +0.04 macroF1, +0.01 spellF1; also physically principled).

## 5. The dead-state persistence question (G4)

The corrected decoder carries a state through an evidence-free stretch until evidence returns. For a dead state entered through the
out-of-play cue this means it can persist through a long ball-tracking gap: 74.9% of `rc1`'s D frames have neither ownership nor dead
evidence, the longest such stretch is 139 s, 57.9% of D frames sit in intervals > 30 s, and 18 of 42 dead intervals contain a ball gap ≥ 5 s.

Does that make `rc1` unsafe? Measured answer: **not demonstrably unsafe, not verified, bounded.**

* Bound: D is 10.0% of the graded frames; the share in intervals > 30 s is ≈ 5.8% of the match. That is the most mislabelled-by-persistence
  frames can be, and wrong or right it moves mainly one class.
* Physical coherence: ball-tracking gaps begin outside the lines in only 14 of 132 cases, and `rc1` labels 12 of those 14 as mostly-D
  (20 of 132 gaps are mostly-D overall) — the ball left the pitch and has not come back. The legacy tie rule labels only 8 of those 14 as mostly-D, because it leaves D at the start of a gap.
* Independent corroboration (a cue the `rc1` decoder never sees): players move slowly in 72.8% of the evidence-free D frames
  against a 48.8% base rate (36% on public-window alive frames, 87% on public-window dead frames); 13 of the 19 evidence-free D
  stretches ≥ 5 s with known movement have a median movement ratio below the 0.7 "slow" line (median of medians 0.56 against 0.71 over all graded frames). On the public window, the ball is untracked on 33.6% of dead frames but only 6.5% of alive frames.
* Not verified: no label of the full match exists; the corroboration is the same weak movement cue (AUC ≈ 0.85 on the public window)
  and is not accuracy. Eight of the 20 mostly-D gaps do not begin with the ball outside the lines and are the unsupported part.
* Pinned behaviour: `tests/test_rc1.py::test_out_of_play_ball_is_dead_and_the_state_is_carried_through_the_following_gap` documents the semantics, including the exit
  within 20 frames of the evidence returning, and that the legacy rule left earlier.

Accepted, with the fallback above and the Smart Tagging experiment as the independent check.

## 6. Remaining uncertainties and accepted risks

| risk | size / evidence | mitigation |
|---|---|---|
| Dead ball under-called while the ball is in view | public dead recall 6%; `rc1` macroF1 0.57 vs 0.69–0.71 for dead-ball variants; `EVALUATION.md` says dead ball "takes a far larger share of frames than of segments" and `rc1` calls 10% of the graded frames dead | known and accepted; the risk is real, not hypothetical — it is the main reason to look at the dead-ball variants again if Phase 4.9 gives them independent support |
| Dead state carried through ball gaps | up to 139 s, ≈ 5.8% of frames in > 30 s dead runs | bounded, partly corroborated, pinned by a test; fallback |
| Public window reused many times | 17 possessions, 4 stoppages | not used for selection; no tuning in Phase 4.8 |
| Ownership carried through ball gaps (41% of graded frames are inside ≥ 5 s gaps) | never re-estimated; handovers localised only when the ball returns | structural limit of tracking-only evidence |
| Direct handover localisation | 4/12 direct H↔A boundaries matched @25 on the public window; transition F1 0.21 | spellF1 is not very sensitive; Smart Tagging team phases are the proposed prior |
| No private full-match estimate | — | stated everywhere; RC1 makes no claim against the baseline |
| NumPy / platform float differences | byte identity guaranteed only for the pinned versions | `requirements-rc1.txt` (`requirements.txt` in the public package) |

## 7. Frozen vs. open

Frozen: the `rc1` configuration (fingerprint pinned in `tests/test_model_registry.py`), its output hash, the historical presets.
Open for RC2 only through the gates in [SMART_TAGGING_INTEGRATION_PLAN.md](SMART_TAGGING_INTEGRATION_PLAN.md).
