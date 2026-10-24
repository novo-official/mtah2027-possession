# solution/ — possession inference (RC1 release candidate; Phase 4.2 decoder, 4.3 dead-ball module, 4.4 ownership / handover options, 4.5 decoder correctness and duration models, 4.6 observation-aware dead-ball evidence)

Tracking → per-frame geometry → soft evidence → three-state decoder. NumPy only, deterministic,
CPU only (about 3 s and 640 MB for the whole match).

## Production route — Release Candidate 1 (`--preset rc1`)

`rc1` is the single frozen configuration prepared for submission: ownership evidence + the physical out-of-play cue + the corrected
(symmetric tie) decoder, no movement-based dead-ball model. Everything else in this folder is experimental, rejected or a historical
reference and is documented as such in `model_registry.py`. Decision, evidence and accepted risks: [`docs/RC1_DECISION_RECORD.md`](../docs/RC1_DECISION_RECORD.md);
risk acceptance: [`docs/FINAL_RISK_ACCEPTANCE.md`](../docs/FINAL_RISK_ACCEPTANCE.md); technical document: [`docs/TECHNICAL_REPORT_FINAL.md`](../docs/TECHNICAL_REPORT_FINAL.md).

**Needs only public files (no restricted data):**

```bash
.venv/bin/python -m solution.predict --list-presets              # every registered preset and its status
.venv/bin/python -m solution.predict --describe rc1              # features on/off, limitations, config fingerprint, source digest
.venv/bin/python -m unittest discover -s tests                   # the data-dependent tests skip themselves
```

**Needs the organiser-provided dataset unpacked under `data/` (or `MTAH_DATA_DIR`) — restricted, never committed:**

```bash
.venv/bin/python -m solution.predict --preset rc1 --out outputs/rc1/possession_rc1.csv   # ~3 s, ~640 MB, CPU only, deterministic
.venv/bin/python evaluation/validate_submission.py outputs/rc1/possession_rc1.csv         # format check (no ground truth)
sha256sum outputs/rc1/possession_rc1.csv                                                  # a9be383b…37f1be
.venv/bin/python evaluation/run_evaluation.py --results outputs/rc1 --bootstrap 1000      # public five-minute window
```

The command prints the preset's config fingerprint, the label counts, runtime, peak memory and the SHA-256 of the file. It refuses to
write into `data/`, `drive-file/`, `evaluation/gt/`, `starter/` or over an official `baseline_*.csv`, writes atomically, and exits with
status 2 and a one-line message when the data is missing. Plain `python -m solution.predict` (no `--preset`) is the legacy Phase 4.2
default, kept so that earlier results stay reproducible — **it is not RC1**. Pinned environment: [`requirements.txt`](../requirements.txt)
(Python 3.14.8, NumPy 2.5.4; byte-identical output is guaranteed for those versions).

**Phase 4.9:** a bounded Smart Tagging prior (`smart_tagging.py`, preset `rc2_tag_experimental`, opt-in) was implemented, tested and **rejected**; RC1 is unchanged and remains the release candidate — [`docs/PHASE_4_9_SMART_TAGGING_REPORT.md`](../docs/PHASE_4_9_SMART_TAGGING_REPORT.md).

Selection evidence (aggregates only, rerunnable): `python -m experiments.rc1_selection.compare [--robust]`, `.persistence`, `.gaps`,
`.sensitivity`. Smart Tagging is **not** part of RC1; its integration plan is [`docs/SMART_TAGGING_INTEGRATION_PLAN.md`](../docs/SMART_TAGGING_INTEGRATION_PLAN.md).

## Regenerate the prediction file

From the repository root, with the official data installed under `data/` (see [DATA.md](../DATA.md)):

```bash
.venv/bin/python -m solution.predict                        # -> outputs/predictions/solution_decoder.csv
.venv/bin/python evaluation/validate_submission.py outputs/predictions/solution_decoder.csv
.venv/bin/python evaluation/run_evaluation.py --results outputs/predictions
```

Run the tests (standard library `unittest`, no extra dependency; the real-data tests skip when the
data is absent):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Other variants (for ablations): `--variant evidence | evidence_hold | hard_decoder`.
Any setting can be overridden for sensitivity runs: `--set switch_cost=24 --set margin_scale_m=5`.
The effective configuration is printed on every run. `outputs/` is git-ignored: generated files are
derived from the restricted tracking and must never be committed.

## Modules

| file | role |
|---|---|
| `config.py` | every tunable number, with the reason for its default; nothing is fitted |
| `data.py` | ATD → arrays in metres; row *i* = video frame *i*; missing/implausible → NaN |
| `features.py` | nearest player per team, velocities from positions, availability masks; per block |
| `evidence.py` | ownership evidence + weight, and a separate dead-ball channel |
| `decoder.py` | log-domain Viterbi over H / A / D, halves decoded independently |
| `dead_ball.py` | **experimental** dead-ball evidence from player movement (Phase 4.3), off by default |
| `ownership.py` | **rejected option** (Phase 4.4): controlled-ownership evidence — strict relative motion, track stability, contact episodes |
| `handover.py` | **rejected option** (Phase 4.4): handover support from consecutive established episodes, as time-varying switch costs |
| `duration.py` | Phase 4.5 dead-ball duration models: hysteresis conditioning and an exact soft-duration decoder (age chain) |
| `observation.py` | Phase 4.6: channel + reliability + zero-point fusion of dead-ball evidence (an unobserved channel contributes exactly 0) |
| `stoppage.py` | Phase 4.6: stoppage-entry channels (movement, out-of-play) |
| `restart.py` | Phase 4.6: restart evidence (sustained movement rise confirmed by the ball) |
| `diagnostics.py` | separate transition breakdown (direct H↔A vs dead-ball boundaries) next to the official scorer |
| `predict.py` | command line entry point (`--preset`, `--describe`, `--list-presets`) and the guarded, atomic `frame,label` writer |
| `smart_tagging.py` | Phase 4.9, **rejected / opt-in**: bounded team-phase prior from the organisers' automatic Smart Tagging (`rc2_tag_experimental`); never used by `rc1` |
| `model_registry.py` | Phase 4.8: status, features on/off, limitations, config fingerprint and reference output hash of every preset; `rc1` is the release candidate |

## Model in one paragraph

For each frame with a ball, the margin between the nearest home and nearest away player gives a
claim `tanh(margin / 3 m)`. It is weighted by how close the nearest player is (a loose ball says
little), how fast the ball moves relative to that player (a ball in flight says little), how many
players are tracked, and how consistent the claim is over ±0.5 s. The product is an *evidence score*
in nats — explicitly **not** a calibrated probability. Without a ball the score and its weight are
exactly zero: absence is no evidence, and it is never read as a dead ball. Dead-ball evidence is a
separate channel; today its only built-in cue is a ball tracked more than 1 m outside the lines, and
it has a hook (`external_dead`) for collective player speed, restarts and tags. The decoder adds
the evidence over time with finite costs for changing state (home↔away 12, alive→dead 10, dead→alive
6 nats), so a state is carried through gaps and left only when evidence outweighs the cost. Each
half is decoded on its own; frames outside the two graded halves are written as `X`.

## Dead-ball module (Phase 4.3) — EXPERIMENTAL, off by default

`use_dead_ball_module=false` (the default) reproduces the Phase 4.2 output byte for byte; a test
pins its SHA-256. The module is opt-in:

```bash
.venv/bin/python -m solution.predict                                              # Phase 4.2 model (default)
.venv/bin/python -m solution.predict --set use_dead_ball_module=true \
    --out outputs/predictions/solution_deadball.csv                               # with the dead-ball module
.venv/bin/python evaluation/validate_submission.py outputs/predictions/solution_deadball.csv
```

What it adds: a signed dead-vs-live score from the **median speed of all tracked players**, rolled
over ±3 s inside each half and compared with the local live-play level (the 75th percentile of the
same series over ±3 min), plus a reliability that is 0 wherever the tracking gives no usable speed.
Player speeds are accepted only when every frame of the difference window was observed and no step
looks like an identity switch. Nothing is derived from ball absence. A second cue (speed of the players
around the ball) and a ball-speed cue exist but are off, because they showed no benefit; a stationary
ball is deliberately not a cue. `dead_compensate_missing_ownership` (on) stops the decoder from
reading a hidden ball as a stoppage: without it, hiding the ball in the robustness test turned 32%
of alive frames into dead ones.

Every cue is a `dead_*` switch in `config.py`, e.g. for ablations:

```bash
--set dead_use_near_ball=true      --set dead_use_ball_motion=true      --set dead_allow_negative=false
--set use_outside_cue=false        --set dead_speed_reference=absolute  --set dead_compensate_missing_ownership=false
```

Known limits: it cannot call a stoppage while a team clearly controls the ball (by design), roughly
60% of the dead intervals it produces last under 5 s, and its tempo reference is sensitive
(`relative_ref` 0.6 / 0.7 / 0.8 changes the dead share a lot). Do not promote it to the default without
full-match evidence.

## Phase 4.4 options — ownership evidence, handover logic, symmetric ties

All off by default; the defaults reproduce Phase 4.2 (and, with `use_dead_ball_module=true`, Phase 4.3) byte
for byte, and tests pin both hashes.

```bash
--set ownership_v2=true        # strict relative motion + track stability + contact-episode evidence
--set use_handover=true        # time-varying home<->away switch costs from consecutive established episodes
--set decoder_late_ties=true   # symmetric tie-breaking (see below)
# individual switches: own_strict_motion, own_track_stability, own_reception (default true when ownership_v2 is on)
```

Status after the controlled evaluation: **`ownership_v2` and `use_handover` were rejected** — no measurable
improvement on the public window, mixed results under paired stress tests, and the contact-episode evidence
adds short possessions on the full match. Kept, switchable and tested, so the negative result is reproducible.
A contested-ball penalty and evidence gating by episode support were also tested and refuted (not implemented).

`decoder_late_ties` is **experimental**. It fixes a real asymmetry in the Phase 4.2 decoder: across a stretch
with no evidence (typically a long missing-ball gap) Viterbi's lowest-index tie-breaking placed a home→away
switch at the end of the stretch but an away→home switch, or the exit from the dead state, at the start. The
option places both at the latest time, so a state is carried through the gap until new evidence arrives.
It changes only evidence-free stretches (12 runs, 4.7% of graded frames in the full match). Treat the gains
seen so far as unproven: they come from a few stretches.

A note on caps: an unsupported direct switch must never cost more than the detour through the dead state
(`enter_dead_cost + exit_dead_cost`), otherwise the decoder invents stoppages to hide handovers;
`handover.max_extra_cost` enforces it and a test checks the invariant.

## Phase 4.5 — named reference configurations, decoder correctness, duration models

Reproduce any reference with `--preset` (default `legacy_p42`); `--set` still overrides single fields:

```bash
.venv/bin/python -m solution.predict --preset legacy_p42        # Phase 4.2, byte for byte
.venv/bin/python -m solution.predict --preset legacy_p43        # Phase 4.3, byte for byte
.venv/bin/python -m solution.predict --preset corrected         # symmetric tie-breaking (decoder_late_ties)
.venv/bin/python -m solution.predict --preset corrected_dead    # corrected + dead-ball module
.venv/bin/python -m solution.predict --preset duration_only     # dead-ball + soft duration decoder (rejected)
.venv/bin/python -m solution.predict --preset combined          # corrected + dead-ball + soft duration (rejected)
.venv/bin/python -m solution.predict --preset corrected_dead --set duration_model=hysteresis   # experimental
```

**Tie correction.** The first-order decoder takes the first maximum, so an exact tie goes to the
lowest state index (H < A < D). A switch *into* a high-index state therefore lands at the end of an
evidence-free stretch and a switch *into* a low-index state at its start (A→H, D→H, D→A at the start;
H→A, H→D, A→D at the end). `decoder_late_ties=true` breaks exact ties toward the switch for every
state pair, so all six land at the end: the previous state is carried until new evidence arrives. It
is mirror-symmetric (tested), never changes tie-free decoding (tested), and a block with no evidence
at all is written as X. Weak but non-zero evidence is never treated as a tie.

**Duration models** (`duration_model`): `hysteresis` conditions the dead-ball evidence with a Schmitt
trigger (neutral on spellF1, fewer dead intervals); `soft` decodes with an exact age chain and a
decaying penalty for ending a stoppage early (consistently lower spellF1 in the stress tests: kept for
the record); `both` combines them. None imposes a minimum duration. `check_transition_safety` verifies
that a direct home<->away switch stays cheaper than the cheapest route through the dead state; the
decoder warns at start-up if a configuration violates it.

## Phase 4.6 — observation-aware dead-ball evidence (EXPERIMENTAL, opt-in)

```bash
.venv/bin/python -m solution.predict --preset oa_only      # observation-aware evidence, corrected decoder
.venv/bin/python -m solution.predict --preset oa_restart   # + restart evidence (experimental)
.venv/bin/python -m solution.predict --preset oa_entry     # + both-teams-slow / persistent out-of-play (REJECTED)
.venv/bin/python -m solution.predict --preset oa_full      # entry + restart (REJECTED)
```
The legacy presets and every Phase 4.5 preset still reproduce their recorded outputs byte for byte.

**Why the Phase 4.3 module had a visibility conflict.** With the legacy emissions `[own/2, -own/2, dead]`
the alive states earn `|own|/2` per frame while the ball is observed and nothing while it is hidden. That is
*identical* (verified numerically, same Viterbi paths) to a dead-channel term of `-|own|/2` that exists only
for visible frames. Its mean is about the same on stoppage frames (0.59) and live frames (0.54), so in effect
it is a constant live prior of ~0.55 nats on visible frames only. The Phase 4.3 compensation subtracts that
mean on hidden frames. It equalises the average threshold, but (a) the threshold is large against the
movement score (max 0.6) and (b) with the ball hidden over a stoppage, detection fell to exactly 0%.
The legacy emission is not an artefact: it emits the *geometric* mean of the two teams for the dead state
(`e_D = (e_H + e_A)/2 = 0`), which is what makes D indifferent to ownership noise. The `mixture_null` emission
(arithmetic mixture) is not, and invents stoppages from contested ownership (test included); it is rejected.

**What the observation-aware path does.** Each dead-ball channel has a score, a reliability and a gain; the fused
score is `sum_c reliability_c * (gain_c * score_c - zero_c)`. An unobserved channel contributes exactly 0, the
zero point is scaled by reliability (so it cannot manufacture evidence in an unobserved stretch), and there is
no visibility compensation. Ownership still only decides H versus A and keeps its legacy emission.

**The unresolved conflict.** The movement channel's zero point (`activity_zero`, nats per frame) is a *policy*
parameter. Counterfactual sweep (ball hidden over alive frames / over stoppage frames; public window):

| zero point | alive frames called D, visible -> hidden | stoppage frames called D, visible / hidden |
|---|---|---|
| 0.0 | 4.6% -> 27.5% | 63% / 87% |
| 0.2 | 3.6% -> 16.1% | 60% / 82% |
| 0.3 (presets) | 2.9% -> 5.8% | 35% / 72% |
| 0.4 | 0.0% -> 0.2% | 4.5% / 28% |
| legacy, compensated | 2.0% -> 0.0% | 30% / **0%** |

No setting makes hiding the ball harmless *and* keeps detection: the movement cue alone separates stopped from
live play only moderately (AUC ~0.85 on the public window, ~0.71 on automatic live/dead proxies). Do not
promote without full-match evidence.
