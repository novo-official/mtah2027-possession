# Smart Tagging prior — integration plan for Phase 4.9

> **Status: superseded by execution.** Phase 4.9 implemented one pre-registered variant of this plan and **rejected** it — see [PHASE_4_9_PREREGISTRATION.md](PHASE_4_9_PREREGISTRATION.md) and [PHASE_4_9_SMART_TAGGING_REPORT.md](PHASE_4_9_SMART_TAGGING_REPORT.md). Differences: the settings live in `solution/smart_tagging.py` (`TagConfig`), the preset itself selects the prior (no `--tags` flag), and RC1 is untouched. The text below is the original plan, kept unchanged.

RC1 stays exactly as released. A Smart Tagging experiment may only produce an **RC2 candidate** beside it; it may not edit `rc1`,
any historical preset, or any pinned hash. This document fixes where the prior would plug in and what it must prove.

What Smart Tagging is (from the organisers' own documentation, `starter/load_smart_tagging.py`): an *automatic* tactical annotation of the
tactical video — phases of play with a Team tag, and set pieces that are restarts at the end of dead-ball stretches; timed in seconds
from the first video frame (`frame = round(seconds × 25)`); accurate to about a second; contains false positives and negatives; phases are
**not** possessions. It is a prior, never a label source. The file is restricted data: parse it locally, print only counts and coverage.

## 1. Where it plugs in

The pipeline is `Tracking → Features → Evidence(own_llr, dead_llr) → decode_blocks → codes`. Only the evidence stage changes.

| interface | today | Smart Tagging use |
|---|---|---|
| `predict.predict_codes(tracking, config, variant, blocks, features, external_dead=None)` | `external_dead` is added to the dead channel (`evidence.dead_evidence`) | **dead prior**: positive nats on the stretch ending at a tagged restart; no negative evidence from absence of a tag |
| ownership channel | no external hook | add an optional `external_own=None` argument next to `external_dead`, summed into `own_llr` before decoding; default `None` ⇒ RC1 output unchanged (byte-identical, tested) |
| new module `solution/tags.py` (RC2 only) | — | `tag_evidence(tags, n_frames, blocks) -> (own_llr, dead_llr, reliability)`: per-frame arrays in the same uncalibrated evidence units, **exactly 0 where no tag applies** |

Rules the new module must follow, because they are what make RC1 trustworthy: absence of a tag is never evidence; evidence is finite
and small compared with the transition costs (a tag can tip a decision, not make one); it is computed per half and cannot cross
halftime; it is symmetric in the teams (a tag for the home team is the mirror of one for the away team).

## 2. What must not change

`decoder.build_emissions`, `viterbi`, `viterbi_time_varying`, `transition_costs`, `decode_blocks` (signature and behaviour),
`check_transition_safety`, `data.load_tracking`, `write_submission`, the `rc1` preset and its fingerprint, every pinned output hash.
A tag prior enters as additive evidence only; the decoder stays tag-agnostic.

## 3. Switching it off

* RC2 settings live in their **own** dataclass (`TagConfig`), *not* as new fields of `Config`. Adding a field to `Config` would change every
  preset's `config_fingerprint` and fail the pins — which is the guard working as intended, so do not route around it.
* `python -m solution.predict --preset rc1` never reads the Smart Tagging file (a test asserts the file is not opened). Tags are opt-in via
  an explicit flag (`--tags`) that is rejected together with `--preset rc1`, and the RC2 candidate gets its own registry entry (`rc2_candidate`, status `experimental`).
* Missing tag file ⇒ clear error when `--tags` is given, and RC1 is unaffected.

## 4. How the comparison against RC1 is made

1. Regenerate RC1 first and check `a9be383b…37f1be`; abort if it differs.
2. Pre-register **one** RC2 configuration before looking at any score: gains derived from the tag semantics and from label-free quantities
   (for example how often tag phases agree with RC1's high-confidence ownership frames), written into the registry with its fingerprint.
3. Run it once through `experiments/rc1_selection/compare.py` (extended with the candidate name): same public metrics, same label-free
   diagnostics, same degradation tests, same sensitivity table. Report **paired** differences against `rc1`.
4. Output goes to `outputs/rc2/`; `outputs/rc1/` is never written.

## 5. Not overfitting the reused public ground truth

* No sweep, no threshold search, no "try a gain of 0.5, 1.0, 2.0" against the public window. One pre-registered configuration, one scoring run.
* The public window is a smoke test (does it break anything?), not a ranking: 17 possessions, 4 stoppages.
* Decisions rest on label-free checks that tags can actually answer: do tagged restarts end RC1's carried dead stretches (the open
  question of [RC1_DECISION_RECORD.md §5](RC1_DECISION_RECORD.md))? do tagged team phases support the H↔A changes RC1 makes in the middle of ball gaps?
  does the tag prior create state changes with no ball or movement evidence nearby (it must not)?
* Tag false positives are expected; the prior must be harmless when a tag is wrong (a wrong tag moves the output by a bounded, tested amount).

## 6. Acceptance gate for RC2

An RC2 may replace RC1 only if **all** hold, each with a test or a recorded measurement:

1. RC1 hash and every historical pin unchanged; `--preset rc1` byte-identical with and without the new code present.
2. G1–G7 of the decision record pass at least as well as RC1 (G4 should improve; if it does not, the tag prior bought nothing on the open risk).
3. Team-swap symmetry and half isolation tests pass with tags enabled.
4. A missing / empty / mis-timed tag file degrades to RC1 behaviour (tested with synthetic tags; no crash, no phantom dead state).
5. A wrong tag (synthetic false positive) changes at most a bounded number of frames.
6. Two independent runs give identical hashes; official validator OK; clean-room run from a source-only copy reproduces the file.
7. A written decision record in the style of this one, with accepted risks, and RC1 retained as fallback.

If the gate is not cleared, RC1 stands and the experiment is documented as a negative result.
