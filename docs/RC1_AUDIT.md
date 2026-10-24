# Phase 4.8 — codebase audit (summary)

Scope: `games.py`, `evaluation/`, `solution/` (15 modules, 2.3k lines), `tests/` (208 tests at start), `experiments/`,
`.gitignore`, dependency files, Phase 4.1–4.7 reports (kept locally in the git-excluded `analysis/`). Written from the
state at the *start* of Phase 4.8; the right-hand column says what was done.

## Functionality by status

| status | modules |
|---|---|
| stable (production path) | `data` (ATD loader), `features`, `evidence` (ownership + out-of-play cue), `decoder`, `config`, `predict` |
| experimental, opt-in | `dead_ball` (4.3), `observation` + `stoppage` + `restart` (4.6), `duration.hysteresis_dead` (4.5) |
| rejected, kept reproducible | `ownership` + `handover` (4.4), soft duration decoder in `duration` (4.5), `mixture_null` emission, `oa_entry`, `oa_full` |
| diagnostics | `diagnostics` (transition breakdown beside the official scorer) |
| research (outside `solution/`) | `experiments/video_feasibility` (Phase 4.7 NO-GO), `experiments/rc1_selection` (this phase) |

`solution/` never imports `experiments/`, `analysis/` or `evaluation/` results; its only third-party import is NumPy (the
standard-library XML parser reads the metadata). No OpenCV / PyTorch / TensorFlow anywhere under `solution/`, `tests/`, `experiments/`.

## Findings

| # | finding | severity | action |
|---|---|---|---|
| 1 | `--set bool=…` turned any unrecognised word (`on`, `True1`, a typo) into **False** silently, so a typo could disable a feature in a "release" run | medium | fixed: strict boolean parsing, clear error (tested) |
| 2 | `write_submission` wrote to any path, non-atomically: an interrupted run left a truncated "valid-looking" file, and nothing stopped `--out` pointing at `data/`, the ground truths, `starter/` or a committed baseline (the known `baseline.py` default-`--out` overwrite trap has the same shape) | medium | fixed: atomic temp-file + rename, protected-path refusal, checked *before* any computation (tested) |
| 3 | missing data or a bad override produced a raw traceback | low | fixed: one-line error, exit code 2 (tested) |
| 4 | no validation of the codes before writing (length, code range) | low | fixed: `validate_codes` (tested) |
| 5 | the Phase 4.5 hysteresis reference existed only as an ad-hoc `--set duration_model=hysteresis` run | low | named preset `corrected_dead_hysteresis`, output hash re-derived (matches the Phase 4.5 file) |
| 6 | presets were bare dict entries: no status, no known limitations, nothing stopping a later edit from silently redefining a historical result | medium | `solution/model_registry.py` + pinned config fingerprints + pinned output hashes for 12 presets |
| 7 | the corrected decoder **carries a dead state through arbitrarily long evidence-free stretches** (74.9% of `rc1`'s D frames; longest 139 s) | design risk | measured, bounded, corroborated, pinned by a test, accepted — [RC1_DECISION_RECORD.md §5](RC1_DECISION_RECORD.md) |
| 8 | with `decoder_late_ties`, a block with no evidence at all is written X (legacy: H). Correct and tested, but it is a behaviour difference worth knowing | info | documented |
| 9 | default `--preset` is still `legacy_p42` (backwards compatible: existing pins and docs rely on it). A run without `--preset` is not RC1 | low | kept; the CLI now prints a note; `--preset rc1` is the documented route |
| 10 | `np.fromstring(text, sep=",")` (ATD parser) is the NumPy text-mode API; it works on 2.5.4 without a deprecation warning (checked with `-W error`) but could be removed upstream | low | not changed (identical output required); mitigated by pinning NumPy |
| 11 | `data.py` / `predict.py` insert the repository root into `sys.path` to import `games.py` | info | harmless, documented; the tests exercise it from a foreign working directory |
| 12 | unused imports: `np` in `diagnostics.py`, `D` in `predict.py`; `_run_lengths` is defined in both `features.py` and `stoppage.py` (behaviour-identical) | trivial | left alone: no demonstrated defect |

## Checked and found sound

* **Frame alignment.** The loader refuses any ATD whose ids are not exactly `1…N` and maps row *i* to video frame *i*
  (`DATA.md`: `video_frame = frame_id − 1`). The organisers' `starter/baseline.py` regenerates the four official baselines byte for byte,
  and the alignment is exercised by the public window: `rc1` names the right team on 88.6% of the ground-truth-alive frames (a misaligned file would sit near chance). Graded blocks are 11185…90651 and 110808…190197 inclusive = 79,467 + 79,390 = **158,857** frames; everything else is X (40,462 frames).
* **No off-by-one in block ends**: `half_blocks` converts the inclusive ends of `games.half_video_blocks` to exclusive stops (tested).
* **Missing values.** Missing / implausible positions are NaN, never 0; a velocity or aggregate exists only when its whole window
  was observed; a missing ball contributes exactly 0 to every state (so it can neither favour a team nor a stoppage).
* **Halves are independent**: features, rolling windows, level references and the Viterbi run per block; tests prove that
  decoding block 2 alone equals decoding it with block 1 present, on synthetic and real data.
* **Numerics.** The decoder is log-domain float64, ties use a 1e-9 tolerance, the evidence is finite for degenerate input (tested),
  divisions that can hit zero are inside `errstate` / guarded.
* **Filesystem side effects.** None besides the requested output file (and Python's own `__pycache__`); no caches, no temp analysis files.
* **Determinism.** Two independent processes, and a run from a foreign working directory with an empty environment, give the same bytes.
