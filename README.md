# Frame-level ball-possession inference from automatic tracking — RC1-FINAL

Solution to **More than a Hack 2027 — Phase 1: Ball Possession** (Metrica Sports, FC Barcelona, Mobile World Capital Barcelona).
Team: **[REGISTERED TEAM NAME — to be filled in by the team before publication]**.

For every one of the 199,319 frames of the tactical video of *FC Barcelona vs Athletic Club*, the program writes `H` (home in possession, ball alive),
`A` (away in possession), `D` (dead ball) or `X` (no answer). It is a deterministic, CPU-only, NumPy-only pipeline: no machine learning, no video, no
network access, no hand labels. This repository regenerates the submitted prediction file **byte for byte** from the organiser-provided dataset.

## Approach (release `RC1-FINAL`, preset `rc1`)

1. **Tracking → geometry.** The automatic tracking (ATD) is converted to metres; anything missing or implausible is `NaN`, never 0.
2. **Ownership evidence.** Per frame, the nearest home and nearest away player to the ball give a bounded claim, weighted by proximity, relative motion,
   number of tracked players and temporal consistency. No ball → exactly zero evidence.
3. **Temporal decoding.** A three-state (H, A, D) log-domain Viterbi decoder with finite transition costs, run independently in each half, with
   symmetric ("late") tie-breaking so a state is carried through an evidence-free stretch until evidence returns.
4. **Dead ball.** Only one physical cue is enabled: the ball *tracked* more than 1 m outside the pitch lines.

Everything that was tried and not selected (movement-based dead-ball models, observation-aware fusion, handover costs, duration models, video features,
Smart Tagging priors) is implemented, tested and **off**; see [`docs/TECHNICAL_REPORT_FINAL.md`](docs/TECHNICAL_REPORT_FINAL.md) and
[`docs/FINAL_RISK_ACCEPTANCE.md`](docs/FINAL_RISK_ACCEPTANCE.md).

## Inputs (not included)

The match assets are licensed to enrolled participants only and are **not** in this repository. Only the tactical tracking is used by the final model:

```
data/tactical/BAR-ATH_tactical_atd.txt     tracking, players and ball
data/tactical/BAR-ATH_tactical_atd.xml     tracking metadata (team membership)
```

Unpack the organiser bundle into `./data` (keeping its names) or point `MTAH_DATA_DIR` at it. See [`DATA.md`](DATA.md).

## Installation

Tested with **Python 3.14.8, NumPy 2.5.4** (pandas 3.0.6 only for the organisers' scorer/validator) on Fedora Linux x86-64. Byte-identical output is guaranteed
for these versions only; other versions are untested.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Reproduce the submission file

```bash
.venv/bin/python -m solution.predict --preset rc1 --out outputs/possession.csv
.venv/bin/python evaluation/validate_submission.py outputs/possession.csv
sha256sum outputs/possession.csv
# expected: a9be383bfa867ae034432d7746894951c89f6f9cd56ad8c9264eb9a5dd37f1be
```

Do **not** pass `-I` to Python for `-m solution.predict` (the repository root must be importable). Plain `python -m solution.predict` without `--preset` runs the
legacy Phase 4.2 model, which is *not* the submission. `--describe rc1` and `--list-presets` need no data. The command refuses to write into `data/`, `evaluation/gt/`
or over a committed baseline, writes atomically, and exits with status 2 and a one-line message if the data is missing.

Score on the organisers' public five-minute window (needs no restricted data other than your own prediction file):

```bash
mkdir -p outputs/score && cp outputs/possession.csv outputs/score/possession_rc1.csv
.venv/bin/python evaluation/run_evaluation.py --results outputs/score --bootstrap 1000
```

## Output contract

`frame,label` header, exactly 199,319 rows for frames 0…199318 in order, labels `H`/`A`/`D`/`X`. The two graded halves (video frames 11185–90651 and
110808–190197) are answered; every other frame is `X`. Row *i* is frame *i* of the **tactical** video.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests          # 268 tests; the real-data ones (23) skip themselves when the dataset is absent
```

## Performance

One CPU core, no GPU: about 3.1–3.4 s per match (≈ 1.9 s parsing the tracking, ≈ 1.2 s features and decoding), peak memory about 630–645 MB.

## Results and limitations (read before relying on this)

Public five-minute window (a worked example reused many times, **not** a validation set): macroF1 0.569, spellF1 0.457. **There is no private full-match estimate.**
Known weaknesses, measured and accepted: dead ball is under-called (public dead recall ≈ 6%), a dead state can be carried through long missing-ball gaps
(up to ≈ 139 s), ownership is not re-estimated while the ball is untracked (≈ 41% of graded frames), direct hand-overs are localised poorly, and the 2,934
graded pre-kickoff frames of the second half (documented as all `D` in [`FAQ.md`](FAQ.md)) are not special-cased. Details: [`docs/FINAL_RISK_ACCEPTANCE.md`](docs/FINAL_RISK_ACCEPTANCE.md).

## Layout

```
solution/        the pipeline (solution/README.md: production path vs experimental modules); predict.py is the entry point
tests/           unit and integration tests (synthetic fixtures; real-data tests are skipped without the dataset)
experiments/     aggregate-only evidence scripts behind the model-selection decision (optional)
docs/            technical report, risk acceptance record, decision record, Phase 4.9 negative result
evaluation/      the organisers' scorer, validator and public ground truth (unchanged)
games.py         the organisers' per-match constants and asset paths (unchanged)
```

## Licence and attribution

MIT. `games.py`, `evaluation/`, `starter/load_smart_tagging.py`, the organiser documents (`CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`, `RULES.md`, `SUBMISSION.md`)
and the public ground truth come unchanged from the organisers' repository, <https://github.com/metrica-sports/more-than-a-hack-2027> (© 2026-2027 Metrica Sports, MIT).
The match data is **not** covered by this licence and is not redistributed. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
