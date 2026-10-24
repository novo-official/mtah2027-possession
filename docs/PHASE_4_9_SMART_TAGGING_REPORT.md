# Phase 4.9 — Smart Tagging experiment: report

**Decision: REJECT.** RC1 stays the selected release candidate, byte-identical (`a9be383bfa867ae034432d7746894951c89f6f9cd56ad8c9264eb9a5dd37f1be`).
The candidate (`rc2_tag_experimental`) is kept, opt-in and tested, so the negative result is reproducible. Nothing was staged, committed, pushed or submitted.
All figures are aggregates; no tag instance, frame-level record or restricted excerpt appears here.

## 1. Research hypothesis (pre-registered)

> A bounded team-phase prior from three eligible attack/defend tag pairs tells the decoder which team is in control where the tracking evidence is weak or
> absent, improving hand-over placement and hidden-ball resilience **without** creating hand-overs or stoppages the tracking does not support.

Pre-registration: [PHASE_4_9_PREREGISTRATION.md](PHASE_4_9_PREREGISTRATION.md), written after a metadata/semantic audit and before any candidate output or score
(SHA-256 at writing `e4b5e56e…ac21a`; one amendment, A1, appended **before** the candidate was run on the match — see §3).

## 2. Data semantics (what the 638 automatic instances provide)

* Time base verified with the organisers' helper (both kickoffs fall inside their tagged windows); `round(25·seconds)`, no offset, nothing shifted. All 638 intervals valid.
* Codes: 14 (12 phase codes, `SET PIECES` 59, `GOALS` 4). `Team` = Barcelona 314 / Atletic Club 288 / `N/A` 36 (32 of the 59 set pieces, incl. every throw-in).
* **Polarity.** Attack codes name the team in possession, defend codes the *defending* team. 142 of 171 attack instances have a defend partner on the identical interval, and
  140 of those 142 name opposite clubs: a pair is one fact stated twice, so it is one vote. The only unpaired attack instances are the 29 `LONG BALL`.
* **Agreement with the team the tracking names** on confident tracking frames (label-free): `BUILD UP` 0.878, `PROGRESSION` 0.850, `ATTACKING TRANSITION` 0.747, `LONG BALL` 0.681, `CREATING CHANCES` 0.617;
  zone codes 0.59–0.70 with unclear team semantics.
* **Coverage.** Team-resolved attack/defend intervals cover 44.5% of graded frames; the three eligible pair types give 112 votes covering 30.6% of graded frames.
* **New information?** Tagged frames have tracking ownership evidence on 82% of frames versus 41% for untagged ones: the tags mostly describe stretches the tracking already sees (they are
  generated from the same video/tracking), so little *complementary* information is expected where it is needed (ball gaps).
* **Set pieces do not give restart timing**: ≈ 35 s fixed windows, 54% without a team, start/end a median +2.8 s / +21 s from the nearest RC1 dead-interval start with quartiles spanning −25…+49 s. The restart mechanism was rejected before evaluation.

## 3. The prior (implemented exactly as registered)

Eligible pairs `BUILD UP`/`DEFENDING IN ATTACKING THIRD`, `PROGRESSION`/`DEFENDING IN MIDDLE THIRD`, `ATTACKING TRANSITION`/`DEFENSIVE TRANSITION`; one vote per pair (conflict or `N/A` ⇒ 0);
total budget **6.0 nats per pair** (= half a hand-over), spread uniformly over the interval (max 0.031 nats/frame against up to 3 nats/frame of tracking evidence); same-side overlaps not summed, opposite sides cancel;
added to the ownership channel only (`predict_codes(..., external_own=...)`); decoder unchanged. `Config` is RC1's, the tag settings live in `solution/smart_tagging.TagConfig` (fingerprint pinned),
so no existing fingerprint changed. Opt-in: only `--preset rc2_tag_experimental` reads the Smart Tagging file.

**Amendment A1 (before any evaluation).** The first implementation failed its own test: a bare `own_llr += prior` gave the favoured team a bonus over the dead state and shortened a stoppage carried through a
gap (synthetic 460 → 60 dead frames). Fixed by also adding `|prior|/2` to the dead channel so "favoured team versus dead ball" is unchanged. No parameter was altered.

## 4. Tests

34 new tests (`tests/test_smart_tagging.py`); whole suite **284 tests, 284 passed** (was 250). They cover XML parsing (cross-checked against the official parser), time-to-frame rounding, half alignment and clipping,
unknown / `N/A` / missing teams, attack-vs-defend polarity, duplicate pairs, conflicting and agreeing overlaps, missing/invalid timestamps, missing or malformed XML, events outside the graded halves, the 2–60 s duration rule,
the per-pair bound and the <1-switch budget, weak/`N/A` tags, strong contradictory tracking evidence, a tag alone not switching a state through a ball gap, a tag tipping a borderline decision only with tracking support,
no creation/removal of stoppages, no-tag fallback = exact RC1, team-swap symmetry, determinism, opt-in (RC1 never touches the tag file), pinned tag configuration, and on the real data: RC1 and "candidate with the
prior off" both reproduce `a9be383b…37f1be`, two CLI runs of the candidate are identical, a missing tag file fails cleanly. A tag can only add bounded evidence: there is no code path from an interval to a label.

## 5. Full-match, label-free diagnostics (graded halves)

| | RC1 | candidate |
|---|---|---|
| possessions (h1 / h2) | 270 (126 / 144) | 272 (128 / 144) |
| mean / median possession | 21.2 s / 13.0 s | 21.0 s / 13.0 s |
| < 3 s / > 60 s possessions | 40 / 18 | 41 / 17 |
| dead intervals · dead share | 42 · 10.00% | 42 · 10.06% |
| H↔A changes | 227 | 229 |
| **H↔A changes with no ball within ±1 s** | **0** | **1** |
| **changes inside ball gaps ≥ 5 s** | **0** | **2** |
| H/A↔D changes | 83 | 83 |
| home share of live frames | 51.5% | 52.6% |

* Frames differing from RC1: 1,948 (1.23% of graded) in 57 segments (median 2 frames, longest 29.9 s); 1,861 are H↔A relabellings. **71% of them (1,389) lie inside ball gaps ≥ 5 s** — the prior acts where the tracking is silent, i.e. it pulls hand-overs into gaps.
* H↔A changes: 8 new (no RC1 change within ±1 s), 6 removed, 49 moved by ≤ 1 s. Of the 8 new ones, tracking evidence points the same way for 2; the tag mass exceeds the tracking mass for 3.
* Changes near tag boundaries (±3 s; 20% of the timeline): RC1 63/227 (27.8%), candidate 66/229 (28.8%), new ones 5/8 — RC1 already changes near phase boundaries about as much; there is no sharp concentration.
* **Specificity controls** (copies of the same tags, shifted / flipped; frames differing from RC1): real 1,948; shifted −30 s 1,836, +30 s 3,624, ±90 s 6,052–8,907, ±180 s 3,394–12,286; all polarities flipped 3,221.
  The real tags move the output about as much as arbitrary shifted ones, and wrong tags are not harmless: a +90 s shift raises the dead share from 10.0% to 13.8% (+180 s: 15.4%), and every shifted/flipped control adds unsupported changes (2–6 inside gaps).

## 6. Robustness and independent value

| test (fixed seeds, 5 paired) | result |
|---|---|
| **Masking test** — hide 20% of the visible ball, agreement with the *unmasked* RC1 on hidden live frames (**pre-registered**) | RC1 0.7555 · real tags 0.7550 (paired gain −0.0005, better in **1/5** seeds) · shifted +90 s 0.7115 · flipped 0.7188 |
| supplementary, 40% hidden | RC1 0.7152 · real tags 0.7202 (+0.0050, 4/5 seeds) · shifted 0.6849 · flipped 0.6904 |
| ball masking 20%, agreement with own clean output | RC1 0.9003 · cand 0.9036; unsupported changes 0.0 vs **4.8** |
| player dropout 25% | 0.8717 vs 0.8801; unsupported 0.0 vs **4.4** |
| identity jumps | 0.9662 vs 0.9683; unsupported 0.0 vs **3.0** |
| long 30 s ball gaps | 0.8866 vs 0.8851; unsupported 0.0 vs **3.8**; dead share 8.94% vs **11.35%** |
| remove 10% / 30% / 50% of pairs | 10 / 529 / 979 frames change (0.006 / 0.33 / 0.62%) — locality holds |
| leave one pair out (112 pairs) | median 0 frames, p90 22, max 748 (29.9 s); 62 pairs change nothing |
| 10-minute tag outage (half 1 / half 2) | 639 / 10 frames change |

The prior is **local** (not fragile in the "few tags change everything" sense) but it provides **no measurable resilience** at the registered protocol, the small supplementary gain at 40% is not specific
(wrong tags lose 3–4 points against the same baseline), and every degraded input produces unsupported changes that RC1 does not.

## 7. Public-window evaluation (the one pre-registered run; reused data, non-decisive)

| | macroF1 [5–95%] | spellF1 [5–95%] | dead F1 | transition F1 @12 / @25 | direct H↔A @25 | dead entry / exit | possessions (GT 17) | fragmentation | team acc. (live) |
|---|---|---|---|---|---|---|---|---|---|
| RC1 | 0.569 [0.437, 0.649] | 0.457 [0.297, 0.617] | 0.11 | 0.11 / 0.21 | 4/12 | 0/4 · 0/4 | 18 | 0.90 | 0.886 |
| candidate | 0.568 [0.434, 0.647] | 0.457 [0.297, 0.617] | 0.11 | 0.11 / 0.16 | 3/12 | 0/4 · 0/4 | 18 | 0.90 | 0.884 |

No difference beyond noise; the transition metrics are marginally lower. No tuning followed this run.

## 8. Runtime and dependencies

Candidate: 3.1–3.3 s wall, peak memory 628 MB (`getrusage`) / 643 MB (`time`), identical to RC1 (decode 0.96 s each; tag parsing is negligible). The 261 KB XML is read in place, nothing is duplicated;
the CSV is 1.68 MB. No new dependency (standard-library XML + NumPy). Candidate CSV `outputs/rc2_tag_experimental/possession.csv` (git-ignored): 199,319 rows, official validator OK,
SHA-256 `3f51ec540d88a3741a62fd440ed7b0d2d8b74d799fa21066464f27412932a699`, regenerated independently with the identical hash. Labels H 75,147 · A 67,730 · D 15,980 · X 40,462.

## 9. Promotion decision against the registered criteria

| criterion | result |
|---|---|
| 1 correctness, RC1 pins | ✅ |
| 2 no new unsupported changes (0 / 0) | ❌ 1 H↔A change with no ball nearby, 2 changes inside gaps (clean run); 3–5 under every degradation |
| 3 dead share ±0.3 pt, ±2 intervals | ✅ clean (+0.06 pt, 0); ❌ under long gaps (+2.4 pt) and with shifted tags (+3.8 pt) |
| 4 locality (≤ 3% differ; 10% removal ≤ 0.5%) | ✅ 1.23%; 0.006% |
| 5 independent value (≥ 4/5 seeds, specific to real tags) | ❌ 1/5 seeds, mean −0.0005 |
| 6 robustness no worse than RC1 | ❌ more unsupported changes everywhere |
| 7 determinism, validator, identical hash | ✅ |
| 8 public window (non-decisive) | neutral / marginally lower |

**REJECT.** The tags are largely a restatement of what the tracking already shows (82% of tagged frames already have tracking evidence); where they could add information — ball gaps — they are too coarse (15–30 s phases)
to localise a hand-over, and a bounded prior either does nothing or moves a hand-over into the gap without support. Making the prior stronger would let tags decide, which the competition guidance and the plan forbid; making it
weaker only shrinks an effect that was already indistinguishable from zero. No retuning was done or is recommended.

## 10. Remaining risks (unchanged from RC1) and recommendation

The RC1 risks stand: dead ball under-called (public dead recall 6%), dead state carried through ball gaps (up to 139 s), ownership not re-estimated in the 40.6% of frames inside ≥ 5 s ball gaps, 4/12 direct hand-overs matched,
no private full-match estimate, public window exhausted. Smart Tagging does not mitigate any of them.

**Recommended final release configuration: `rc1`** (fallback: the official `baseline_smooth50`), with the dead-ball variants (`corrected_dead`, `oa_only`) remaining experimental as documented in the RC1 decision record.
The remaining decision for submission preparation is the user's: accept RC1's dead-ball under-calling, or knowingly choose a dead-ball variant despite its documented gate failures.

Reproduce: `python -m experiments.smart_tagging.audit` · `…evaluate [--skip-robust]` · `…public_once` (single-shot; refuses to run twice) · `python -m solution.predict --preset rc2_tag_experimental --out outputs/rc2_tag_experimental/possession.csv`.
