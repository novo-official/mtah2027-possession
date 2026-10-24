"""Versioned registry of the named model configurations.

Parameters are defined once, in `config.py` (`PRESETS`); this module adds what a preset needs to be
trusted and reproduced without reading the research history: its status, what is switched on and
off, the decoder semantics, the known limitations, how to regenerate it and (where one was ever
recorded) the SHA-256 of its full-match output. Nothing here changes a prediction.

Status vocabulary
    release-candidate   the single configuration frozen for submission preparation (`rc1`)
    reference           historical output kept byte-reproducible; not a candidate
    experimental        implemented, tested, opt-in; not selected (reasons in docs/RC1_DECISION_RECORD.md)
    rejected            kept so that the negative result stays reproducible

Guard rails (tests/test_model_registry.py)
    * every key of `config.PRESETS` is registered here and vice versa;
    * `config_fingerprint` of every preset is pinned, so editing a historical preset fails a test
      instead of silently redefining it;
    * the output hash of every preset with a recorded reference is re-derived from the installed data.

The hashes are digests of CSV files derived from the restricted tracking. A digest cannot be inverted
into labels, but the files themselves are never committed.
"""

import glob
import hashlib
import os
from dataclasses import dataclass, fields
from types import MappingProxyType

from . import smart_tagging
from .config import PRESETS

RELEASE_CANDIDATE = "rc1"
RELEASE_VERSION = "RC1"
# Conservative external fallback: the organisers' own baseline. It is not produced by solution/ and its
# private-match score is the one published in starter/baseline.py, so it is the only known quantity.
FALLBACK_BASELINE = {
    "name": "baseline_smooth50",
    "file": "evaluation/results/baseline_smooth50.csv",
    "sha256": "6d4e04b80b9016cc16ada99697bf219d3a74b58750dd6d5ec5a4cd5fc32ab4b9",
    "reproduce": ".venv/bin/python starter/baseline.py --window 50 --out <scratch dir>   # never use the default --out",
    "note": "Published full-match score (EVALUATION.md, starter/baseline.py docstring): macroF1 0.6724, spellF1 0.2433.",
}
# Official baselines reproduced byte for byte in Phase 4.8 (committed under evaluation/results/).
OFFICIAL_BASELINES = MappingProxyType({
    "baseline_raw": "9471afeb95722454a96189481561adfb73b3963365a83969f04a986930f12511",
    "baseline_smooth25": "c1fd857c2c0bd5299260b6c55dcd419bde41a4c5af46f9bbcb41b4b3b577353f",
    "baseline_smooth50": "6d4e04b80b9016cc16ada99697bf219d3a74b58750dd6d5ec5a4cd5fc32ab4b9",
    "baseline_smooth125": "41704ddccc1edcddf6a0781e3fdfa06ae07b19fb9ead3bf67f663bfc7b1900ba",
})


@dataclass(frozen=True)
class ModelSpec:
    name: str
    status: str
    introduced: str                  # phase that introduced the configuration
    description: str
    enabled: tuple
    disabled: tuple
    decoder: str                     # decoder semantics in one line
    limitations: tuple
    reference_sha256: object = None  # SHA-256 of the full-match CSV, where one was recorded

    @property
    def config(self):
        return PRESETS[self.name]

    @property
    def fingerprint(self):
        return config_fingerprint(self.config)

    @property
    def tag_config(self):
        """The Smart Tagging settings of this preset, or None (the prior is off for every preset but the experimental one)."""
        return smart_tagging.TAG_PRESETS.get(self.name)

    @property
    def tag_fingerprint(self):
        return None if self.tag_config is None else config_fingerprint(self.tag_config)

    @property
    def reproduce(self):
        return ".venv/bin/python -m solution.predict --preset {} --out outputs/<dir>/{}.csv".format(self.name, self.name)


def config_fingerprint(config):
    """SHA-256 over every field name and value of a Config, in declaration order (stable, order-sensitive)."""
    text = "\n".join("{}={!r}".format(f.name, getattr(config, f.name)) for f in fields(config))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_digest():
    """SHA-256 over the names and bytes of every solution/*.py file (sorted): the code identity of a release.
    Used in docs/RC1_RELEASE_MANIFEST.md; any change to the production code changes it."""
    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(here, "*.py"))):
        digest.update(os.path.basename(path).encode("utf-8") + b"\0")
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


_LEGACY_TIES = "legacy Viterbi ties: lowest state index wins, so a switch into H/A/D may land at the START of an evidence-free stretch"
_CORRECTED_TIES = "corrected ties (decoder_late_ties): a state is carried through an evidence-free stretch until new evidence arrives; mirror-symmetric in the teams"
_OWNERSHIP = "ownership evidence (nearest-player claim x proximity x relative motion x observation x consistency)"
_OUTSIDE = "out-of-play cue (ball tracked > 1 m outside the lines)"
_MOVEMENT = "Phase 4.3 movement dead-ball module (median player speed relative to the local live-play level, visibility-compensated)"
_OA = "Phase 4.6 observation-aware dead-ball fusion (movement channel with zero point 0.30 + out-of-play channel)"
_NO_PERSIST = "dead evidence decides D; no D frame is held only by transition costs"

_SPECS = [
    ModelSpec(
        "rc1", "release-candidate", "4.8",
        "Release candidate 1: Phase 4.2 ownership evidence with the corrected (symmetric tie) decoder and the physical "
        "out-of-play cue. No movement-based dead-ball model.",
        (_OWNERSHIP, _OUTSIDE, _CORRECTED_TIES, "per-half independent decoding; frames outside the graded halves are X"),
        (_MOVEMENT, _OA, "restart evidence", "ownership_v2", "handover costs", "duration models"),
        _CORRECTED_TIES,
        ("D is rare and almost entirely tied to the ball leaving the pitch: public-window dead recall ~6%, so stoppages with "
         "the ball in view are called live",
         "a dead state entered through the out-of-play cue is carried through ball-tracking gaps without further evidence "
         "(longest evidence-free stretch ~139 s; 75% of D frames are evidence-free)",
         "ball gaps >= 5 s cover ~41% of graded frames; ownership is carried through them, never re-estimated",
         "no independent full-match ground truth: all full-match numbers are label-free behaviour descriptions"),
        "a9be383bfa867ae034432d7746894951c89f6f9cd56ad8c9264eb9a5dd37f1be"),
    ModelSpec(
        "rc2_tag_experimental", "rejected", "4.9",
        "RC1 plus a bounded Smart Tagging team-phase prior (docs/PHASE_4_9_PREREGISTRATION.md). REJECTED in Phase 4.9 "
        "(docs/PHASE_4_9_SMART_TAGGING_REPORT.md): no measurable independent value, creates unsupported changes, "
        "not specific to the real tags. Kept so that the negative result stays reproducible. Without the prior it is "
        "byte-identical to rc1.",
        (_OWNERSHIP, _OUTSIDE, _CORRECTED_TIES, "Smart Tagging team-phase prior: three eligible attack/defend pairs, "
         "6.0 nats total per pair, added to the ownership channel"),
        (_MOVEMENT, _OA, "restart evidence", "set-piece tags", "zone-code tags", "ownership_v2", "handover costs", "duration models"),
        _CORRECTED_TIES + "; decoder unchanged",
        ("all limitations of rc1 (dead ball under-called, dead state carried through ball gaps)",
         "tags are automatic phases, not possessions; ~31% of graded frames receive a prior; pairs with N/A or conflicting teams contribute zero",
         "pulls hand-overs into ball gaps (71% of the frames it changes lie in gaps >= 5 s): 1 H<->A change with no ball nearby and 2 changes inside gaps appear, 0 in rc1",
         "wrong or shifted tags move the dead share (+3.8 pt for a 90 s shift) and the hidden-ball agreement (-4 pt)"),
        None),
    ModelSpec(
        "legacy_p42", "reference", "4.2",
        "Phase 4.2 model: ownership evidence, legacy tie-breaking, out-of-play cue only.",
        (_OWNERSHIP, _OUTSIDE), (_MOVEMENT, _OA, "corrected ties"), _LEGACY_TIES,
        ("tie-breaking asymmetry: A->H, D->H, D->A switches land at the start of evidence-free stretches",),
        "50f32a07fd2989f7c2a4e8e9b75f13acf133582f52f3b4f20e2d8f5f3417bba3"),
    ModelSpec(
        "legacy_p43", "reference", "4.3",
        "Phase 4.3 model: legacy_p42 plus the movement dead-ball module.",
        (_OWNERSHIP, _OUTSIDE, _MOVEMENT), (_OA, "corrected ties"), _LEGACY_TIES,
        ("hiding the ball turns stoppage detection to 0%", "25% player-track dropout adds ~10 points of dead share",
         "46% of dead intervals last < 3 s", "dead share moves -5.6/+8.4 points for relative_ref 0.6/0.8"),
        "367b7415e3f2f700115c1846f7ba3ce366b34203732539f4a81d37af79f148a2"),
    ModelSpec(
        "corrected", "reference", "4.5",
        "Phase 4.5 corrected decoder on the Phase 4.2 evidence. Same configuration values as rc1 (kept as the "
        "historical name; rc1 is pinned separately).",
        (_OWNERSHIP, _OUTSIDE, _CORRECTED_TIES), (_MOVEMENT, _OA), _CORRECTED_TIES,
        ("see rc1",),
        "a9be383bfa867ae034432d7746894951c89f6f9cd56ad8c9264eb9a5dd37f1be"),
    ModelSpec(
        "corrected_dead", "experimental", "4.5",
        "Corrected decoder plus the Phase 4.3 movement dead-ball module. Best-scoring non-OA model on the public window; "
        "fails the player-dropout and short-dead-interval gates (docs/RC1_DECISION_RECORD.md).",
        (_OWNERSHIP, _OUTSIDE, _MOVEMENT, _CORRECTED_TIES), (_OA, "restart evidence"), _CORRECTED_TIES + "; " + _NO_PERSIST,
        ("25% player-track dropout adds ~10 points of dead share", "ball hidden over a stoppage: 0% detected",
         "46% of dead intervals last < 3 s"),
        "f9984daa615e83f6822321851b3645c58264ef4e3717a0dadaa59ed3112ab5fb"),
    ModelSpec(
        "corrected_dead_hysteresis", "experimental", "4.5",
        "corrected_dead with Schmitt-trigger conditioning of the dead evidence (neutral on the public window).",
        (_OWNERSHIP, _OUTSIDE, _MOVEMENT, _CORRECTED_TIES, "dead-evidence hysteresis"), (_OA, "restart evidence"),
        _CORRECTED_TIES, ("does not remove the player-dropout sensitivity (+8.7 points of dead share)",),
        "17062aa9c43e928f6a0c7881bf832d582aeae0156e33ac238315b5d6558649e3"),
    ModelSpec(
        "duration_only", "rejected", "4.5",
        "legacy_p43 with the soft dead-duration decoder (lower spellF1 in every stress test).",
        (_OWNERSHIP, _OUTSIDE, _MOVEMENT, "soft duration decoder"), (_OA, "corrected ties"), _LEGACY_TIES,
        ("rejected in Phase 4.5",),
        "76b87a18b3649f2f228e5e618b60f84a30b1abca1d3b565340ecfbe09b0009a8"),
    ModelSpec(
        "combined", "rejected", "4.5",
        "corrected_dead with the soft dead-duration decoder.",
        (_OWNERSHIP, _OUTSIDE, _MOVEMENT, _CORRECTED_TIES, "soft duration decoder"), (_OA,), _CORRECTED_TIES,
        ("rejected in Phase 4.5",),
        "515a9f94c73d5610d2a5048a71cd01348dabc4a5cd2a612dba89446e4514e1e5"),
    ModelSpec(
        "oa_only", "experimental", "4.6",
        "Corrected decoder with observation-aware dead-ball fusion. Best public-window dead-ball F1; not selected "
        "(policy zero point on a cliff, 51% of ball-gap frames called D, unverifiable 24% dead share).",
        (_OWNERSHIP, _OA, _CORRECTED_TIES), (_MOVEMENT, "restart evidence", "visibility compensation"),
        _CORRECTED_TIES + "; " + _NO_PERSIST,
        ("activity_zero is a policy parameter: 0.2/0.4 move the dead share +3.4/-6.6 points",
         "87% of D frames lie inside ball gaps >= 5 s", "59 of 212 dead-boundary changes have no ball within +-1 s",
         "hiding the ball raises false stoppages on alive frames 3.1% -> 5.8%"),
        "f6ec0c4b883b64d585b57adcc35ab14f5bd7dc7a138138e23d41bf07f899a20c"),
    ModelSpec(
        "oa_restart", "experimental", "4.6",
        "oa_only plus restart evidence (no measurable score effect).",
        (_OWNERSHIP, _OA, "restart evidence", _CORRECTED_TIES), (_MOVEMENT,), _CORRECTED_TIES + "; " + _NO_PERSIST,
        ("as oa_only",),
        "af9d8ad3baa0d7640b6e0310a5205b3a177233337e8358d76dc37112dd688a0e"),
    ModelSpec(
        "oa_entry", "rejected", "4.6",
        "oa_only with both-teams-slow activity and persistent out-of-play requirement (rejected).",
        (_OWNERSHIP, _OA, "both-teams activity statistic", _CORRECTED_TIES), (_MOVEMENT, "restart evidence"),
        _CORRECTED_TIES, ("rejected in Phase 4.6",),
        "76929a46d209de9c7bb22b6573c86bcb5ff2acf28165f5fb34a921b9eb61e4f1"),
    ModelSpec(
        "oa_full", "rejected", "4.6",
        "oa_entry plus restart evidence (rejected).",
        (_OWNERSHIP, _OA, "both-teams activity statistic", "restart evidence", _CORRECTED_TIES), (_MOVEMENT,),
        _CORRECTED_TIES, ("rejected in Phase 4.6",),
        "33443674a6e4162316de37a73f32cc2a5ee0829d03cadb1f279f611e1bed602b"),
    ModelSpec(
        "oa_mixture_null_rejected", "rejected", "4.6",
        "oa_only with the mixture-null emission, which invents phantom stoppages from contested ownership.",
        (_OWNERSHIP, _OA, "mixture_null emission", _CORRECTED_TIES), (_MOVEMENT,), _CORRECTED_TIES,
        ("phantom stoppages in 59-67 of 80 random sequences (Phase 4.6)",), None),
]

REGISTRY = MappingProxyType({spec.name: spec for spec in _SPECS})


def get(name):
    """The ModelSpec for a preset name, or a KeyError listing the registered names."""
    if name not in REGISTRY:
        raise KeyError("unknown preset {!r}; registered: {}".format(name, sorted(REGISTRY)))
    return REGISTRY[name]


def names(status=None):
    return [n for n, s in REGISTRY.items() if status is None or s.status == status]


def describe(name):
    """Plain-text card for one preset (used by `python -m solution.predict --describe`)."""
    spec = get(name)
    lines = ["{} [{}]  introduced in Phase {}".format(spec.name, spec.status, spec.introduced), spec.description,
             "decoder:  " + spec.decoder, "enabled:"]
    lines += ["  + " + item for item in spec.enabled]
    lines += ["disabled:"] + ["  - " + item for item in spec.disabled]
    lines += ["limitations:"] + ["  ! " + item for item in spec.limitations]
    if spec.tag_config is not None:
        lines += ["smart tagging prior (experimental, opt-in):", spec.tag_config.describe(),
                  "tag config fingerprint: " + spec.tag_fingerprint]
    lines += ["config fingerprint: " + spec.fingerprint,
              "source digest:      " + source_digest(),
              "reference sha256:   " + (spec.reference_sha256 or "(none recorded)"),
              "reproduce:          " + spec.reproduce]
    return "\n".join(lines)
