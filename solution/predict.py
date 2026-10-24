"""Regenerate the prediction file from the installed official assets.

    .venv/bin/python -m solution.predict --preset rc1 --out outputs/rc1/possession_rc1.csv   # release candidate 1
    .venv/bin/python -m solution.predict                      # legacy default (Phase 4.2) -> outputs/predictions/solution_decoder.csv
    .venv/bin/python -m solution.predict --describe rc1       # status, features, limitations, fingerprint (no data needed)
    .venv/bin/python -m solution.predict --variant evidence_hold --out outputs/predictions/x.csv
    .venv/bin/python -m solution.predict --set switch_cost=24 --set control_radius_m=5

Variants (all deterministic):
    decoder        soft ownership evidence -> three-state Viterbi decoder (the model)
    evidence       soft evidence only, hard sign, X where the evidence is weak or absent
    evidence_hold  as `evidence`, but X is replaced by the last label of the half (no decoder)
    hard_decoder   ablation: the baseline's hard nearest-player rule fed to the same decoder

Frames outside the two published graded halves are written as X (they are never scored).
Presets are documented in solution/model_registry.py; `rc1` is the frozen release candidate.
The output follows the submission contract: header `frame,label`, one row per video frame
(199,319), labels in H/A/D/X. Check it with `python evaluation/validate_submission.py <file>`.
"""

import argparse
import hashlib
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402

from . import model_registry, smart_tagging  # noqa: E402
from .config import DEFAULT, PRESETS, parse_overrides  # noqa: E402
from .duration import check_transition_safety, hysteresis_dead  # noqa: E402
from .observation import fuse  # noqa: E402
from .restart import restart_channel  # noqa: E402
from .stoppage import stoppage_channels  # noqa: E402
from .data import half_blocks, load_tracking  # noqa: E402
from .decoder import A, D, H, LABELS, X, decode_blocks  # noqa: E402
from .dead_ball import dead_ball_evidence, ownership_compensation  # noqa: E402
from .evidence import Evidence, dead_evidence, hard_nearest_evidence, ownership_evidence  # noqa: E402
from .handover import handover_costs  # noqa: E402
from .ownership import control_evidence, control_signals  # noqa: E402
from .features import compute_features  # noqa: E402

VARIANTS = ("decoder", "evidence", "evidence_hold", "hard_decoder")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "predictions", "solution_decoder.csv")


def block_ranges(game=games.GAME):
    return [(first, stop) for _, first, stop in half_blocks(game)]


def evidence_labels(own_llr, blocks, threshold, hold):
    """Hard H/A from the sign of the evidence where |evidence| >= threshold; else X, or the last
    label of the block when `hold` (back-filled at the head of the block)."""
    codes = np.full(len(own_llr), X, dtype=np.int8)
    for start, stop in blocks:
        seg = own_llr[start:stop]
        out = np.full(stop - start, X, dtype=np.int8)
        strong = np.abs(seg) >= threshold
        out[strong & (seg > 0)] = H
        out[strong & (seg < 0)] = A
        if hold:
            known = np.flatnonzero(out != X)
            if len(known):
                carry = np.maximum.accumulate(np.where(out != X, np.arange(len(out)), -1))
                carry[carry < 0] = known[0]
                out = out[carry]
        codes[start:stop] = out
    return codes


def predict_codes(tracking, config=DEFAULT, variant="decoder", blocks=None, features=None,
                  external_dead=None, external_own=None):
    """(codes int8 H0 A1 D2 X3, features, evidence) for one recording.

    `external_own` (optional, evidence units, > 0 favours home) is added to the ownership channel; None leaves the
    output byte-identical. It is the hook for the experimental Smart Tagging prior (solution/smart_tagging.py)."""
    if variant not in VARIANTS:
        raise ValueError("variant must be one of {}".format(VARIANTS))
    blocks = block_ranges() if blocks is None else blocks
    features = compute_features(tracking, blocks, config) if features is None else features

    # Ownership evidence: the Phase 4.2 evidence, or the Phase 4.4 controlled-ownership refinement.
    signals = None
    if variant == "hard_decoder":
        hard = hard_nearest_evidence(features, config)
        own_llr, own_weight, own_parts = hard.own_llr, hard.own_weight, {}
    elif config.ownership_v2:
        own_llr, own_weight, own_parts, signals = control_evidence(tracking, features, blocks, config)
    else:
        own_llr, own_weight, own_parts = ownership_evidence(features, config)

    if external_own is not None:
        external_own = np.asarray(external_own, dtype=np.float64)
        if external_own.shape != own_llr.shape:
            raise ValueError("external ownership evidence has shape {}, expected {}".format(external_own.shape, own_llr.shape))
        external_own = np.nan_to_num(external_own, nan=0.0)
        own_llr = own_llr + external_own
        # Neutrality to the dead state: with the emissions [own/2, -own/2, dead] a bare ownership term would give the
        # favoured team a bonus over D and so shorten a dead state carried through an evidence-free stretch. Adding
        # |prior|/2 to the dead channel keeps "favoured team versus dead" exactly as it was; the prior then only
        # moves the home/away decision. (Found by test before any evaluation; see PHASE_4_9_PREREGISTRATION.md, amendment.)
        tag_dead = 0.5 * np.abs(external_own)
        external_dead = tag_dead if external_dead is None else np.asarray(external_dead, dtype=np.float64) + tag_dead

    dead = None
    observation = None
    if config.dead_model not in ("legacy", "oa"):
        raise ValueError("dead_model must be 'legacy' or 'oa'")
    if config.dead_model == "oa":
        # Phase 4.6: observation-aware fusion. Replaces the Phase 4.3 module and its visibility
        # compensation: unobserved channels contribute exactly 0, nothing is added for a hidden ball.
        channels, ratio = stoppage_channels(tracking, features, blocks, config)
        restart_events = []
        if config.use_restart:
            channel, restart_events = restart_channel(tracking, features, blocks, ratio, config)
            channels["restart"] = channel
        observation = fuse(list(channels.values()), tracking.n_frames)
        external_dead = observation.dead_net if external_dead is None else external_dead + observation.dead_net
    elif config.use_dead_ball_module:
        # Phase 4.3: movement-based dead-ball evidence, through the existing external hook.
        dead = dead_ball_evidence(tracking, blocks, config)
        score = dead.llr
        if config.dead_compensate_missing_ownership and variant != "hard_decoder":
            score = score - ownership_compensation(own_llr, own_weight, blocks, config)
        external_dead = score if external_dead is None else external_dead + score

    if config.dead_model == "oa":
        dead_llr = np.nan_to_num(np.asarray(external_dead, dtype=np.float64), nan=0.0)   # the outside cue is a channel here
    else:
        dead_llr = dead_evidence(features, config, external_dead)
    if config.duration_model in ("hysteresis", "both"):
        dead_llr = hysteresis_dead(dead_llr, blocks, config)
    evidence = Evidence(own_llr=own_llr, own_weight=own_weight, dead_llr=dead_llr, parts=own_parts)
    if dead is not None:
        evidence.parts["dead_ball"] = dead
    if observation is not None:
        evidence.parts["dead_observation"] = observation
        evidence.parts["restart_events"] = restart_events

    switch_extra = None
    if config.use_handover and variant == "decoder":
        if signals is None:
            signals = control_signals(tracking, features, blocks, config)
        handovers = handover_costs(signals, blocks, config)
        evidence.parts["handovers"] = handovers
        switch_extra = handovers.switch_extra

    if variant in ("decoder", "hard_decoder"):
        codes = decode_blocks(evidence.own_llr, evidence.dead_llr, blocks, config, switch_extra)
    else:
        codes = evidence_labels(evidence.own_llr, blocks, config.abstain_below,
                                hold=(variant == "evidence_hold"))
    return codes, features, evidence


def validate_codes(codes, n_expected=None):
    """Raise ValueError unless `codes` is one int8-compatible label code (0..3) per video frame."""
    n_expected = games.GAMES[games.GAME]["n_video_frames"] if n_expected is None else n_expected
    codes = np.asarray(codes)
    if codes.ndim != 1 or len(codes) != n_expected:
        raise ValueError("expected {} labels, got shape {}".format(n_expected, codes.shape))
    if not np.issubdtype(codes.dtype, np.integer) or codes.min() < 0 or codes.max() > 3:
        raise ValueError("label codes must be integers in 0..3 (H, A, D, X)")
    return codes


def _protected(path):
    """Why `path` must never be overwritten by a prediction run, or None. Protects the restricted data,
    the ground truths, the starter code and the committed official baselines."""
    real = os.path.realpath(path)
    folders = [os.path.realpath(games.DATA_DIR), os.path.join(ROOT, "drive-file"), os.path.join(ROOT, "evaluation", "gt"),
               os.path.join(ROOT, "starter")]
    for folder in folders:
        if real == folder or real.startswith(folder + os.sep):
            return "inside the protected folder " + os.path.relpath(folder, ROOT)
    results = os.path.join(ROOT, "evaluation", "results")
    if os.path.dirname(real) == results and os.path.basename(real).startswith("baseline_"):
        return "an official committed baseline"
    return None


def write_submission(path, codes):
    """The whole submission format: `frame,label`, one row per video frame. Written atomically
    (temporary file in the same folder, then rename), so an interrupted run never leaves a truncated
    file, and never over the restricted data, ground truths or official baselines."""
    reason = _protected(path)
    if reason:
        raise ValueError("refusing to write {}: {}".format(path, reason))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    letters = np.array(list(LABELS))[codes]
    handle, tmp = tempfile.mkstemp(dir=folder, prefix=".tmp_submission_", suffix=".csv")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as out:
            out.write("frame,label\n")
            out.writelines("{},{}\n".format(i, letter) for i, letter in enumerate(letters))
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)          # mkstemp creates 0600; a submission file should look like any other
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_memory_mb():
    """Peak resident set size of this process in MB (None where the platform has no `resource`)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default="decoder")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="named configuration (default: legacy_p42, the Phase 4.2 model; the release candidate is rc1)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a config field (repeatable)")
    parser.add_argument("--describe", metavar="PRESET", choices=sorted(PRESETS),
                        help="print the registry card of a preset and exit (no data needed)")
    parser.add_argument("--list-presets", action="store_true", help="list the registered presets and exit")
    args = parser.parse_args(argv)

    if args.list_presets:
        for name in sorted(PRESETS):
            print("{:<28s} {}".format(name, model_registry.get(name).status))
        return 0
    if args.describe:
        print(model_registry.describe(args.describe))
        return 0

    refused = _protected(args.out)
    if refused:
        print("error: refusing to write {}: {}".format(args.out, refused), file=sys.stderr)
        return 2
    if args.preset is None:
        args.preset = "legacy_p42"
        print("note: no --preset given; using legacy_p42 (Phase 4.2). The release candidate is --preset rc1.")
    try:
        config = parse_overrides(PRESETS[args.preset], args.set)
    except (KeyError, ValueError) as error:
        print("error: {}".format(error.args[0] if error.args else error), file=sys.stderr)
        return 2
    if args.set:
        print("note: --set overrides applied on top of {}; the output is no longer that preset.".format(args.preset))
    safety = check_transition_safety(config)
    if not safety["safe"]:
        print("WARNING: a direct home<->away switch ({direct}) is not cheaper than the cheapest route through "
              "the dead state ({cheapest_detour}); expect phantom stoppages.".format(**safety))
    print("preset: {}  fingerprint {}".format(args.preset, model_registry.config_fingerprint(config)[:16]))
    print("configuration:\n" + config.describe())
    start = time.time()
    try:
        tracking = load_tracking(config=config)
    except (IOError, OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    loaded = time.time()
    external_own = None
    tag_config = smart_tagging.TAG_PRESETS.get(args.preset)
    if tag_config is not None:
        try:
            external_own, tag_report = smart_tagging.tag_prior(
                smart_tagging.require_file(), tracking.n_frames, block_ranges(), config.switch_cost, tag_config)
        except (IOError, OSError, ValueError) as error:
            print("error: {}".format(error), file=sys.stderr)
            return 2
        print("smart tagging prior (experimental): {}".format(tag_report))
    codes, _, _ = predict_codes(tracking, config, args.variant, external_own=external_own)
    done = time.time()
    validate_codes(codes, tracking.n_frames)
    try:
        write_submission(args.out, codes)
    except ValueError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    counts = {LABELS[k]: int(np.sum(codes == k)) for k in range(4)}
    print("\nvariant {}: {}".format(args.variant, counts))
    print("load {:.1f}s  features+decode {:.1f}s  write {:.1f}s  total {:.1f}s  peak memory {} MB".format(
        loaded - start, done - loaded, time.time() - done, time.time() - start,
        "n/a" if peak_memory_mb() is None else "{:.0f}".format(peak_memory_mb())))
    print("wrote", os.path.relpath(args.out, ROOT))
    print("sha256", file_sha256(args.out))
    print("check: python evaluation/validate_submission.py", os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
