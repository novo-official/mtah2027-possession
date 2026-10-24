"""Every tunable number of the possession pipeline, in one place.

Nothing here was fitted. The defaults are physically motivated round numbers (a player's reach,
a sprint speed, how long a team needs the ball before we call it a possession), and each one is
explained next to its definition. `python -m solution.predict --set name=value` overrides any of
them for sensitivity runs; the effective configuration is printed on every run so a result can
always be traced to its settings.

Units: metres, seconds, frames at `fps`, and "evidence units". An evidence unit is a log-likelihood
-ratio-like score (nats). It is **not** a calibrated probability: no calibration has been
demonstrated, and nothing downstream may treat it as one. Its only job is to make "how much more
does this frame favour one explanation than the other" additive across frames.
"""

from dataclasses import dataclass, fields, replace


@dataclass(frozen=True)
class Config:
    # -- data ---------------------------------------------------------------------------------
    fps: int = 25
    # Coordinates this far (m) outside the pitch rectangle are treated as tracker errors and
    # become missing. The ball legitimately travels well outside the lines (into the stands, to a
    # ball boy), so the margin must be generous: the real tracking has ball positions more than
    # 30 m outside the pitch that are out of play, not errors, and discarding them would throw
    # away the strongest dead-ball cue. 50 m is stadium scale; anything beyond is an error.
    coord_margin_m: float = 50.0
    # Velocity caps (m/s). Values above are position jumps (identity switches, re-acquisitions),
    # not motion, so the velocity is discarded rather than clipped. 45 m/s is above the fastest
    # recorded shots; 12 m/s is above a sprint.
    ball_max_speed: float = 45.0
    player_max_speed: float = 12.0
    # Half-width (frames) of the central difference used for velocities: v = (x[t+k]-x[t-k])/(2k/fps).
    # k=4 spans 0.32 s, short enough to follow a pass, long enough to average out position jitter.
    # The tracker's own ball speed column is ~91% empty and is never used.
    velocity_halfwidth: int = 4

    # -- ownership evidence ---------------------------------------------------------------------
    # evidence = gain * weight * tanh(margin / margin_scale), margin = d_away - d_home (metres, >0
    # favours home), weight = proximity * motion * observation * consistency, each in [0, 1].
    #
    # Distance margin at which the nearer team's claim is about 76% saturated (tanh(1)=0.76). A
    # few metres is roughly the distance a defender can close while a pass is in flight.
    margin_scale_m: float = 3.0
    # Control radius: how far the nearest player can be before the ball is "loose" and the frame
    # stops being informative about who controls it. proximity = exp(-(d_min/d0)^2).
    control_radius_m: float = 4.0
    # Speed (m/s) of the ball relative to the nearest player at which motion evidence is halved:
    # motion = 1 / (1 + (rel_speed/v0)^2). A dribble keeps the ball within a couple of m/s of its
    # carrier; a pass or clearance is far faster, and during flight "nearest" says little.
    relative_speed_scale: float = 6.0
    # Motion factor used when the relative speed cannot be computed (a velocity is unavailable).
    # Deliberately below 1: unknown motion is not the same as confirmed control.
    motion_unknown: float = 0.7
    # Teams with fewer valid tracked players than this have their weight scaled down linearly,
    # reaching 1.0 at `full_players`. A margin computed against a nearly empty team is unreliable.
    full_players: int = 8
    # Temporal consistency: within +/- this many frames, the agreement of the sign of the
    # evidence sets a factor in [consistency_floor, 1]. Flickering evidence (contested balls)
    # is down-weighted; set consistency_halfwidth=0 to disable.
    consistency_halfwidth: int = 12
    consistency_floor: float = 0.5
    # Scale of the evidence in nats per frame at weight 1 and saturated margin.
    evidence_gain: float = 3.0

    # -- dead-ball evidence (interface + one deliberately weak physical cue) -------------------------
    # Ball absence is NEVER dead-ball evidence. The only built-in cue is a ball *tracked* outside
    # the pitch lines by more than this margin (m): physically out of play, rare among alive
    # frames, but present on only a minority of dead frames. Everything else (collective player
    # speed, restarts, tags) enters later through `dead_llr` without touching the decoder.
    outside_margin_m: float = 1.0
    outside_dead_gain: float = 2.0     # nats per frame while the ball is outside by that margin
    use_outside_cue: bool = True

    # -- dead-ball evidence module (solution/dead_ball.py) --------------------------------------
    # Master switch. False reproduces the Phase 4.2 model exactly; each cue below is also
    # independently switchable so every one can be ablated. Scores are signed (>0 favours dead,
    # <0 favours alive) and multiplied by a reliability in [0, 1]; where the tracking gives no
    # usable observation the score is exactly 0 ("I do not know"), never a confident dead call.
    use_dead_ball_module: bool = False
    dead_use_activity: bool = True        # collective player movement
    dead_use_near_ball: bool = False      # movement of the players around the ball (off: see README)
    dead_use_ball_motion: bool = False    # reliable ball speed (off by default: see README)
    dead_allow_negative: bool = True      # let high activity count as evidence for alive play
    # Player speeds come from positions (central difference), accepted for a slot only when every
    # frame of the difference window is observed and no single step exceeds `max_step_m`. A larger
    # step is an identity switch or re-acquisition, not motion, and a NaN gap inside the window
    # could hide one. "native" uses the tracker's own speed column instead (ablation only).
    dead_speed_source: str = "positions"
    max_step_m: float = 1.0               # per frame; 25 m/s, beyond any player
    ball_max_step_m: float = 2.0          # per frame; 50 m/s, beyond any kick
    # A frame needs at least this many valid player speeds for its collective statistic (median
    # over players, robust to a few bad tracks). Fewer: no statistic, no evidence.
    min_speed_players: int = 8
    # Temporal aggregation: rolling median of the per-frame statistic over +/- this many frames
    # (6 s total), computed inside each half only. A stoppage lasts seconds; a pause in a
    # run of play does not, so short fluctuations are averaged out.
    dead_window_halfwidth: int = 75
    # The aggregate needs this fraction of valid frames in its window, else it is unavailable.
    dead_min_coverage: float = 0.5
    # Walking vs jogging boundary (m/s) and the width over which the score saturates:
    # score = tanh((walk_speed_ref - speed) / walk_speed_scale). Players walk at ~1-1.5 m/s
    # during a stoppage and jog at 2-3.5 m/s or more in play; 1.6 sits between.
    walk_speed_ref: float = 1.6
    walk_speed_scale: float = 0.8
    # Reference for "slow": "absolute" uses walk_speed_ref above for the whole match. "relative"
    # compares the aggregate with the local live-play level L, the `dead_level_quantile` of the same
    # series over +/- `dead_level_halfwidth` frames (3 min), and scores tanh((relative_ref - v/L) /
    # relative_scale). Tempo varies a lot (fresh legs early, slow build-up later), so a fixed
    # speed over-reads slow live play as a stoppage; stoppages are short, so a high quantile of a
    # long window still tracks the live level. Needs no labels.
    dead_speed_reference: str = "relative"
    dead_level_halfwidth: int = 4500
    dead_level_quantile: float = 0.75
    dead_level_min_frames: int = 750       # fewer observed frames in the level window: no evidence
    relative_ref: float = 0.7
    relative_scale: float = 0.3
    # Players closer than this to the ball (m) form the "around the ball" group; at least
    # `near_ball_min_players` of them must have a valid speed, and the rolling statistic needs
    # `near_ball_min_coverage` of its window (lower: it is only defined while the ball is seen).
    near_ball_radius_m: float = 10.0
    near_ball_min_players: int = 3
    near_ball_min_coverage: float = 0.3
    # Ball speed (m/s) boundary and width for the optional ball-motion cue.
    ball_speed_ref: float = 4.0
    ball_speed_scale: float = 2.0
    # Visibility compensation. In the decoder, observed ownership evidence gives the winning
    # alive state a bonus of |own|/2; with no ball there is no bonus, which silently lowers the
    # bar for a dead call although nothing about play changed (masking the ball alone raised the
    # alive frames labelled dead from 9% to 32% in the Phase 4.3 robustness test). With this on,
    # frames without ownership evidence receive the average bonus observed elsewhere in the
    # same half as a dead-score offset, so visibility alone does not move the alive/dead balance.
    dead_compensate_missing_ownership: bool = True
    # Evidence units per frame at full reliability and saturated score.
    activity_gain: float = 0.6
    near_ball_gain: float = 0.6
    ball_motion_gain: float = 0.3
    # Alive-favouring (negative) evidence is scaled by this factor (1.0 = symmetric). Ownership
    # evidence already supports alive play, so extra alive evidence is optional.
    alive_evidence_scale: float = 1.0

    # -- controlled-ownership evidence and handover logic (solution/ownership.py, handover.py) ---
    # Master switch for the Phase 4.4 ownership evidence. False keeps the Phase 4.2 / 4.3 evidence
    # exactly. When True, the three features below start from the same claim / proximity /
    # observation / consistency terms as the original and change one aspect each.
    ownership_v2: bool = False
    # Relative motion: the motion factor uses the strictly valid velocity of the nearest player of
    # the team that claims the ball (not the nearest overall), so a ball and a player moving
    # together count as control and a ball flying past a player does not.
    own_strict_motion: bool = True
    # Track stability: a claiming player's track that was only just (re)acquired is trusted less:
    # factor = stability_floor + (1 - stability_floor) * min(1, age / stability_age_full).
    own_track_stability: bool = True
    stability_age_full: int = 25
    stability_floor: float = 0.5
    # Controlled reception: a team "has contact" while its nearest player is within
    # `contact_radius_m` of the ball AND moves with it (relative speed below `contact_rel_speed`,
    # a ball at the feet or being carried). Contact interrupted for at most `episode_bridge`
    # frames is one episode; its strength is 1 - exp(-contact_frames / episode_scale), so a single
    # touch is weak and a second of control is strong. Evidence = reception_gain * (S_home - S_away).
    own_reception: bool = True
    contact_radius_m: float = 2.0
    contact_rel_speed: float = 4.0
    episode_bridge: int = 6
    episode_scale: float = 12.0
    reception_gain: float = 1.5
    # Handover logic (needs the episodes above). An episode with strength >= `established_strength`
    # is an established period of control. Between two consecutive established episodes of
    # different teams, switching to the later team is made cheaper: the home<->away switch cost
    # is raised by handover_extra * switch_cost (capped below the dead-state detour) everywhere except in a ramp that falls back to the
    # normal cost over the `handover_slack` frames before the new episode begins (and for
    # `handover_pad` frames after). Unsupported switches therefore cost more; supported ones
    # cost the normal amount; a missing ball never lowers any cost.
    use_handover: bool = False
    established_strength: float = 0.6
    handover_slack: int = 75
    handover_pad: int = 12
    handover_extra: float = 0.25

    # -- decoder ------------------------------------------------------------------------------
    # Additive costs (nats) for entering a state from another one. Staying is free. All finite:
    # an unsupported transition is penalised, never forbidden.
    #
    # Home<->away hand-over. About 12 nats is ~8 frames (0.3 s) of strong, consistent evidence
    # (about 1.5 nats/frame), or several seconds of weak evidence. Real hand-overs pass this
    # easily; single-frame contested-ball flips do not.
    switch_cost: float = 12.0
    # Alive -> dead and dead -> alive are costed separately from hand-overs because they are
    # different events with different evidence. Dead evidence is weak for now, so entering is
    # not made cheap.
    enter_dead_cost: float = 10.0
    exit_dead_cost: float = 6.0
    # Log-prior over the state at the start of each half (H, A, D). Uniform: no hard-coded answer.
    initial_logprior: tuple = (0.0, 0.0, 0.0)

    # Tie-breaking in the decoder. "legacy" (default, Phase 4.2 behaviour) lets Viterbi's
    # lowest-state-index rule decide ties, which is asymmetric: across a stretch with no evidence a
    # home->away switch lands at the END of the stretch but an away->home switch at the START.
    # True breaks exact ties symmetrically toward the latest switch (possession stays with the last
    # team until the other has the ball), for both directions.
    decoder_late_ties: bool = False

    # -- dead-ball duration modelling (solution/duration.py) ---------------------------------------
    # "none" (default) keeps the first-order decoder. "hysteresis" conditions the dead-ball evidence
    # so that entering D needs sustained stoppage evidence and leaving it needs sustained live
    # evidence; "soft" decodes with a soft penalty on ending a dead interval early; "both" does both.
    # Neither imposes a minimum duration: every penalty is finite and strong evidence overrides it.
    duration_model: str = "none"
    # Hysteresis (Schmitt trigger on the rolling mean of the dead-ball score, nats per frame, over
    # +/- `hysteresis_halfwidth` frames). Armed: the mean reached `hysteresis_enter` and has not
    # since fallen below `hysteresis_exit`. Positive dead evidence outside an armed stretch is scaled
    # by `unarmed_positive_scale` (a lone high-score frame cannot start a stoppage); negative
    # (live-play) evidence inside an armed stretch is scaled by `armed_negative_scale` (small
    # fluctuations do not end it). Nothing here refers to a team, so it is symmetric by construction.
    hysteresis_halfwidth: int = 37
    hysteresis_enter: float = 0.20
    hysteresis_exit: float = 0.05
    unarmed_positive_scale: float = 0.25
    armed_negative_scale: float = 0.5
    # Soft duration prior: leaving a dead interval of age k < `soft_min_dead_frames` costs an extra
    # `short_dead_penalty * (1 - k / soft_min_dead_frames)` nats on top of `exit_dead_cost`. At 5 s
    # (125 frames) the extra cost is zero; it is never infinite, so a genuine short stoppage with
    # strong evidence is still decoded. A dead interval that reaches the start or end of a half is
    # not penalised (its true extent is unknown there).
    soft_min_dead_frames: int = 125
    short_dead_penalty: float = 6.0

    # -- observation-aware dead-ball evidence (Phase 4.6: observation.py, stoppage.py, restart.py) ----
    # Emission model for the alive-vs-dead decision. "legacy" (default) keeps the Phase 4.2-4.5
    # emissions [own/2, -own/2, dead], which are exactly equivalent to giving the dead channel an
    # implicit live-play term of -|own|/2 whenever the ball is observed (and nothing when it is not).
    # "mixture_null" (REJECTED, kept so the failure is reproducible) treats the dead state's ownership
    # likelihood as the equal ARITHMETIC mixture of the teams', a bounded bonus <= log 2 that is zero
    # when the ball is unobserved. It is not indifferent to ownership noise (the legacy emission is,
    # because it emits the geometric mean (e_H + e_A)/2 = 0), so the dead state absorbs contested
    # ownership with no dead evidence at all: 59-67 of 80 random sequences contained a phantom stoppage.
    emission_mode: str = "legacy"
    # "legacy" = the Phase 4.3 dead-ball module + visibility compensation. "oa" = observation-aware
    # fusion: every channel contributes reliability * (gain * score - zero_point), a channel that is
    # unavailable contributes exactly 0, and no visibility compensation is applied.
    dead_model: str = "legacy"
    # Zero point of the movement channel in nats per frame, scaled by that channel's reliability so
    # it vanishes where there is no movement observation (no invented evidence in unobserved gaps).
    # It is a POLICY parameter, not a derived constant: the observation-aware presets use 0.30. The
    # counterfactual sweep (README) shows a cliff between 0.2 (sensitive, hiding the ball adds 12
    # points of false stoppages) and 0.4 (safe, but visible detection collapses to ~5%).
    activity_zero: float = 0.0
    # Movement statistic: "pooled" = median speed of all players (Phase 4.3); "both_teams" = the
    # larger of the two teams' median speeds, so the score is dead-like only if BOTH teams are slow.
    oa_activity_stat: str = "pooled"
    # The ball must stay outside the lines by more than `outside_margin_m` for at least this many
    # consecutive frames before the out-of-play cue counts (1 = the legacy cue). Rejects line jitter.
    oa_outside_min_run: int = 1
    # Restart evidence: negative dead evidence over `restart_pulse` frames after play resumes. A
    # restart is a sustained rise of the movement ratio (from <= restart_low to >= restart_high,
    # medians over `restart_window` frames) CONFIRMED by the ball (reliable renewed ball motion, or the
    # start of established contact). Players moving while the ball is visible and still is not a
    # restart. With the ball hidden throughout, movement alone counts at `restart_movement_only_scale`.
    use_restart: bool = False
    restart_window: int = 38
    restart_low: float = 0.6
    restart_high: float = 0.85
    restart_ball_speed: float = 5.0
    restart_ball_frames: int = 5
    restart_search_before: int = 50
    restart_search_after: int = 75
    restart_pulse: int = 50
    restart_gain: float = 1.5
    restart_movement_only_scale: float = 0.5

    # -- evidence-only output (ablation) --------------------------------------------------------
    # Frames with |evidence| below this get X in the evidence-only variants.
    abstain_below: float = 0.25

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def describe(self):
        return "\n".join("  {:<22s} {}".format(f.name, getattr(self, f.name)) for f in fields(self))


DEFAULT = Config()


_TRUE = frozenset(("1", "true", "yes", "on"))
_FALSE = frozenset(("0", "false", "no", "off"))


def parse_overrides(config, pairs):
    """Config with `name=value` strings applied, typed against the field defaults."""
    types = {f.name: type(getattr(config, f.name)) for f in fields(config)}
    updates = {}
    for pair in pairs or ():
        name, _, text = pair.partition("=")
        if name not in types:
            raise KeyError("unknown config field {!r}; have {}".format(name, sorted(types)))
        kind = types[name]
        if kind is bool:
            word = text.strip().lower()
            if word not in _TRUE and word not in _FALSE:
                raise ValueError("{}: {!r} is not a boolean; use one of {}".format(name, text, sorted(_TRUE | _FALSE)))
            updates[name] = word in _TRUE
        elif kind is tuple:
            updates[name] = tuple(float(x) for x in text.split(","))
        else:
            updates[name] = kind(text)
    return config.replace(**updates)


# Named reference configurations. "legacy_*" reproduce the Phase 4.2 / 4.3 outputs byte for byte;
# the others are explicit, versioned experimental references and are opt-in.
PRESETS = {
    "legacy_p42": DEFAULT,
    "legacy_p43": DEFAULT.replace(use_dead_ball_module=True),
    "corrected": DEFAULT.replace(decoder_late_ties=True),
    "corrected_dead": DEFAULT.replace(decoder_late_ties=True, use_dead_ball_module=True),
    "duration_only": DEFAULT.replace(use_dead_ball_module=True, duration_model="soft"),
    "combined": DEFAULT.replace(decoder_late_ties=True, use_dead_ball_module=True, duration_model="soft"),
}
# Phase 4.6 observation-aware references (all use the corrected decoder semantics; the legacy
# dead-ball module, its visibility compensation and the rejected duration / ownership options are off).
_OA = DEFAULT.replace(decoder_late_ties=True, dead_model="oa", activity_zero=0.30,
                      dead_use_near_ball=False, dead_use_ball_motion=False,
                      dead_compensate_missing_ownership=False)
PRESETS.update({
    "oa_only": _OA,
    "oa_entry": _OA.replace(oa_activity_stat="both_teams", oa_outside_min_run=12),
    "oa_restart": _OA.replace(use_restart=True),
    "oa_full": _OA.replace(oa_activity_stat="both_teams", oa_outside_min_run=12, use_restart=True),
    # documented as REJECTED: the mixture-null emission lets the dead state absorb ownership noise
    "oa_mixture_null_rejected": _OA.replace(emission_mode="mixture_null"),
})
# Phase 4.5 hysteresis reference (EXPERIMENTAL, neutral on the public window): the corrected decoder with the Phase 4.3
# dead-ball module and Schmitt-trigger conditioning of the dead-ball evidence. It existed only as an ad-hoc
# `--set duration_model=hysteresis` run; the name makes it reproducible. Output hash recorded in model_registry.py.
PRESETS["corrected_dead_hysteresis"] = PRESETS["corrected_dead"].replace(duration_model="hysteresis")

# -- Phase 4.8 release candidate ------------------------------------------------------------------------------------
# RC1 is written out field by field on purpose: it must not change when an experimental preset above is edited.
# Enabled : ownership evidence (Phase 4.2), the physical out-of-play cue (ball tracked > 1 m outside the lines),
#           the corrected decoder semantics (`decoder_late_ties`, Phase 4.5). Everything else is the Phase 4.2 default.
# Disabled: Phase 4.3 movement dead-ball module, Phase 4.6 observation-aware fusion, restart evidence, ownership_v2,
#           handover costs, duration models. Frozen: model_registry.config_fingerprint() is pinned in tests/.
# Its prediction is identical to `corrected` today; the two are pinned independently so they can diverge deliberately.
RC1 = Config(decoder_late_ties=True, use_dead_ball_module=False, dead_model="legacy", ownership_v2=False,
             use_handover=False, duration_model="none", use_restart=False, emission_mode="legacy")
PRESETS["rc1"] = RC1

# -- Phase 4.9 experimental candidate ------------------------------------------------------------------------------
# Uses the RC1 configuration unchanged (written out field by field again, so it cannot follow edits to `RC1`); the
# Smart Tagging prior itself is configured in solution/smart_tagging.py (TagConfig / TAG_PRESETS), NOT here, so the
# fingerprint of every Config above stays as pinned. Without the prior this preset is byte-identical to rc1.
PRESETS["rc2_tag_experimental"] = Config(decoder_late_ties=True, use_dead_ball_module=False, dead_model="legacy",
                                         ownership_v2=False, use_handover=False, duration_model="none",
                                         use_restart=False, emission_mode="legacy")
