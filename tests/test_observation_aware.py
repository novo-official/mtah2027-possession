"""Phase 4.6 tests: observation-aware dead-ball evidence, stoppage-entry channels and restart evidence.

Synthetic scenarios verify implementation invariants (neutrality of unobserved channels, visibility
invariance of the dead-vs-live decision for identical player evidence, restart rules, symmetry).
They are not evidence of accuracy on a real match.

Run with:   .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import test_dead_ball as tdb  # noqa: E402
from solution import data, decoder, duration, features, observation, predict, restart, stoppage  # noqa: E402
from solution.config import DEFAULT, PRESETS  # noqa: E402

OA = PRESETS["oa_only"]
N = 6000


def circles(n, speeds_by_slot, radius=8.0):
    """(n, S, 2) players on circles; speeds_by_slot is (n, S) m/s."""
    theta = np.cumsum(speeds_by_slot / radius / 25.0, axis=0)
    rng = np.random.RandomState(7)
    k = speeds_by_slot.shape[1]
    centres = np.stack([rng.uniform(20, 85, k), rng.uniform(15, 53, k)], axis=1)
    phase = rng.uniform(0, 2 * np.pi, k)
    ang = theta + phase[None, :]
    return (centres[None] + radius * np.stack([np.cos(ang), np.sin(ang)], axis=-1)).astype(np.float32)


def two_team_scene(n, home_speed, away_speed):
    speeds = np.concatenate([np.tile(np.asarray(home_speed, float)[:, None], (1, 11)),
                             np.tile(np.asarray(away_speed, float)[:, None], (1, 11))], axis=1)
    players = circles(n, speeds)
    ball = np.tile(np.array([3.0, 3.0], dtype=np.float32), (n, 1))
    return data.Tracking(players, np.zeros(players.shape[:2], np.float32), ball, np.array([0] * 11 + [1] * 11, np.int8),
                         np.zeros(22, bool), (105.3, 68.0), 25)


def decode(tracking, config=OA, blocks=None):
    blocks = blocks or [(0, tracking.n_frames)]
    return predict.predict_codes(tracking, config, "decoder", blocks)[0]


class TestEmissions(unittest.TestCase):
    def test_legacy_mode_is_the_original_formula(self):
        own, dead = np.array([2.0, -1.0, 0.0]), np.array([0.5, 0.0, 1.0])
        np.testing.assert_allclose(decoder.build_emissions(own, dead), [[1.0, -1.0, 0.5], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])

    def test_legacy_equals_normalised_emissions_with_an_implicit_live_term(self):
        rng = np.random.RandomState(0)
        own, dead = rng.randn(500) * 2, rng.randn(500)
        legacy = decoder.build_emissions(own, dead)
        norm = np.stack([np.minimum(own, 0), np.minimum(-own, 0), dead - 0.5 * np.abs(own)], axis=1)
        diff = legacy - norm
        np.testing.assert_allclose(diff, diff[:, :1] * np.ones((1, 3)), atol=1e-12)      # a per-frame constant
        np.testing.assert_allclose(diff[:, 0], 0.5 * np.abs(own))
        cost = decoder.transition_costs(DEFAULT)
        np.testing.assert_array_equal(decoder.viterbi(legacy, cost, np.zeros(3)), decoder.viterbi(norm, cost, np.zeros(3)))

    def test_mixture_null_keeps_home_versus_away_and_is_bounded_and_zero_without_evidence(self):
        own = np.linspace(-9, 9, 41)
        em = decoder.build_emissions(own, np.zeros(41), "mixture_null")
        np.testing.assert_allclose(em[:, 0] - em[:, 1], own, atol=1e-12)
        bonus = np.maximum(em[:, 0], em[:, 1]) - em[:, 2]
        self.assertTrue((bonus >= -1e-12).all() and (bonus < np.log(2) + 1e-12).all())
        self.assertAlmostEqual(float(decoder.build_emissions(np.zeros(1), np.zeros(1), "mixture_null").any()), 0.0)
        with self.assertRaises(ValueError):
            decoder.build_emissions(own, own, "nope")

    def test_mixture_null_is_rejected_because_it_invents_stoppages_from_ownership_noise(self):
        rng = np.random.RandomState(11)
        phantom = {"legacy": 0, "mixture_null": 0}
        for _ in range(30):
            own = rng.randn(2000) * rng.choice([0.8, 1.5, 3.0])
            for mode in phantom:
                codes = decoder.decode_blocks(own, np.zeros(2000), [(0, 2000)], DEFAULT.replace(emission_mode=mode))
                phantom[mode] += int((codes == decoder.D).any())
        self.assertEqual(phantom["legacy"], 0)             # the legacy emission is indifferent to ownership noise
        self.assertGreater(phantom["mixture_null"], 10)    # the rejected one is not


class TestFusion(unittest.TestCase):
    def test_an_unobserved_channel_contributes_exactly_zero_even_with_a_finite_score_and_a_zero_point(self):
        score = np.array([1.0, np.nan, 1.0, 1.0])
        rel = np.array([1.0, 1.0, 0.0, 0.5])
        ch = observation.Channel("c", score, rel, gain=2.0, zero=0.3)
        np.testing.assert_allclose(ch.contribution(), [1.7, 0.0, 0.0, 0.5 * 1.7])

    def test_the_zero_point_scales_with_reliability_so_it_never_acts_on_unobserved_stretches(self):
        ch = observation.Channel("c", np.zeros(4), np.array([1.0, 0.5, 0.0, 1.0]), gain=1.0, zero=0.4)
        np.testing.assert_allclose(ch.contribution(), [-0.4, -0.2, 0.0, -0.4])

    def test_fusion_adds_channels_and_records_what_was_observed_by_source(self):
        a = observation.Channel("a", np.array([1.0, np.nan]), np.array([1.0, 1.0]), 1.0, source="player")
        b = observation.Channel("b", np.array([1.0, 1.0]), np.array([0.0, 1.0]), 1.0, source="ball")
        ev = observation.fuse([a, b], 2)
        np.testing.assert_allclose(ev.dead_net, [1.0, 1.0])
        self.assertEqual(ev.player_supported.tolist(), [True, False])
        self.assertEqual(ev.ball_supported.tolist(), [False, True])
        self.assertEqual(ev.supported.tolist(), [True, True])
        self.assertTrue(set(ev.channels) == {"a", "b"})


class TestStoppageChannels(unittest.TestCase):
    def test_pooled_statistic_reads_one_slow_team_as_dead_but_both_teams_does_not(self):
        n = 8000
        home = np.where(np.arange(n) < 4000, 4.0, 0.6)             # from frame 4000 only the home team slows
        away = np.full(n, 4.0)
        t = two_team_scene(n, home, away)
        f = features.compute_features(t, [(0, n)], DEFAULT)
        pooled = stoppage.activity_channel(t, 0, n, DEFAULT.replace(oa_activity_stat="pooled"))[0]
        both = stoppage.activity_channel(t, 0, n, DEFAULT.replace(oa_activity_stat="both_teams"))[0]
        self.assertGreater(float(np.nanmedian(pooled[5000:7500])), 0.0)       # looks like a stoppage
        self.assertLess(float(np.nanmedian(both[5000:7500])), 0.0)            # but one team is still playing
        with self.assertRaises(ValueError):
            stoppage.activity_channel(t, 0, n, DEFAULT.replace(oa_activity_stat="nope"))

    def test_activity_is_unavailable_without_player_speeds(self):
        t = tdb.scene(3.0, n=3000)
        players = t.player_xy.copy()
        players[:] = np.nan
        t2 = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps)
        f = features.compute_features(t2, [(0, 3000)], DEFAULT)
        channels, ratio = stoppage.stoppage_channels(t2, f, [(0, 3000)], OA)
        self.assertTrue((channels["activity"].reliability == 0).all())
        self.assertTrue((channels["activity"].contribution() == 0).all())

    def test_outside_cue_needs_persistence_and_is_unavailable_when_the_ball_is_unobserved(self):
        t = tdb.scene(3.0, n=200)
        ball = t.ball_xy.copy()
        ball[20:23] = (52.0, 71.5)                                  # 3-frame excursion past the touchline: jitter
        ball[60:100] = (52.0, 71.5)                                 # 40 frames: ball really out
        ball[120:140] = np.nan                                      # ball unobserved
        t2 = data.Tracking(t.player_xy, t.player_speed, ball, t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps)
        f = features.compute_features(t2, [(0, 200)], DEFAULT)
        cfg = OA.replace(oa_outside_min_run=12)
        score, rel = stoppage.outside_channel(f, 0, 200, cfg)
        self.assertEqual(float(score[21]), 0.0)                     # jitter rejected
        self.assertEqual(float(score[80]), 1.0)                     # persistent: counts
        self.assertTrue(np.isnan(score[130]) and rel[130] == 0.0)   # unobserved: unavailable, never a cue
        legacy_like = stoppage.outside_channel(f, 0, 200, OA.replace(oa_outside_min_run=1))[0]
        self.assertEqual(float(legacy_like[21]), 1.0)

    def test_switches_remove_channels_and_blocks_are_isolated(self):
        t = tdb.scene(3.0, n=3000)
        f = features.compute_features(t, [(0, 3000)], DEFAULT)
        none, _ = stoppage.stoppage_channels(t, f, [(0, 3000)], OA.replace(dead_use_activity=False, use_outside_cue=False))
        self.assertEqual(none, {})
        both, _ = stoppage.stoppage_channels(t, f, [(0, 1500), (1700, 3000)], OA)
        self.assertTrue(np.isnan(both["activity"].score[1550]) and both["activity"].reliability[1550] == 0)


class TestRestartEvidence(unittest.TestCase):
    def ratio_step(self, before=0.4, after=1.0, n=600, at=300):
        r = np.full(n, before)
        r[at:] = after
        return r

    def test_a_sustained_rise_from_play_slow_to_play_level_is_a_candidate_and_noise_is_not(self):
        cands = restart.rise_candidates(self.ratio_step(), DEFAULT)
        self.assertEqual(len(cands), 1)
        self.assertLessEqual(abs(int(cands[0]) - 300), DEFAULT.restart_window + 5)
        flat = np.full(600, 0.4)
        self.assertEqual(len(restart.rise_candidates(flat, DEFAULT)), 0)
        spike = np.full(600, 0.4)
        spike[300:306] = 1.2
        self.assertEqual(len(restart.rise_candidates(spike, DEFAULT)), 0)
        slow_rise = self.ratio_step(before=0.7, after=1.0)                 # was not a stoppage: starts above restart_low
        self.assertEqual(len(restart.rise_candidates(slow_rise, DEFAULT)), 0)

    def restart_scene(self, ball):
        """Stoppage (players walk) from 1500, restart (players run) at 3500; ball behaviour varies."""
        n = N
        speed = tdb.profile(n, [(0, n, 4.0), (1500, 3500, 0.7)])
        t = tdb.scene(speed, n=n, spot=(1500, 3500))
        ball_xy = t.ball_xy.copy()
        if ball == "moves_at_restart":
            ball_xy[3500:3560] = np.stack([3.0 + 0.4 * np.arange(60), np.full(60, 3.0)], axis=1)       # 10 m/s
        elif ball == "stationary_visible":
            ball_xy[3500:] = (3.0, 3.0)
        elif ball == "hidden":
            ball_xy[1500:] = np.nan
        return data.Tracking(t.player_xy, t.player_speed, ball_xy.astype(np.float32), t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps)

    def run_restart(self, tracking, config):
        blocks = [(0, tracking.n_frames)]
        f = features.compute_features(tracking, blocks, config)
        _, ratio = stoppage.stoppage_channels(tracking, f, blocks, config)
        return restart.restart_channel(tracking, f, blocks, ratio, config)

    def test_renewed_ball_motion_confirms_a_restart_and_the_pulse_is_negative_and_bounded(self):
        channel, events = self.run_restart(self.restart_scene("moves_at_restart"), OA.replace(use_restart=True))
        self.assertTrue(any(kind == "ball" for _, kind in events))
        frame = [f for f, k in events if k == "ball"][0]
        self.assertLessEqual(abs(frame - 3500), 2 * DEFAULT.restart_window + DEFAULT.restart_search_after)
        pulse = channel.contribution()
        self.assertTrue((pulse <= 0).all())
        self.assertLessEqual(int((pulse < 0).sum()), DEFAULT.restart_pulse * len(events))
        self.assertAlmostEqual(float(pulse.min()), -DEFAULT.restart_gain)

    def test_players_moving_while_the_visible_ball_stays_still_is_not_a_restart(self):
        channel, events = self.run_restart(self.restart_scene("stationary_visible"), OA.replace(use_restart=True))
        self.assertEqual(events, [])
        self.assertTrue((channel.contribution() == 0).all())

    def test_a_hidden_ball_does_not_prevent_detecting_a_clear_movement_restart_at_reduced_strength(self):
        channel, events = self.run_restart(self.restart_scene("hidden"), OA.replace(use_restart=True))
        self.assertTrue(any(kind == "movement" for _, kind in events))
        self.assertAlmostEqual(float(channel.contribution().min()), -DEFAULT.restart_gain * DEFAULT.restart_movement_only_scale)

    def test_the_restart_pulse_carries_no_team(self):
        t = self.restart_scene("moves_at_restart")
        swapped = data.Tracking(t.player_xy, t.player_speed, t.ball_xy, 1 - t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps)
        a, _ = self.run_restart(t, OA.replace(use_restart=True))
        b, _ = self.run_restart(swapped, OA.replace(use_restart=True))
        np.testing.assert_allclose(a.contribution(), b.contribution())

    def test_a_restart_exits_the_dead_state_near_the_supported_moment(self):
        t = self.restart_scene("moves_at_restart")
        codes = decode(t, OA.replace(use_restart=True))
        self.assertGreater(float((codes[1700:3300] == decoder.D).mean()), 0.5)         # the stoppage is called
        last_dead = int(np.flatnonzero(codes == decoder.D).max())
        self.assertLessEqual(last_dead, 3500 + 6 * 25)                                # within ~6 s of the restart
        self.assertFalse((codes[3800:] == decoder.D).any())


class TestCounterfactualVisibilityInvariants(unittest.TestCase):
    """Tests A-F of the Phase 4.6 specification, on synthetic player evidence."""

    def live(self, mask=None):
        return tdb.scene(np.full(N, 3.5), n=N, mask_ball=mask)

    def test_A_hiding_the_ball_during_live_play_creates_no_stoppage(self):
        mask = np.zeros(N, bool)
        mask[1000:2600] = True
        mask[3500:5000] = True
        for config in (OA, PRESETS["oa_full"], PRESETS["corrected_dead"]):
            self.assertFalse((decode(self.live(mask), config) == decoder.D).any())
            self.assertFalse((decode(self.live(), config) == decoder.D).any())

    def stoppage(self, hidden):
        speed = tdb.profile(N, [(0, N, 4.0), (2500, 3700, 0.7)])
        t = tdb.scene(speed, n=N, spot=(2500, 3700))
        if hidden:
            mask = np.zeros(N, bool)
            mask[2400:3800] = True
            t = t.with_ball_masked(mask)
        return t

    def test_B_a_stoppage_is_still_called_when_the_ball_is_hidden_over_it(self):
        for hidden in (False, True):
            codes = decode(self.stoppage(hidden))
            self.assertGreater(float((codes[2600:3600] == decoder.D).mean()), 0.6, "hidden=%s" % hidden)
            self.assertFalse((codes[:2300] == decoder.D).any())

    def test_C_with_no_usable_evidence_the_model_invents_nothing(self):
        t = tdb.scene(3.0, n=1500)
        nothing = data.Tracking(np.full_like(t.player_xy, np.nan), t.player_speed, np.full_like(t.ball_xy, np.nan),
                                t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps)
        self.assertTrue((decode(nothing) == decoder.X).all())                      # a block with nothing observed: no answer
        # evidence, then nothing: the earlier state is carried, no transition is created in the gap
        part = data.Tracking(t.player_xy.copy(), t.player_speed, t.ball_xy.copy(), t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps)
        part.player_xy[900:] = np.nan
        part.ball_xy[900:] = np.nan
        codes = decode(part)
        self.assertEqual(len(set(codes[950:].tolist())), 1)
        self.assertEqual(int(codes[950]), int(codes[800]))

    def test_D_strong_independent_dead_evidence_enters_D_with_ownership_unavailable(self):
        speed = tdb.profile(N, [(0, N, 4.0), (2500, 3700, 0.6)])
        t = tdb.scene(speed, n=N, ball="absent")                                   # the ball is never observed
        codes = decode(t)
        self.assertGreater(float((codes[2650:3550] == decoder.D).mean()), 0.6)
        self.assertFalse((codes[:2300] == decoder.D).any())

    def test_E_a_short_collective_slowdown_in_live_play_is_not_a_stoppage(self):
        for ball in ("carried", "absent"):
            speed = tdb.profile(N, [(0, N, 4.0), (3000, 3060, 0.9)])               # 2.4 s
            self.assertFalse((decode(tdb.scene(speed, n=N, ball=ball)) == decoder.D).any(), ball)

    def test_F_play_resuming_ends_the_dead_state_near_the_restart(self):
        speed = tdb.profile(N, [(0, N, 4.0), (2000, 3500, 0.6)])
        codes = decode(tdb.scene(speed, n=N, spot=(2000, 3500)))
        last_dead = int(np.flatnonzero(codes == decoder.D).max())
        self.assertGreater(last_dead, 3300)
        self.assertLessEqual(last_dead, 3500 + 5 * 25)

    def test_visibility_does_not_change_the_dead_decision_for_identical_player_evidence_when_nothing_is_controlled(self):
        # the ball lies far from every player the whole time, so ownership evidence is ~0 either way
        speed = tdb.profile(N, [(0, N, 4.0), (2500, 3700, 0.6)])
        seen = decode(tdb.scene(speed, n=N, spot=(0, N)), OA)
        mask = np.ones(N, bool)
        hidden = decode(tdb.scene(speed, n=N, spot=(0, N), mask_ball=mask), OA)
        self.assertLess(float((seen != hidden).mean()), 0.02)


class TestPresetHygieneAndSafety(unittest.TestCase):
    NAMES = ("oa_only", "oa_entry", "oa_restart", "oa_full")

    def test_observation_aware_presets_keep_rejected_features_off_and_the_corrected_decoder_on(self):
        for name in self.NAMES:
            c = PRESETS[name]
            self.assertTrue(c.decoder_late_ties and c.dead_model == "oa")
            self.assertFalse(c.ownership_v2 or c.use_handover or c.use_dead_ball_module)
            self.assertFalse(c.dead_compensate_missing_ownership)               # no fixed missing-ball offset
            self.assertEqual((c.duration_model, c.emission_mode), ("none", "legacy"))

    def test_each_new_feature_can_be_switched_off_individually(self):
        self.assertFalse(PRESETS["oa_only"].use_restart)
        self.assertEqual(PRESETS["oa_only"].oa_activity_stat, "pooled")
        self.assertEqual(PRESETS["oa_only"].oa_outside_min_run, 1)
        self.assertTrue(PRESETS["oa_full"].use_restart and PRESETS["oa_full"].oa_outside_min_run > 1)

    def test_legacy_presets_are_unchanged(self):
        self.assertEqual(PRESETS["legacy_p42"], DEFAULT)
        self.assertEqual(PRESETS["legacy_p43"], DEFAULT.replace(use_dead_ball_module=True))
        self.assertEqual((DEFAULT.dead_model, DEFAULT.emission_mode, DEFAULT.use_restart), ("legacy", "legacy", False))

    def test_transition_costs_stay_safe_for_every_new_preset(self):
        for name in self.NAMES:
            self.assertTrue(duration.check_transition_safety(PRESETS[name])["safe"], name)

    def test_no_phantom_dead_state_from_ownership_noise_when_there_is_no_dead_evidence(self):
        rng = np.random.RandomState(5)
        for name in self.NAMES:
            for _ in range(10):
                own = rng.randn(1500) * rng.choice([0.8, 1.5, 3.0])
                codes = decoder.decode_blocks(own, np.zeros(1500), [(0, 1500)], PRESETS[name])
                self.assertFalse((codes == decoder.D).any(), name)

    def test_home_away_symmetry_and_half_isolation_of_the_whole_observation_aware_pipeline(self):
        speed = tdb.profile(N, [(0, N, 4.0), (2500, 3700, 0.6)])
        t = tdb.scene(speed, n=N)
        flipped = data.Tracking(t.player_xy, t.player_speed, t.ball_xy, 1 - t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps)
        a, b = decode(t, PRESETS["oa_full"]), decode(flipped, PRESETS["oa_full"])
        np.testing.assert_array_equal(np.where(a == decoder.H, decoder.A, np.where(a == decoder.A, decoder.H, a)), b)
        both = decode(t, OA, blocks=[(0, 3000), (3100, N)])
        alone = decode(t, OA, blocks=[(3100, N)])
        np.testing.assert_array_equal(both[3100:], alone[3100:])
        np.testing.assert_array_equal(decode(t, OA), decode(t, OA))


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")


@unittest.skipUnless(os.path.isfile(REAL_RAW), "tactical ATD not installed")
class TestRealData(unittest.TestCase):
    def test_observation_aware_presets_are_deterministic_and_leave_no_graded_frame_unanswered(self):
        tracking = data.load_tracking()
        graded = np.zeros(tracking.n_frames, bool)
        for s, e in predict.block_ranges():
            graded[s:e] = True
        a, _, ev = predict.predict_codes(tracking, PRESETS["oa_full"], "decoder")
        b, _, _ = predict.predict_codes(tracking, PRESETS["oa_full"], "decoder")
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a[graded] != decoder.X).all() and (a[~graded] == decoder.X).all())
        self.assertIn("dead_observation", ev.parts)


if __name__ == "__main__":
    unittest.main()
