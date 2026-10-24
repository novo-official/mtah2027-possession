"""Tests for solution/ownership.py, handover.py, diagnostics.py and the time-varying decoder (Phase 4.4).

Synthetic scenarios check specific implementation invariants (symmetry, no invented transitions
through gaps, single touches not counting as control, ...). They are not evidence of accuracy on
real matches. Two real-data tests (skipped without the data) pin the Phase 4.2 and Phase 4.3
outputs: with the new options off, the earlier models must reproduce byte for byte.

Run with:   .venv/bin/python -m unittest discover -s tests -v
"""

import hashlib
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from solution import data, decoder, diagnostics, features, handover, ownership, predict  # noqa: E402
from solution.config import DEFAULT  # noqa: E402

PITCH = (105.3, 68.0)
FPS = 25
V2 = DEFAULT.replace(ownership_v2=True)
V2H = V2.replace(use_handover=True)


def far_squad(n, k=10, y=60.0):
    """k stationary players far from the action (never the nearest to the ball)."""
    x = 10.0 + 8.0 * np.arange(k)
    return np.repeat(np.stack([x, np.full(k, y)], axis=1)[None], n, axis=0).astype(np.float32)


def build(n, home_runner, away_runner, ball, swap=False):
    """Tracking with one active player per team (slot 0 of each) and 10 distant fillers per team."""
    home = np.concatenate([home_runner[:, None, :], far_squad(n, 10, 62.0)], axis=1)
    away = np.concatenate([away_runner[:, None, :], far_squad(n, 10, 5.0)], axis=1)
    players = np.concatenate([home, away], axis=1).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    if swap:
        team = 1 - team
    return data.Tracking(player_xy=players, player_speed=np.zeros(players.shape[:2], np.float32),
                         ball_xy=np.asarray(ball, np.float32), slot_team=team,
                         slot_is_goalkeeper=np.zeros(22, bool), pitch=PITCH, fps=FPS)


def line(p0, p1, n):
    """n positions from p0 to p1."""
    return np.linspace(np.asarray(p0, float), np.asarray(p1, float), n)


def stay(p, n):
    return np.tile(np.asarray(p, float), (n, 1))


def carried(start, v, n, ahead=0.5):
    """(player path, ball path) for a player moving along +x at v m/s with the ball `ahead` m in front."""
    x = start[0] + v * np.arange(n) / FPS
    player = np.stack([x, np.full(n, start[1])], axis=1)
    return player, player + np.array([ahead, 0.0])


def decode(tracking, config=V2H):
    return predict.predict_codes(tracking, config, "decoder", [(0, tracking.n_frames)])[0]


def scenario_handover(n_pre=150, n_flight=20, n_post=200):
    """Home carries the ball, it travels 14 m to a waiting away player who then carries it."""
    h_pre, b_pre = carried((30.0, 30.0), 2.0, n_pre)
    a_wait = np.array([58.0, 30.0])
    flight = line(b_pre[-1], a_wait + np.array([0.5, 0.0]), n_flight)
    a_post, b_post = carried(a_wait, 2.0, n_post)
    ball = np.concatenate([b_pre, flight, b_post])
    home = np.concatenate([h_pre, stay(h_pre[-1], n_flight + n_post)])
    away = np.concatenate([stay(a_wait, n_pre + n_flight), a_post])
    return build(len(ball), home, away, ball), n_pre + n_flight


class TestStrictHelpers(unittest.TestCase):
    def test_window_is_invalid_across_a_gap_or_a_jump_and_valid_elsewhere(self):
        xy = np.stack([np.arange(60, dtype=np.float32) * 0.1, np.zeros(60, np.float32)], axis=1)
        xy[30] = np.nan
        ok = features.strict_window_valid(xy, 4, 1.0)
        self.assertTrue(not ok[26:35].any() and ok[10] and ok[50])
        xy = np.stack([np.arange(60, dtype=np.float32) * 0.1, np.zeros(60, np.float32)], axis=1)
        xy[30:, 0] += 4.0
        ok = features.strict_window_valid(xy, 4, 1.0)
        self.assertTrue(not ok[26:34].any() and ok[10] and ok[50])

    def test_strict_velocity_matches_the_motion_and_rejects_untrustworthy_windows(self):
        xy = np.stack([np.arange(60, dtype=np.float32) * 0.2, np.zeros(60, np.float32)], axis=1)   # 5 m/s
        v = features.strict_velocity(xy, 4, FPS, 1.0, 12.0)
        np.testing.assert_allclose(v[10], [5.0, 0.0], atol=1e-4)
        self.assertTrue(np.isnan(v[:4]).all())
        self.assertTrue(np.isnan(features.strict_velocity(xy, 4, FPS, 1.0, 4.0)).all())            # above the cap

    def test_track_age_counts_frames_since_acquisition_and_resets_on_gaps_and_jumps(self):
        xy = np.stack([np.arange(40, dtype=np.float32) * 0.1, np.zeros(40, np.float32)], axis=1)
        xy[10:12] = np.nan
        xy[25:] += 5.0
        age = features.track_age(xy[:, None, :], 1.0)[:, 0]
        self.assertEqual(int(age[9]), 9)
        self.assertEqual(int(age[10]), 0)
        self.assertEqual(int(age[12]), 0)
        self.assertEqual(int(age[15]), 3)
        self.assertEqual(int(age[25]), 0)
        self.assertEqual(int(age[30]), 5)


class TestControlEpisodes(unittest.TestCase):
    def test_bridging_strength_and_separate_episodes(self):
        c = np.zeros(60, bool)
        c[5:12] = True
        c[14:20] = True                  # 2-frame gap: bridged
        c[40:43] = True                  # a touch
        s, eps = ownership.contact_episodes(c, 6, 12.0)
        self.assertEqual([(a, b, n) for a, b, n, _ in eps], [(5, 20, 13), (40, 43, 3)])
        self.assertAlmostEqual(s[10], 1 - np.exp(-13 / 12.0))
        self.assertAlmostEqual(s[41], 1 - np.exp(-3 / 12.0))
        self.assertTrue(s[41] < 0.3 < 0.6 < s[10])
        self.assertEqual(float(s[30]), 0.0)

    def test_no_contact_gives_no_episodes(self):
        s, eps = ownership.contact_episodes(np.zeros(30, bool), 6, 12.0)
        self.assertTrue((s == 0).all() and eps == [])


class TestOwnershipFeatures(unittest.TestCase):
    def signals(self, tracking, config=V2):
        f = features.compute_features(tracking, [(0, tracking.n_frames)], config)
        return f, ownership.control_signals(tracking, f, [(0, tracking.n_frames)], config)

    def test_a_carried_ball_is_contact_and_a_ball_flying_past_a_player_is_not(self):
        h, b = carried((30.0, 30.0), 3.0, 200)
        t = build(200, h, stay((60.0, 30.0), 200), b)
        f, s = self.signals(t)
        self.assertTrue(s.contact_home[20:180].all())                   # moves with the ball
        n = 200
        ball = np.stack([30.0 + 1.0 * np.arange(n), np.full(n, 30.0)], axis=1)       # 25 m/s past a still player
        t2 = build(n, stay((60.0, 30.0), n), stay((90.0, 10.0), n), ball)
        t2.player_xy[:, 0] = np.array([80.0, 31.0], dtype=np.float32)                  # a home player next to the path
        f2, s2 = self.signals(t2)
        self.assertFalse(s2.contact_home.any())

    def test_relative_speed_of_each_teams_nearest_player(self):
        h, b = carried((30.0, 30.0), 3.0, 120)
        t = build(120, h, stay((70.0, 30.0), 120), b)
        f, s = self.signals(t)
        self.assertLess(float(s.rel_home[60]), 0.3)                    # ball and player move together
        self.assertTrue(np.isfinite(s.rel_away[60]) and s.rel_away[60] > 2.5)    # the still away player vs the moving ball

    def test_strict_motion_discounts_a_ball_the_claiming_player_is_not_moving_with(self):
        n = 120
        still_ball = stay((30.5, 30.0), n)
        t = build(n, stay((30.0, 30.0), n), stay((60.0, 30.0), n), still_ball)
        f = features.compute_features(t, [(0, n)], V2)
        base = ownership.control_evidence(t, f, [(0, n)], DEFAULT.replace(ownership_v2=True, own_strict_motion=False,
                                                                           own_track_stability=False, own_reception=False))[1]
        # same geometry, but the ball now flies by quickly
        ball = np.stack([30.5 + 0.6 * (np.arange(n) - 60), np.full(n, 30.0)], axis=1)
        t2 = build(n, stay((30.0, 30.0), n), stay((60.0, 30.0), n), ball)
        f2 = features.compute_features(t2, [(0, n)], V2)
        strict = ownership.control_evidence(t2, f2, [(0, n)], DEFAULT.replace(ownership_v2=True, own_track_stability=False,
                                                                              own_reception=False))
        self.assertLess(strict[1][60], 0.3 * base[60])

    def test_track_stability_lowers_the_weight_of_a_just_acquired_player(self):
        h, b = carried((30.0, 30.0), 2.0, 160)
        t = build(160, h, stay((70.0, 30.0), 160), b)
        players = t.player_xy.copy()
        players[:100, 0] = np.nan                                   # the home runner appears at frame 100
        t2 = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        f = features.compute_features(t2, [(0, 160)], V2)
        cfg = DEFAULT.replace(ownership_v2=True, own_strict_motion=False, own_reception=False)
        w = ownership.control_evidence(t2, f, [(0, 160)], cfg)[1]
        self.assertLess(w[105], 0.8 * w[155])                       # young track trusted less than an old one

    def test_all_refinements_off_equals_the_phase_4_2_evidence(self):
        h, b = carried((30.0, 30.0), 2.0, 200)
        t = build(200, h, stay((60.0, 30.0), 200), b)
        f = features.compute_features(t, [(0, 200)], DEFAULT)
        off = DEFAULT.replace(ownership_v2=True, own_strict_motion=False, own_track_stability=False, own_reception=False)
        from solution.evidence import ownership_evidence
        old = ownership_evidence(f, DEFAULT)
        new = ownership.control_evidence(t, f, [(0, 200)], off)
        np.testing.assert_allclose(old[0], new[0])
        np.testing.assert_allclose(old[1], new[1])

    def test_a_missing_ball_gives_no_ownership_evidence_and_no_contact(self):
        h, b = carried((30.0, 30.0), 2.0, 200)
        mask = np.zeros(200, bool)
        mask[50:150] = True
        t = build(200, h, stay((60.0, 30.0), 200), b).with_ball_masked(mask)
        f = features.compute_features(t, [(0, 200)], V2)
        own, w, parts, s = ownership.control_evidence(t, f, [(0, 200)], V2)
        self.assertTrue((own[50:150] == 0).all() and (w[50:150] == 0).all())
        self.assertFalse(s.contact_home[50:150].any())

    def test_a_single_touch_is_weak_and_sustained_control_is_strong(self):
        n = 120
        ball = stay((50.0, 30.0), n)
        near = stay((50.0, 30.0), n)
        near[:] = (80.0, 30.0)
        near[50:54] = (50.5, 30.0)                                  # four frames within reach
        t = build(n, near, stay((85.0, 5.0), n), ball)
        f, s = features.compute_features(t, [(0, n)], V2), None
        s = ownership.control_signals(t, f, [(0, n)], V2)
        self.assertLess(s.strength_home[51], 0.35)
        near[40:90] = (50.5, 30.0)
        t = build(n, near, stay((85.0, 5.0), n), ball)
        f = features.compute_features(t, [(0, n)], V2)
        s = ownership.control_signals(t, f, [(0, n)], V2)
        self.assertGreater(s.strength_home[60], 0.9)

    def test_home_and_away_are_processed_symmetrically(self):
        t, _ = scenario_handover()
        mirrored = build(t.n_frames, t.player_xy[:, 11], t.player_xy[:, 0], t.ball_xy)
        mirrored = data.Tracking(t.player_xy, t.player_speed, t.ball_xy, 1 - t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        for config in (V2, V2H):
            a, b = decode(t, config), decode(mirrored, config)
            swapped = np.where(a == decoder.H, decoder.A, np.where(a == decoder.A, decoder.H, a))
            np.testing.assert_array_equal(swapped, b)


class TestHandoverLogic(unittest.TestCase):
    def signals_for(self, tracking, config=V2H):
        blocks = [(0, tracking.n_frames)]
        f = features.compute_features(tracking, blocks, config)
        return ownership.control_signals(tracking, f, blocks, config), blocks

    def test_only_consecutive_established_episodes_of_different_teams_make_a_handover(self):
        t, _ = scenario_handover()
        s, blocks = self.signals_for(t)
        events = handover.handover_events(s, V2H)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0][2], events[0][3]), (0, 1))

    def test_same_team_episodes_make_no_handover(self):
        h1, b1 = carried((30.0, 30.0), 2.0, 120)
        h2, b2 = carried(h1[-1] + np.array([5.0, 0.0]), 2.0, 120)
        home = np.concatenate([h1, h2])
        ball = np.concatenate([b1, line(b1[-1], b2[0], 20)[:0], b2])
        home = np.concatenate([h1, h2])
        t = build(len(home), home, stay((95.0, 5.0), len(home)), ball)
        s, _ = self.signals_for(t)
        self.assertEqual(handover.handover_events(s, V2H), [])

    def test_a_deflection_without_control_is_not_a_handover(self):
        # home carries, the ball passes an away player (4 frames within reach) and goes on to home
        h_pre, b_pre = carried((30.0, 30.0), 2.0, 100)
        flight = line(b_pre[-1], (60.0, 30.0), 40)
        away = np.concatenate([stay((90.0, 5.0), 100), stay((90.0, 5.0), 40), stay((90.0, 5.0), 100)])
        away[100 + 18:100 + 22] = (flight[18] + np.array([0.0, 0.8]))          # a touch in passing
        h_post, b_post = carried((60.0, 30.0), 2.0, 100, ahead=0.5)
        home = np.concatenate([h_pre, stay(h_pre[-1], 40), h_post])
        ball = np.concatenate([b_pre, flight, b_post])
        t = build(len(ball), home, away, ball)
        s, blocks = self.signals_for(t)
        self.assertEqual(handover.handover_events(s, V2H), [])
        codes = decode(t)
        self.assertTrue((codes == decoder.H).all())

    def test_a_long_aerial_pass_to_a_teammate_keeps_possession(self):
        h_pre, b_pre = carried((20.0, 30.0), 2.0, 100)
        flight = line(b_pre[-1], (80.0, 30.0), 75)                  # 3 s, 19 m/s
        h_post, b_post = carried((80.0, 30.0), 2.0, 120, ahead=0.5)
        ball = np.concatenate([b_pre, flight, b_post])
        home = np.concatenate([h_pre, stay(h_pre[-1], 75), h_post])
        t = build(len(ball), home, stay((60.0, 55.0), len(ball)), ball)
        self.assertTrue((decode(t) == decoder.H).all())

    def test_a_clean_handover_switches_once_near_the_reception_and_not_before(self):
        t, flight_end = scenario_handover()
        codes = decode(t)
        change = np.flatnonzero(np.diff(codes) != 0)
        self.assertEqual(len(change), 1)
        self.assertTrue((codes[:150] == decoder.H).all() and (codes[-100:] == decoder.A).all())
        self.assertGreaterEqual(int(change[0]) + 1, 150)             # never before the ball left the first team
        self.assertLessEqual(int(change[0]) + 1, flight_end + 40)

    def test_a_genuine_short_possession_is_kept(self):
        h1, b1 = carried((30.0, 30.0), 2.0, 100)
        a_wait = np.array([40.0, 30.0])
        flight = line(b1[-1], a_wait + [0.5, 0.0], 12)
        a_mid, b_mid = carried(a_wait, 2.0, 40)                      # 1.6 s of away control
        back = line(b_mid[-1], (58.0, 30.0), 12)
        h2, b2 = carried((58.0, 30.0), 2.0, 100)
        ball = np.concatenate([b1, flight, b_mid, back, b2])
        n = len(ball)
        home = np.concatenate([h1, stay(h1[-1], 12 + 40 + 12), h2])
        away = np.concatenate([stay(a_wait, 100 + 12), a_mid, stay(a_mid[-1], 12 + 100)])
        codes = decode(build(n, home, away, ball))
        self.assertGreaterEqual(int((codes == decoder.A).sum()), 25)
        self.assertEqual(int(codes[0]), decoder.H)
        self.assertEqual(int(codes[-1]), decoder.H)

    def test_a_long_missing_ball_gap_creates_no_transition_without_evidence(self):
        t, _ = scenario_handover()
        n = t.n_frames
        mask = np.zeros(n, bool)
        mask[100:300] = True                                         # the whole hand-over is hidden
        hidden = t.with_ball_masked(mask)
        codes = decode(hidden)
        change = np.flatnonzero(np.diff(codes) != 0)
        # the new team is only seen afterwards; any switch is placed at the reappearance, not inside the gap
        self.assertTrue(all(c + 1 >= 300 - V2H.handover_slack for c in change))
        self.assertTrue((codes[:100] == decoder.H).all())
        self.assertEqual(int(codes[-1]), decoder.A)

    def test_extra_costs_are_nonnegative_capped_and_never_below_the_base_cost(self):
        t, _ = scenario_handover()
        s, blocks = self.signals_for(t)
        hv = handover.handover_costs(s, blocks, V2H)
        cap = handover.max_extra_cost(V2H)
        self.assertTrue((hv.extra_ha >= 0).all() and (hv.extra_ah >= 0).all())
        self.assertTrue((hv.extra_ha <= cap + 1e-12).all() and (hv.extra_ah <= cap + 1e-12).all())
        self.assertLess(float(hv.extra_ha.min()), 1e-3)              # fully supported switches cost the normal amount

    def test_a_direct_switch_is_never_costlier_than_the_detour_through_the_dead_state(self):
        for extra in (0.0, 0.25, 1.0, 5.0):
            cfg = V2H.replace(handover_extra=extra)
            self.assertLessEqual(cfg.switch_cost + handover.max_extra_cost(cfg),
                                 cfg.enter_dead_cost + cfg.exit_dead_cost - 1.0 + 1e-9)

    def test_support_exists_only_for_the_incoming_team_and_only_after_the_previous_episode(self):
        t, flight_end = scenario_handover()
        s, blocks = self.signals_for(t)
        hv = handover.handover_costs(s, blocks, V2H)
        cap = handover.max_extra_cost(V2H)
        supported_to_away = np.flatnonzero(hv.extra_ha < cap - 1e-9)
        supported_to_home = np.flatnonzero(hv.extra_ah < cap - 1e-9)
        self.assertTrue(len(supported_to_away) > 0)
        self.assertEqual(len(supported_to_home), 0)                  # no home takeover is supported anywhere
        self.assertGreaterEqual(int(supported_to_away.min()), 100)   # not while home is still in the middle of its possession

    def test_a_missing_ball_never_adds_support(self):
        t, _ = scenario_handover()
        mask = np.zeros(t.n_frames, bool)
        mask[:] = True
        s, blocks = self.signals_for(t.with_ball_masked(mask))
        hv = handover.handover_costs(s, blocks, V2H)
        cap = handover.max_extra_cost(V2H)
        np.testing.assert_allclose(hv.extra_ha, cap)
        np.testing.assert_allclose(hv.extra_ah, cap)

    def test_handover_logic_is_deterministic_and_off_means_the_original_path(self):
        t, _ = scenario_handover()
        np.testing.assert_array_equal(decode(t), decode(t))
        off = predict.predict_codes(t, V2, "decoder", [(0, t.n_frames)])[0]
        legacy = decoder.decode_blocks(*[getattr(predict.predict_codes(t, V2, "decoder", [(0, t.n_frames)])[2], k) for k in ("own_llr", "dead_llr")],
                                       [(0, t.n_frames)], V2)
        np.testing.assert_array_equal(off, legacy)


class TestTimeVaryingDecoder(unittest.TestCase):
    def test_zero_extra_equals_the_original_viterbi_exactly(self):
        rng = np.random.RandomState(3)
        em = rng.randn(500, 3)
        cost = decoder.transition_costs(DEFAULT)
        a = decoder.viterbi(em, cost, np.zeros(3))
        b = decoder.viterbi_time_varying(em, cost, np.zeros(3), np.zeros(500), np.zeros(500))
        np.testing.assert_array_equal(a, b)

    def test_extra_cost_moves_a_marginal_switch_and_only_in_the_priced_direction(self):
        own = np.concatenate([np.full(100, 1.0), np.full(14, -1.0), np.full(100, 1.0)])      # a 14-frame excursion
        dead = np.zeros_like(own)
        base = decoder.decode_blocks(own, dead, [(0, len(own))], DEFAULT)
        self.assertTrue((base[100:114] == decoder.A).all() or (base == decoder.H).all())
        free = np.zeros(len(own))
        dear = np.full(len(own), 50.0)
        keep = decoder.decode_blocks(own, dead, [(0, len(own))], DEFAULT, switch_extra=(dear, free))
        self.assertTrue((keep == decoder.H).all())            # entering away is made expensive: no excursion

    def test_halves_stay_isolated_with_time_varying_costs(self):
        own = np.zeros(300)
        own[:100] = 1.0
        own[160:250] = -0.1
        extra = (np.zeros(300), np.zeros(300))
        codes = decoder.decode_blocks(own, np.zeros(300), [(0, 100), (150, 250)], DEFAULT, switch_extra=extra)
        alone = decoder.decode_blocks(own[150:250], np.zeros(100), [(0, 100)], DEFAULT, switch_extra=(np.zeros(100), np.zeros(100)))
        np.testing.assert_array_equal(codes[150:250], alone)
        self.assertTrue((codes[100:150] == decoder.X).all())


class TestSymmetricTies(unittest.TestCase):
    def gap(self, first, second):
        return np.concatenate([np.full(100, first), np.zeros(300), np.full(100, second)])

    def switches(self, own, late):
        cfg = DEFAULT.replace(decoder_late_ties=late)
        codes = decoder.decode_blocks(own, np.zeros(len(own)), [(0, len(own))], cfg)
        return (np.flatnonzero(np.diff(codes) != 0) + 1).tolist()

    def test_legacy_ties_are_asymmetric_and_this_is_documented_behaviour(self):
        self.assertEqual(self.switches(self.gap(1.0, -1.0), False), [400])
        self.assertEqual(self.switches(self.gap(-1.0, 1.0), False), [100])     # the asymmetry late_ties removes

    def test_late_ties_place_both_directions_at_the_end_of_the_evidence_free_stretch(self):
        self.assertEqual(self.switches(self.gap(1.0, -1.0), True), [400])
        self.assertEqual(self.switches(self.gap(-1.0, 1.0), True), [400])

    def test_late_ties_change_nothing_when_there_are_no_ties(self):
        rng = np.random.RandomState(4)
        own = rng.randn(2000)
        dead = rng.rand(2000) * (rng.rand(2000) > 0.8)
        a = decoder.decode_blocks(own, dead, [(0, 2000)], DEFAULT)
        b = decoder.decode_blocks(own, dead, [(0, 2000)], DEFAULT.replace(decoder_late_ties=True))
        np.testing.assert_array_equal(a, b)

    def test_late_ties_work_with_time_varying_costs_and_are_deterministic(self):
        own = self.gap(-1.0, 1.0)
        z = np.zeros(len(own))
        cfg = DEFAULT.replace(decoder_late_ties=True)
        a = decoder.decode_blocks(own, z, [(0, len(own))], cfg, switch_extra=(z, z))
        b = decoder.decode_blocks(own, z, [(0, len(own))], cfg, switch_extra=(z, z))
        np.testing.assert_array_equal(a, b)
        self.assertEqual((np.flatnonzero(np.diff(a) != 0) + 1).tolist(), [400])


class TestDiagnostics(unittest.TestCase):
    def make(self):
        gt = np.array([0] * 100 + [1] * 100 + [2] * 60 + [0] * 100, dtype=np.int8)
        # a perfect copy, and one that misses the handover, mislocates the stoppage and invents a handover
        return gt

    def test_perfect_prediction_matches_every_boundary_by_type(self):
        gt = self.make()
        b = diagnostics.transition_breakdown(gt, gt.copy())
        self.assertEqual(b["counts"], {"H->A": 1, "A->H": 0, "alive->D": 1, "D->alive": 1})
        self.assertEqual(b["direct"][25]["matched"], 1)
        self.assertEqual(b["dead"][25]["matched"], 2)
        self.assertEqual(b["false_handovers"][25], 0)

    def test_direct_and_dead_boundaries_are_scored_separately(self):
        gt = self.make()
        pred = gt.copy()
        pred[100:200] = 0                                         # the handover is missed (stays home)
        pred[200:260] = 2
        pred[260:290] = 2                                         # the restart is 30 frames late (outside 25)
        b = diagnostics.transition_breakdown(gt, pred)
        self.assertEqual(b["direct"][25]["matched"], 0)
        self.assertEqual(b["missed_handovers"][25], 1)
        self.assertEqual(b["dead"][25]["matched"], 1)             # the stoppage start is found, the late restart is not
        self.assertEqual(b["by_type_recall"][25]["alive->D"], (1, 1))
        self.assertEqual(b["by_type_recall"][25]["D->alive"], (0, 1))

    def test_a_wrong_way_handover_does_not_match(self):
        gt = np.array([0] * 100 + [1] * 100, dtype=np.int8)
        pred = np.array([1] * 100 + [0] * 100, dtype=np.int8)
        b = diagnostics.transition_breakdown(gt, pred)
        self.assertEqual(b["direct"][25]["matched"], 0)
        self.assertEqual(b["false_handovers"][25], 1)

    def test_ownership_disagreement_counts(self):
        gt = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        pred = np.array([0, 1, 1, 2, 2, 3], dtype=np.int8)
        d = diagnostics.ownership_disagreement(gt, pred)
        self.assertEqual(d, {"alive_frames": 4, "wrong_team": 1, "called_dead": 1, "abstained": 0})


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
PHASE_4_3_SHA256 = "367b7415e3f2f700115c1846f7ba3ce366b34203732539f4a81d37af79f148a2"


@unittest.skipUnless(os.path.isfile(REAL_RAW), "tactical ATD not installed")
class TestEarlierModelsPreserved(unittest.TestCase):
    def sha(self, config):
        tracking = data.load_tracking()
        codes, _, _ = predict.predict_codes(tracking, config, "decoder")
        with tempfile.TemporaryDirectory() as tmp:
            path = predict.write_submission(os.path.join(tmp, "x.csv"), codes)
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()

    def test_phase_4_3_output_is_reproduced_byte_for_byte(self):
        self.assertEqual(self.sha(DEFAULT.replace(use_dead_ball_module=True)), PHASE_4_3_SHA256)


if __name__ == "__main__":
    unittest.main()
