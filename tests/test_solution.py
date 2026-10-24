"""Tests for the possession pipeline in solution/.

Run with:   .venv/bin/python -m unittest discover -s tests -v

Almost everything uses small synthetic fixtures, so the suite runs without the restricted data.
The few tests that read the real tactical ATD are skipped when it is not installed, and they
assert only booleans and counts: nothing from the data is printed.
"""

import csv
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from solution import data, decoder, evidence, features, predict  # noqa: E402
from solution.config import DEFAULT, parse_overrides  # noqa: E402

PITCH = (105.3, 68.0)
FPS = 25


def make_tracking(ball, home, away, n_home_slots=None):
    """Tracking from metre arrays. ball (N,2); home/away (N, k, 2). NaN = missing."""
    ball = np.asarray(ball, dtype=np.float32)
    home = np.asarray(home, dtype=np.float32)
    away = np.asarray(away, dtype=np.float32)
    players = np.concatenate([home, away], axis=1)
    team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    n = ball.shape[0]
    return data.Tracking(player_xy=players, player_speed=np.zeros(players.shape[:2], np.float32),
                         ball_xy=ball, slot_team=team,
                         slot_is_goalkeeper=np.zeros(len(team), bool), pitch=PITCH, fps=FPS)


def crowd(n, k=10, base_x=20.0):
    """k players far away from the action, same for every frame."""
    x = base_x + 3.0 * np.arange(k)
    xy = np.stack([x, np.full(k, 5.0)], axis=1)
    return np.repeat(xy[None], n, axis=0)


def scene(n, ball_xy, home_near=None, away_near=None):
    """n identical frames: ball at ball_xy, optional extra near players, 10 distant fillers per team."""
    ball = np.tile(np.asarray(ball_xy, dtype=np.float32), (n, 1))
    home = crowd(n, 10, 5.0)
    away = crowd(n, 10, 5.0) + np.array([0.0, 50.0], dtype=np.float32)
    if home_near is not None:
        home = np.concatenate([home, np.tile(np.asarray(home_near, np.float32), (n, 1, 1))], axis=1)
    if away_near is not None:
        away = np.concatenate([away, np.tile(np.asarray(away_near, np.float32), (n, 1, 1))], axis=1)
    return make_tracking(ball, home, away)


class TestParsingAndAlignment(unittest.TestCase):
    def write_files(self, tmp, ids, with_extra_token=False):
        meta = os.path.join(tmp, "meta.xml")
        with open(meta, "w") as fh:
            fh.write(
                "<main><Metadata><Sessions><Session><MatchParameters><Teams>"
                "<Team id='T1'/><Team id='T2'/></Teams></MatchParameters>"
                "<Players>"
                "<Player id='p1' teamId='T1'><ProviderPlayerParameters><ProviderParameter>"
                "<Name>position_type</Name><Value>Goalkeeper</Value></ProviderParameter>"
                "</ProviderPlayerParameters></Player>"
                "<Player id='p2' teamId='T2'><ProviderPlayerParameters><ProviderParameter>"
                "<Name>position_type</Name><Value>Unknown</Value></ProviderParameter>"
                "</ProviderPlayerParameters></Player>"
                "</Players></Session></Sessions></Metadata></main>")
        raw = os.path.join(tmp, "raw.txt")
        with open(raw, "w") as fh:
            for i in ids:
                extra = ",1" if with_extra_token and i == ids[0] else ""
                fh.write("{}:0.5,0.5,1.0;NaN,NaN,NaN:0.25,0.75,NaN{}\n".format(i, extra))
        return meta, raw

    def test_team_mapping_first_team_is_home_and_goalkeeper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta, _ = self.write_files(tmp, [1])
            team, keeper = data.parse_metadata(meta)
        self.assertEqual(team.tolist(), [data.HOME, data.AWAY])
        self.assertEqual(keeper.tolist(), [True, False])

    def test_frame_alignment_row_i_is_video_frame_i(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, raw = self.write_files(tmp, [1, 2, 3])
            ids, players, ball = data.parse_raw(raw, 2, expected_frames=3)
        self.assertEqual(ids.tolist(), [1, 2, 3])        # ATD id = video frame + 1
        self.assertEqual(players.shape, (3, 2, 3))
        self.assertTrue(np.isnan(players[0, 1]).all())
        self.assertAlmostEqual(float(ball[2, 0]), 0.25)

    def test_non_contiguous_ids_are_rejected_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, raw = self.write_files(tmp, [1, 2, 4])
            with self.assertRaises(ValueError):
                data.parse_raw(raw, 2)

    def test_wrong_record_count_and_malformed_lines_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, raw = self.write_files(tmp, [1, 2])
            with self.assertRaises(ValueError):
                data.parse_raw(raw, 2, expected_frames=3)
            _, raw = self.write_files(tmp, [1, 2], with_extra_token=True)
            with self.assertRaises(ValueError):
                data.parse_raw(raw, 2)

    def test_half_blocks_are_the_published_graded_halves(self):
        blocks = data.half_blocks()
        self.assertEqual([(a, b) for _, a, b in blocks], [(11185, 90652), (110808, 190198)])


class TestCoordinates(unittest.TestCase):
    def test_unit_conversion_scales_each_axis_by_its_own_pitch_side(self):
        out = data.sanitize(np.array([[0.5, 0.5], [1.0, 0.0]]), PITCH, 15.0)
        np.testing.assert_allclose(out, [[52.65, 34.0], [105.3, 0.0]], rtol=1e-6)

    def test_missing_and_implausible_positions_become_nan_in_both_coordinates(self):
        raw = np.array([[np.nan, 0.5], [0.5, np.inf], [5.0, 0.5], [-0.5, 0.5], [1.05, 0.5]])
        out = data.sanitize(raw, PITCH, 15.0)       # explicit margin; the default is wider
        self.assertTrue(np.isnan(out[:4]).all())          # nan, inf, 5 pitches away, 52 m behind the line
        self.assertTrue(np.isfinite(out[4]).all())        # 5 m beyond the touchline is kept (ball can be out)

    def test_ball_just_outside_the_lines_is_kept(self):
        out = data.sanitize(np.array([[-0.01, 0.5]]), PITCH, 15.0)
        self.assertTrue(np.isfinite(out).all())


class TestFeatures(unittest.TestCase):
    def test_nearest_distance_per_team_and_hard_rule(self):
        t = scene(5, (50.0, 34.0), home_near=[[(50.0, 31.0)]], away_near=[[(50.0, 36.0)]])
        f = features.compute_features(t, None, DEFAULT)
        np.testing.assert_allclose(f.d_home, 3.0, atol=1e-5)
        np.testing.assert_allclose(f.d_away, 2.0, atol=1e-5)
        self.assertTrue((f.nearest_team == data.AWAY).all())

    def test_hard_rule_tie_goes_to_lowest_slot_like_the_baseline(self):
        t = scene(3, (50.0, 34.0), home_near=[[(50.0, 32.0)]], away_near=[[(50.0, 36.0)]])
        f = features.compute_features(t, None, DEFAULT)
        self.assertTrue((f.nearest_team == data.HOME).all())   # home slots precede away slots

    def test_missing_ball_makes_everything_unavailable(self):
        t = scene(6, (50.0, 34.0), home_near=[[(50.0, 31.0)]], away_near=[[(50.0, 36.0)]])
        t = t.with_ball_masked(np.array([False, False, True, True, False, False]))
        f = features.compute_features(t, None, DEFAULT)
        self.assertEqual(f.ball_valid.tolist(), [True, True, False, False, True, True])
        self.assertTrue(np.isnan(f.d_home[2:4]).all() and np.isnan(f.d_away[2:4]).all())
        self.assertTrue((f.nearest_team[2:4] == -1).all())
        self.assertTrue(np.isnan(f.rel_speed[2:4]).all())
        self.assertEqual(f.gap_length.tolist(), [0, 0, 2, 2, 0, 0])

    def test_missing_players_are_never_nearest(self):
        t = scene(3, (50.0, 34.0), home_near=[[(50.0, 33.0)]], away_near=[[(50.0, 36.0)]])
        players = t.player_xy.copy()
        players[:, 10] = np.nan                       # the home player 1 m from the ball disappears
        t = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper,
                          PITCH, FPS)
        f = features.compute_features(t, None, DEFAULT)
        self.assertTrue((f.nearest_team == data.AWAY).all())
        self.assertTrue((f.d_home > 3.0).all())

    def test_ball_velocity_from_positions_and_jump_rejection(self):
        n = 40
        x = 20.0 + np.arange(n, dtype=np.float32)            # 1 m per frame = 25 m/s
        ball = np.stack([x, np.full(n, 34.0, np.float32)], axis=1)
        ball[20, 0] += 100.0                                 # one tracker jump
        t = make_tracking(ball, crowd(n), crowd(n) + np.array([0.0, 50.0], np.float32))
        f = features.compute_features(t, None, DEFAULT)
        self.assertAlmostEqual(float(f.ball_speed[10]), 25.0, places=3)
        self.assertTrue(np.isnan(f.ball_speed[:4]).all() and np.isnan(f.ball_speed[-4:]).all())
        # a central difference at t reads frames t-4 and t+4 only, so a jump at frame 20 spoils
        # exactly t = 16 and t = 24; those are discarded, every other interior frame stays clean
        bad = {16, 24}
        for t_ in range(4, n - 4):
            if t_ in bad:
                self.assertTrue(np.isnan(f.ball_speed[t_]))
            else:
                self.assertAlmostEqual(float(f.ball_speed[t_]), 25.0, places=3)

    def test_ball_outside_distance(self):
        ball = np.array([[50.0, 34.0], [-2.0, 34.0], [110.3, 72.0]], np.float32)
        t = make_tracking(ball, crowd(3), crowd(3) + np.array([0.0, 50.0], np.float32))
        f = features.compute_features(t, None, DEFAULT)
        np.testing.assert_allclose(f.ball_outside_m, [0.0, 2.0, np.hypot(5.0, 4.0)], atol=1e-5)

    def test_blocks_isolate_and_leave_the_rest_unavailable(self):
        t = scene(30, (50.0, 34.0), home_near=[[(50.0, 31.0)]], away_near=[[(50.0, 36.0)]])
        f = features.compute_features(t, [(0, 10), (20, 30)], DEFAULT)
        self.assertEqual(f.in_block.tolist(), [True] * 10 + [False] * 10 + [True] * 10)
        self.assertFalse(f.ball_valid[10:20].any())
        with self.assertRaises(ValueError):
            features.compute_features(t, [(0, 31)], DEFAULT)


def evidence_for(n=60, ball=(50.0, 34.0), home_near=None, away_near=None, mask=None, config=DEFAULT):
    t = scene(n, ball, home_near, away_near)
    if mask is not None:
        t = t.with_ball_masked(mask)
    f = features.compute_features(t, None, config)
    return f, evidence.build_evidence(f, config)


class TestEvidence(unittest.TestCase):
    def test_sign_follows_the_nearer_team(self):
        _, ev = evidence_for(home_near=[[(50.0, 33.0)]], away_near=[[(50.0, 38.0)]])
        self.assertTrue((ev.own_llr > 0).all())
        _, ev = evidence_for(home_near=[[(50.0, 38.0)]], away_near=[[(50.0, 33.0)]])
        self.assertTrue((ev.own_llr < 0).all())

    def test_equidistant_teams_give_no_evidence(self):
        _, ev = evidence_for(home_near=[[(50.0, 32.0)]], away_near=[[(50.0, 36.0)]])
        np.testing.assert_allclose(ev.own_llr, 0.0, atol=1e-6)

    def test_small_margin_is_weaker_than_large_margin_and_scale_is_monotone(self):
        _, tight = evidence_for(home_near=[[(50.0, 33.5)]], away_near=[[(50.0, 35.0)]])
        _, clear = evidence_for(home_near=[[(50.0, 33.5)]], away_near=[[(50.0, 40.0)]])
        self.assertLess(abs(tight.own_llr[30]), abs(clear.own_llr[30]))
        self.assertLessEqual(abs(clear.own_llr[30]), DEFAULT.evidence_gain + 1e-9)

    def test_loose_ball_far_from_everyone_carries_little_weight(self):
        _, near = evidence_for(home_near=[[(50.0, 33.5)]], away_near=[[(50.0, 40.0)]])
        _, far = evidence_for(home_near=[[(50.0, 24.0)]], away_near=[[(50.0, 44.0)]])
        self.assertLess(far.own_weight[30], 0.2 * near.own_weight[30])

    def test_ball_in_flight_is_down_weighted(self):
        n = 60
        home = np.concatenate([crowd(n, 10, 5.0), np.tile(np.array([[[50.0, 35.0]]], np.float32), (n, 1, 1))], axis=1)
        away = np.concatenate([crowd(n, 10, 5.0) + np.array([0.0, 50.0], np.float32),
                               np.tile(np.array([[[50.0, 28.0]]], np.float32), (n, 1, 1))], axis=1)
        still = np.tile(np.array([50.0, 34.0], np.float32), (n, 1))
        flying = np.stack([50.0 + (np.arange(n) - 30) * 1.0, np.full(n, 34.0)], axis=1).astype(np.float32)  # 25 m/s
        ev_still = evidence.build_evidence(
            features.compute_features(make_tracking(still, home, away), None, DEFAULT), DEFAULT)
        ev_fly = evidence.build_evidence(
            features.compute_features(make_tracking(flying, home, away), None, DEFAULT), DEFAULT)
        # at frame 30 both have the same geometry (home player 1 m away); only the motion differs
        self.assertGreater(ev_still.parts["motion"][30], 0.99)
        self.assertLess(ev_fly.parts["motion"][30], 0.1)
        self.assertLess(abs(ev_fly.own_llr[30]), 0.2 * abs(ev_still.own_llr[30]))

    def test_missing_ball_is_no_evidence_and_not_dead_evidence(self):
        mask = np.zeros(60, bool)
        mask[20:50] = True
        f, ev = evidence_for(home_near=[[(50.0, 33.0)]], away_near=[[(50.0, 38.0)]], mask=mask)
        self.assertTrue((ev.own_llr[20:50] == 0).all())
        self.assertTrue((ev.own_weight[20:50] == 0).all())
        self.assertTrue((ev.dead_llr == 0).all())           # absence never supports a dead ball
        self.assertFalse(ev.available[20:50].any())
        self.assertTrue(ev.available[:20].all())

    def test_too_few_tracked_players_lowers_the_weight(self):
        f, ev = evidence_for(home_near=[[(50.0, 33.0)]], away_near=[[(50.0, 38.0)]])
        t = scene(60, (50.0, 34.0), [[(50.0, 33.0)]], [[(50.0, 38.0)]])
        players = t.player_xy.copy()
        players[:, 1:10] = np.nan                                   # leave one home filler + the near player
        t2 = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        ev2 = evidence.build_evidence(features.compute_features(t2, None, DEFAULT), DEFAULT)
        self.assertLess(ev2.own_weight[30], ev.own_weight[30])

    def test_flickering_claim_is_down_weighted_by_consistency(self):
        n = 60
        ball = np.tile(np.array([50.0, 34.0], np.float32), (n, 1))
        even = (np.arange(n) % 2 == 0)
        home_y = np.where(even, 33.0, 38.0).astype(np.float32)     # nearer team alternates every frame
        away_y = np.where(even, 38.0, 33.0).astype(np.float32)
        home_near = np.stack([np.full(n, 50.0, np.float32), home_y], axis=1)[:, None, :]
        away_near = np.stack([np.full(n, 50.0, np.float32), away_y], axis=1)[:, None, :]
        t = make_tracking(ball, np.concatenate([crowd(n), home_near], axis=1),
                          np.concatenate([crowd(n) + np.array([0.0, 50.0], np.float32), away_near], axis=1))
        ev = evidence.build_evidence(features.compute_features(t, None, DEFAULT), DEFAULT)
        steady = evidence_for(n=n, home_near=[[(50.0, 33.0)]], away_near=[[(50.0, 38.0)]])[1]
        self.assertAlmostEqual(float(steady.parts["consistency"][30]), 1.0, places=9)
        # 25 frames in the window, alternating: |sum of signs| = 1
        self.assertAlmostEqual(float(ev.parts["consistency"][30]),
                               DEFAULT.consistency_floor + (1 - DEFAULT.consistency_floor) / 25.0, places=9)

    def test_dead_cue_needs_a_tracked_ball_outside_the_lines(self):
        ball = np.array([[50.0, 34.0], [50.0, 69.5], [50.0, 68.5], [50.0, np.nan]], np.float32)
        t = make_tracking(ball, crowd(4), crowd(4) + np.array([0.0, 50.0], np.float32))
        f = features.compute_features(t, None, DEFAULT)
        dead = evidence.dead_evidence(f, DEFAULT)
        self.assertEqual(dead.tolist(), [0.0, DEFAULT.outside_dead_gain, 0.0, 0.0])
        off = evidence.dead_evidence(f, DEFAULT.replace(use_outside_cue=False))
        self.assertTrue((off == 0).all())

    def test_external_dead_evidence_hook_adds_and_validates_shape(self):
        t = scene(5, (50.0, 34.0))
        f = features.compute_features(t, None, DEFAULT)
        dead = evidence.dead_evidence(f, DEFAULT, external=np.array([0, 1.0, 0, np.nan, 2.0]))
        self.assertEqual(dead.tolist(), [0.0, 1.0, 0.0, 0.0, 2.0])
        with self.assertRaises(ValueError):
            evidence.dead_evidence(f, DEFAULT, external=np.zeros(4))

    def test_hard_nearest_evidence_matches_the_baseline_rule(self):
        f, _ = evidence_for(home_near=[[(50.0, 33.0)]], away_near=[[(50.0, 38.0)]])
        hard = evidence.hard_nearest_evidence(f, DEFAULT)
        self.assertTrue((hard.own_llr == DEFAULT.evidence_gain).all())


class TestDecoder(unittest.TestCase):
    cfg = DEFAULT

    def decode(self, own, dead=None, blocks=None):
        own = np.asarray(own, dtype=float)
        dead = np.zeros_like(own) if dead is None else np.asarray(dead, dtype=float)
        blocks = [(0, len(own))] if blocks is None else blocks
        return decoder.decode_blocks(own, dead, blocks, self.cfg)

    def test_no_evidence_gives_one_constant_state_and_no_transition(self):
        codes = self.decode(np.zeros(300))
        self.assertEqual(len(set(codes.tolist())), 1)

    def test_sustained_evidence_switches_at_the_boundary(self):
        own = np.concatenate([np.full(100, 1.0), np.full(100, -1.0)])
        codes = self.decode(own)
        self.assertTrue((codes[:100] == decoder.H).all() and (codes[100:] == decoder.A).all())

    def test_flickering_evidence_does_not_oscillate(self):
        own = np.where(np.arange(400) % 2 == 0, 1.0, -1.0) * 1.0
        own[:3] = 1.0
        codes = self.decode(own)
        self.assertEqual(int(np.sum(np.diff(codes) != 0)), 0)

    def test_short_burst_is_suppressed_but_a_longer_possession_survives(self):
        # an inserted possession pays two switches (2 x switch_cost) against its own evidence
        base = np.full(300, 1.0)
        short = base.copy(); short[100:100 + 15] = -1.0          # 15 < 2 x 12
        longer = base.copy(); longer[100:100 + 40] = -1.0        # 40 > 24
        self.assertTrue((self.decode(short) == decoder.H).all())
        c = self.decode(longer)
        self.assertTrue((c[100:140] == decoder.A).all() and (c[:100] == decoder.H).all())

    def test_costs_are_penalties_not_prohibitions(self):
        cost = decoder.transition_costs(self.cfg)
        self.assertTrue(np.isfinite(cost).all())
        self.assertTrue((np.diag(cost) == 0).all())
        off = cost[~np.eye(3, dtype=bool)]
        self.assertTrue((off > 0).all())
        # a huge amount of contrary evidence always wins
        own = np.concatenate([np.full(50, 1.0), np.full(2000, -5.0)])
        self.assertTrue((self.decode(own)[-1000:] == decoder.A).all())

    def test_handover_and_dead_ball_have_separate_costs(self):
        cost = decoder.transition_costs(self.cfg.replace(switch_cost=7.0, enter_dead_cost=3.0, exit_dead_cost=2.0))
        self.assertEqual((cost[0, 1], cost[1, 0]), (7.0, 7.0))
        self.assertEqual((cost[0, 2], cost[1, 2]), (3.0, 3.0))
        self.assertEqual((cost[2, 0], cost[2, 1]), (2.0, 2.0))

    def test_dead_state_needs_sustained_dead_evidence_and_returns_afterwards(self):
        own = np.full(200, 1.0)
        dead = np.zeros(200)
        dead[80:100] = 2.0                                        # 20 frames: 40 nats vs enter+exit 16
        codes = self.decode(own, dead)
        self.assertTrue((codes[85:95] == decoder.D).all())
        self.assertEqual(int(codes[0]), decoder.H)
        self.assertEqual(int(codes[-1]), decoder.H)
        blip = np.zeros(200); blip[80:84] = 2.0                   # 4 frames: 8 nats < 16
        self.assertFalse((self.decode(own, blip) == decoder.D).any())

    def test_missing_evidence_carries_the_previous_state_and_never_creates_dead_or_transitions(self):
        own = np.zeros(1000); own[:100] = 1.0; own[900:] = 0.0    # evidence for H, then a 900-frame gap
        codes = self.decode(own)
        self.assertTrue((codes == decoder.H).all())
        own = np.zeros(1000); own[:100] = -1.0
        codes = self.decode(own)
        self.assertTrue((codes == decoder.A).all())               # A held through the gap, not D

    def test_genuine_evidence_after_a_gap_still_switches(self):
        own = np.zeros(1000); own[:100] = 1.0; own[800:] = -1.0
        codes = self.decode(own)
        self.assertEqual(int(codes[50]), decoder.H)
        self.assertTrue((codes[800:] == decoder.A).all())

    def test_halves_are_decoded_independently_and_the_gap_is_x(self):
        own = np.zeros(300)
        own[:100] = 1.0                       # first block strongly home
        own[160:250] = -0.1                   # second block weak away: total -9 nats
        codes = self.decode(own, blocks=[(0, 100), (150, 250)])
        self.assertTrue((codes[:100] == decoder.H).all())
        self.assertTrue((codes[100:150] == decoder.X).all() and (codes[250:] == decoder.X).all())
        # carried over from the first half, home would be cheaper (no switch cost) than away
        self.assertTrue((codes[160:250] == decoder.A).all())
        alone = decoder.decode_blocks(own[150:250], np.zeros(100), [(0, 100)], self.cfg)
        np.testing.assert_array_equal(codes[150:250], alone)

    def test_decoding_is_deterministic(self):
        rng = np.random.RandomState(0)
        own = rng.randn(5000); dead = rng.rand(5000) * (rng.rand(5000) > 0.9)
        a = self.decode(own, dead); b = self.decode(own, dead)
        np.testing.assert_array_equal(a, b)

    def test_ties_resolve_to_the_lowest_state(self):
        self.assertEqual(int(self.decode(np.zeros(5))[0]), decoder.H)

    def test_viterbi_matches_brute_force_on_a_tiny_problem(self):
        rng = np.random.RandomState(1)
        em = rng.randn(6, 3)
        cost = decoder.transition_costs(self.cfg.replace(switch_cost=1.0, enter_dead_cost=1.5, exit_dead_cost=0.5))
        path = decoder.viterbi(em, cost, np.zeros(3))
        best, best_path = -1e18, None
        import itertools
        for p in itertools.product(range(3), repeat=6):
            s = em[0, p[0]] + sum(em[t, p[t]] - cost[p[t - 1], p[t]] for t in range(1, 6))
            if s > best + 1e-12:
                best, best_path = s, p
        self.assertEqual(tuple(path.tolist()), best_path)


class TestOutputContract(unittest.TestCase):
    def test_csv_structure_and_labels(self):
        codes = np.array([3, 3, 0, 1, 2, 3], dtype=np.int8)
        with tempfile.TemporaryDirectory() as tmp:
            path = predict.write_submission(os.path.join(tmp, "out", "x.csv"), codes)
            with open(path) as fh:
                rows = list(csv.reader(fh))
            frame = pm.read_label_csv(path)
        self.assertEqual(rows[0], ["frame", "label"])
        self.assertEqual([r[1] for r in rows[1:]], list("XXHADX"))
        self.assertEqual([int(r[0]) for r in rows[1:]], list(range(6)))
        self.assertIsNone(pm.validate_submission(frame, 6))

    def test_evidence_variants_use_only_valid_labels_and_hold_fills_x(self):
        own = np.array([0, 0, 1.0, 0, 0, -1.0, 0, 0])
        x = predict.evidence_labels(own, [(0, 8)], 0.25, hold=False)
        h = predict.evidence_labels(own, [(0, 8)], 0.25, hold=True)
        self.assertEqual(x.tolist(), [3, 3, 0, 3, 3, 1, 3, 3])
        self.assertEqual(h.tolist(), [0, 0, 0, 0, 0, 1, 1, 1])        # head back-filled, tail carried

    def test_config_overrides_are_typed_and_validated(self):
        cfg = parse_overrides(DEFAULT, ["switch_cost=20", "use_outside_cue=false", "initial_logprior=0,1,2"])
        self.assertEqual((cfg.switch_cost, cfg.use_outside_cue, cfg.initial_logprior), (20.0, False, (0.0, 1.0, 2.0)))
        with self.assertRaises(KeyError):
            parse_overrides(DEFAULT, ["nonsense=1"])


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")


@unittest.skipUnless(os.path.isfile(REAL_RAW), "tactical ATD not installed")
class TestRealData(unittest.TestCase):
    """Assertions only; no data values are printed."""

    @classmethod
    def setUpClass(cls):
        cls.tracking = data.load_tracking()
        cls.cfg = DEFAULT

    def test_shape_and_alignment(self):
        self.assertEqual(self.tracking.n_frames, games.source_frames("tactical"))
        self.assertEqual(self.tracking.n_slots, 46)

    def test_hard_rule_reproduces_the_committed_raw_baseline_exactly(self):
        if not os.path.isfile(os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")):
            self.skipTest("the organisers' committed baselines are not part of this checkout")
        f = features.compute_features(self.tracking, None, self.cfg)
        mine = np.where(f.nearest_team == 0, 0, np.where(f.nearest_team == 1, 1, 2))
        committed = pm.label_codes(pm.read_label_csv(
            os.path.join(ROOT, "evaluation", "results", "baseline_raw.csv")))
        self.assertTrue(np.array_equal(mine, committed))

    def test_ball_availability_matches_the_published_figure(self):
        valid = int(np.isfinite(self.tracking.ball_xy).all(axis=1).sum())
        self.assertEqual(valid, 113488)       # DATA.md: 113,488 frames with a ball

    def test_full_pipeline_output_is_valid_and_deterministic(self):
        codes1, _, _ = predict.predict_codes(self.tracking, self.cfg)
        codes2, _, _ = predict.predict_codes(self.tracking, self.cfg)
        self.assertTrue(np.array_equal(codes1, codes2))
        self.assertEqual(len(codes1), 199319)
        graded = np.zeros(len(codes1), bool)
        for a, b in predict.block_ranges():
            graded[a:b] = True
        self.assertTrue((codes1[~graded] == decoder.X).all())
        self.assertTrue((codes1[graded] != decoder.X).all())


if __name__ == "__main__":
    unittest.main()
