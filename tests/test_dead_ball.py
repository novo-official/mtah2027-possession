"""Tests for solution/dead_ball.py and its integration (Phase 4.3).

These use synthetic motion and check *intended algorithm behaviour*. They are not evidence of
accuracy on real matches. One real-data test (skipped when the data is absent) pins the Phase 4.2
output: with the module disabled the pipeline must reproduce it exactly.

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

from solution import data, dead_ball, decoder, evidence, features, predict  # noqa: E402
from solution.config import DEFAULT  # noqa: E402

PITCH = (105.3, 68.0)
FPS = 25
ON = DEFAULT.replace(use_dead_ball_module=True, dead_use_near_ball=False)
ABS = ON.replace(dead_speed_reference="absolute")


def circling_players(n, speed, radius=8.0, n_per_team=11):
    """22 players on circles at the given per-frame speed profile (m/s); returns (n, 22, 2) metres.

    Motion on a circle keeps every player on the pitch for any duration while having an exactly
    known speed (up to the small chord error of the central difference).
    """
    speed = np.broadcast_to(np.asarray(speed, dtype=float), (n,))
    theta = np.cumsum(speed / radius / FPS)
    k = 2 * n_per_team
    rng = np.random.RandomState(7)
    centres = np.stack([rng.uniform(20, 85, k), rng.uniform(15, 53, k)], axis=1)
    phase = rng.uniform(0, 2 * np.pi, k)
    ang = theta[:, None] + phase[None, :]
    xy = centres[None] + radius * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    return xy.astype(np.float32)


def scene(speed, n=None, ball="carried", mask_ball=None, spot=None):
    """Tracking: 22 circling players; the ball is 1 m from home player 0 ('carried') or absent.

    `spot=(a, b)` leaves the ball lying at a restart spot far from every player during frames
    [a, b) instead of being carried: a realistic stoppage. (A ball that stays tightly controlled by
    a team gives the alive states up to 1.5 nats per frame, more than any dead score can overcome
    by design, so a stoppage with the ball still in a player's possession is not detectable.)
    """
    speed = np.asarray(speed, dtype=float)
    n = n or len(speed)
    players = circling_players(n, speed if speed.ndim else float(speed))
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    ball_xy = players[:, 0, :] + np.array([1.0, 0.0], dtype=np.float32)
    if spot is not None:
        ball_xy = ball_xy.copy()
        ball_xy[spot[0]:spot[1]] = np.array([3.0, 3.0], dtype=np.float32)
    if ball == "absent":
        ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
    elif ball == "stationary":
        ball_xy = np.tile(np.array([52.0, 34.0], dtype=np.float32), (n, 1))
    t = data.Tracking(player_xy=players, player_speed=np.zeros(players.shape[:2], np.float32),
                      ball_xy=ball_xy.astype(np.float32), slot_team=team,
                      slot_is_goalkeeper=np.zeros(22, bool), pitch=PITCH, fps=FPS)
    if mask_ball is not None:
        t = t.with_ball_masked(mask_ball)
    return t


def decode(tracking, config=ON):
    n = tracking.n_frames
    return predict.predict_codes(tracking, config, "decoder", [(0, n)])[0]


def profile(n, segments):
    """Piecewise-constant speed profile: segments = [(start, stop, speed), ...]."""
    v = np.full(n, segments[0][2], dtype=float)
    for a, b, s in segments:
        v[a:b] = s
    return v


class TestStrictSpeed(unittest.TestCase):
    def line(self, n=60, v=2.0):
        x = np.arange(n, dtype=np.float32) * (v / FPS)
        return np.stack([x, np.zeros(n, np.float32)], axis=1)

    def test_clean_motion_gives_the_right_speed_and_edges_are_unavailable(self):
        s = dead_ball.strict_speed(self.line(), 4, FPS, 1.0, 12.0)
        np.testing.assert_allclose(s[4:-4], 2.0, atol=1e-4)
        self.assertTrue(np.isnan(s[:4]).all() and np.isnan(s[-4:]).all())

    def test_missing_frames_inside_the_window_invalidate_it(self):
        xy = self.line()
        xy[30:32] = np.nan                                   # a 2-frame gap
        s = dead_ball.strict_speed(xy, 4, FPS, 1.0, 12.0)
        self.assertTrue(np.isnan(s[26:36]).all())            # every window that spans the gap
        self.assertTrue(np.isfinite(s[10]) and np.isfinite(s[45]))

    def test_identity_jump_is_not_read_as_motion(self):
        xy = self.line()
        xy[30:] += np.array([3.0, 0.0], dtype=np.float32)    # re-acquired as a different player 3 m away
        s = dead_ball.strict_speed(xy, 4, FPS, 1.0, 12.0)
        self.assertTrue(np.isnan(s[26:34]).all())            # windows spanning the jump are rejected
        np.testing.assert_allclose(s[10], 2.0, atol=1e-4)
        np.testing.assert_allclose(s[45], 2.0, atol=1e-4)

    def test_speeds_above_the_cap_are_discarded(self):
        s = dead_ball.strict_speed(self.line(v=20.0), 4, FPS, 1.0, 12.0)
        self.assertTrue(np.isnan(s).all())

    def test_a_missing_player_is_not_a_stationary_player(self):
        xy = np.stack([self.line(), self.line()], axis=1)    # (T, 2 players, 2)
        xy[:, 1] = np.nan
        s = dead_ball.strict_speed(xy, 4, FPS, 1.0, 12.0)
        self.assertTrue(np.isnan(s[:, 1]).all())
        self.assertTrue(np.isfinite(s[10, 0]))


class TestRollingStatistics(unittest.TestCase):
    def test_median_ignores_nan_and_requires_coverage(self):
        x = np.full(101, 2.0)
        x[35:66] = np.nan
        med, cov = dead_ball.rolling_median(x, 10, 0.5)
        self.assertAlmostEqual(med[10], 2.0)
        self.assertTrue(np.isnan(med[50]))                   # a window with no data has no statistic
        self.assertEqual(cov[50], 0.0)

    def test_local_quantile_needs_enough_observations(self):
        v = np.full(3000, np.nan)
        v[:100] = 2.0
        level = dead_ball.local_quantile(v, 500, 0.75, 750)
        self.assertTrue(np.isnan(level).all())               # 100 observations < 750 required


class TestDeadBallEvidence(unittest.TestCase):
    N = 3000

    def evidence_for(self, tracking, config=ON):
        return dead_ball.dead_ball_evidence(tracking, [(0, tracking.n_frames)], config)

    def test_no_usable_player_speeds_means_no_evidence_and_no_reliability(self):
        t = scene(3.0, n=self.N)
        players = t.player_xy.copy()
        players[:] = np.nan
        t = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        db = self.evidence_for(t)
        self.assertTrue((db.llr == 0).all())
        self.assertTrue((db.reliability == 0).all())

    def test_too_few_valid_players_gives_no_statistic(self):
        t = scene(3.0, n=self.N)
        players = t.player_xy.copy()
        players[:, 5:] = np.nan                               # only 5 players left (< 8 required)
        t = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        self.assertTrue((self.evidence_for(t).llr == 0).all())

    def test_evidence_does_not_depend_on_ball_visibility_for_the_activity_cue(self):
        speed = profile(self.N, [(0, self.N, 3.5), (1200, 1700, 0.8)])
        seen = self.evidence_for(scene(speed))
        absent = self.evidence_for(scene(speed, ball="absent"))
        np.testing.assert_allclose(seen.llr, absent.llr)      # ball absence adds no dead evidence

    def test_fast_players_with_a_missing_ball_are_evidence_for_live_play(self):
        db = self.evidence_for(scene(4.0, n=self.N, ball="absent"))
        mid = db.llr[500:2500]
        self.assertTrue((mid < 0).all())

    def test_dead_cue_never_fires_from_an_unobserved_ball_alone(self):
        t = scene(4.0, n=self.N, ball="absent")
        f = features.compute_features(t, [(0, self.N)], ON)
        self.assertTrue((evidence.dead_evidence(f, ON) == 0).all())

    def test_ball_far_outside_the_plausible_area_is_treated_as_missing(self):
        raw = np.array([[0.5, 0.5], [3.0, 0.5]])             # second point 200+ m outside
        out = data.sanitize(raw, PITCH, DEFAULT.coord_margin_m)
        self.assertTrue(np.isfinite(out[0]).all() and np.isnan(out[1]).all())

    def test_ball_modestly_outside_counts_but_an_untracked_ball_does_not(self):
        t = scene(3.0, n=50)
        ball = t.ball_xy.copy()
        ball[10:20] = np.array([52.0, 71.5], dtype=np.float32)   # 3.5 m beyond the touchline
        ball[30:40] = np.nan
        t = data.Tracking(t.player_xy, t.player_speed, ball, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        f = features.compute_features(t, [(0, 50)], ON)
        dead = evidence.dead_evidence(f, ON)
        self.assertTrue((dead[10:20] > 0).all())
        self.assertTrue((dead[30:40] == 0).all())

    def test_stationary_ball_during_active_play_is_not_dead_evidence(self):
        moving = self.evidence_for(scene(4.0, n=self.N)).llr
        still = self.evidence_for(scene(4.0, n=self.N, ball="stationary")).llr
        np.testing.assert_allclose(moving, still)             # no stationary-ball cue exists
        codes = decode(scene(4.0, n=self.N, ball="stationary"))
        self.assertFalse((codes == decoder.D).any())
        # even with the optional ball-motion cue switched on, fast players outweigh it
        codes = decode(scene(4.0, n=self.N, ball="stationary"), ON.replace(dead_use_ball_motion=True))
        self.assertFalse((codes == decoder.D).any())

    def test_a_brief_slowdown_inside_active_play_is_averaged_out(self):
        speed = profile(self.N, [(0, self.N, 4.0), (1500, 1550, 1.0)])     # 2 s of slowing
        self.assertFalse((decode(scene(speed)) == decoder.D).any())

    def test_a_clear_stoppage_is_detected_and_play_around_it_is_not(self):
        speed = profile(self.N, [(0, self.N, 4.0), (1200, 1700, 0.8)])     # 20 s of walking
        codes = decode(scene(speed, spot=(1200, 1700)))
        stop = codes[1200:1700]
        self.assertGreater((stop == decoder.D).mean(), 0.6)
        self.assertFalse((codes[:1000] == decoder.D).any())
        self.assertFalse((codes[1950:] == decoder.D).any())

    def test_sudden_restart_ends_the_dead_state_within_the_aggregation_window(self):
        speed = profile(self.N, [(0, self.N, 4.0), (1200, 1700, 0.8)])
        codes = decode(scene(speed, spot=(1200, 1700)))
        self.assertTrue((codes == decoder.D).any())
        last_dead = np.flatnonzero(codes == decoder.D).max()
        self.assertLess(last_dead, 1700 + 2 * ON.dead_window_halfwidth)     # within one rolling window of the restart
        self.assertTrue((codes[2000:] != decoder.D).all())

    def test_fully_observed_normal_possession_stays_alive_and_home(self):
        codes = decode(scene(3.5, n=self.N))
        self.assertTrue((codes == decoder.H).all())

    def test_player_identity_jumps_do_not_create_a_stoppage_or_fake_speed(self):
        t = scene(4.0, n=self.N)
        players = t.player_xy.copy()
        for frame in range(100, self.N, 150):
            players[frame:, 3] += np.array([2.5, 0.0], dtype=np.float32)    # repeated re-acquisitions
        t2 = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        a, b = self.evidence_for(t).llr, self.evidence_for(t2).llr
        np.testing.assert_allclose(a, b, atol=1e-6)           # the median over players is untouched by one bad track

    def test_negative_evidence_can_be_disabled(self):
        on = self.evidence_for(scene(4.0, n=self.N), ON).llr
        off = self.evidence_for(scene(4.0, n=self.N), ON.replace(dead_allow_negative=False)).llr
        self.assertTrue((on[500:2500] < 0).all())
        self.assertTrue((off >= 0).all())

    def test_cues_are_independently_switchable(self):
        t = scene(profile(self.N, [(0, self.N, 3.0)]), n=self.N)
        none = self.evidence_for(t, ON.replace(dead_use_activity=False, dead_use_near_ball=False, dead_use_ball_motion=False))
        self.assertTrue((none.llr == 0).all() and not none.parts)
        near = self.evidence_for(t, ON.replace(dead_use_activity=False, dead_use_near_ball=True))
        self.assertEqual(sorted(near.parts), ["near_ball"])
        both = self.evidence_for(t, ON.replace(dead_use_near_ball=True))
        self.assertEqual(sorted(both.parts), ["activity", "near_ball"])

    def test_reliability_reflects_coverage(self):
        t = scene(3.0, n=self.N)
        players = t.player_xy.copy()
        players[1000:2000, :] = np.nan                        # all players unobserved for 40 s
        t = data.Tracking(players, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        db = self.evidence_for(t)
        self.assertTrue((db.reliability[1100:1900] == 0).all())
        self.assertTrue((db.llr[1100:1900] == 0).all())
        self.assertGreater(db.reliability[300], 0.9)

    def test_blocks_are_isolated_and_outside_frames_are_untouched(self):
        n = 2 * self.N
        speed = np.concatenate([np.full(self.N, 4.0), np.full(self.N, 0.6)])
        t = scene(speed, n=n)
        both = dead_ball.dead_ball_evidence(t, [(0, self.N), (self.N + 100, n)], ON)
        second_alone = dead_ball.dead_ball_evidence(t, [(self.N + 100, n)], ON)
        np.testing.assert_allclose(both.llr[self.N + 100:], second_alone.llr[self.N + 100:])
        self.assertTrue((both.llr[self.N:self.N + 100] == 0).all())
        first_alone = dead_ball.dead_ball_evidence(t, [(0, self.N)], ON)
        np.testing.assert_allclose(both.llr[:self.N], first_alone.llr[:self.N])

    def test_deterministic(self):
        t = scene(profile(self.N, [(0, self.N, 3.5), (800, 1300, 0.9)]))
        np.testing.assert_array_equal(self.evidence_for(t).llr, self.evidence_for(t).llr)
        np.testing.assert_array_equal(decode(t), decode(t))

    def test_native_speed_source_is_selectable(self):
        t = scene(3.0, n=self.N)
        native = np.full(t.player_xy.shape[:2], 3.0, np.float32)
        t = data.Tracking(t.player_xy, native, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, PITCH, FPS)
        db = self.evidence_for(t, ABS.replace(dead_speed_source="native"))
        self.assertTrue(np.isfinite(db.llr).all())
        with self.assertRaises(ValueError):
            self.evidence_for(t, ABS.replace(dead_speed_source="nope"))


class TestVisibilityCompensation(unittest.TestCase):
    def test_offset_is_the_mean_alive_bonus_on_frames_without_ownership_evidence(self):
        own = np.array([2.0, -2.0, 0.0, 0.0])
        weight = np.array([1.0, 1.0, 0.0, 0.0])
        off = dead_ball.ownership_compensation(own, weight, [(0, 4)], ON)
        np.testing.assert_allclose(off, [0.0, 0.0, 1.0, 1.0])          # mean(|own|/2) = 1.0

    def test_disabled_or_unobserved_block_gives_zero(self):
        own, weight = np.zeros(4), np.zeros(4)
        self.assertTrue((dead_ball.ownership_compensation(own, weight, [(0, 4)], ON) == 0).all())
        own, weight = np.array([2.0, 2.0, 0, 0]), np.array([1.0, 1.0, 0, 0])
        cfg = ON.replace(dead_compensate_missing_ownership=False)
        self.assertTrue((dead_ball.ownership_compensation(own, weight, [(0, 4)], cfg) == 0).all())

    def test_a_tightly_controlled_ball_is_never_called_dead_by_movement_alone(self):
        speed = profile(3000, [(0, 3000, 4.0), (1200, 1700, 0.8)])          # players walk, ball stays carried
        self.assertFalse((decode(scene(speed)) == decoder.D).any())

    def test_hiding_the_ball_does_not_create_dead_ball_calls(self):
        n = 3000
        speed = np.full(n, 3.0)
        mask = np.zeros(n, bool)
        mask[1000:2000] = True
        visible = decode(scene(speed))
        hidden = decode(scene(speed, mask_ball=mask))
        self.assertFalse((visible == decoder.D).any())
        self.assertFalse((hidden == decoder.D).any())


class TestIntegration(unittest.TestCase):
    def test_module_is_off_by_default_and_off_means_phase_4_2(self):
        self.assertFalse(DEFAULT.use_dead_ball_module)
        t = scene(profile(3000, [(0, 3000, 3.5), (1000, 1500, 0.8)]))
        off = predict.predict_codes(t, DEFAULT, "decoder", [(0, 3000)])[0]
        explicit = predict.predict_codes(t, DEFAULT.replace(use_dead_ball_module=False), "decoder", [(0, 3000)])[0]
        np.testing.assert_array_equal(off, explicit)
        self.assertFalse((off == decoder.D).any())             # without the module, slow players alone never make D

    def test_evidence_carries_the_module_output_for_diagnostics(self):
        t = scene(3.0, n=3000)
        _, _, ev = predict.predict_codes(t, ON, "decoder", [(0, 3000)])
        self.assertIn("dead_ball", ev.parts)
        self.assertEqual(ev.parts["dead_ball"].llr.shape, (3000,))

    def test_halves_stay_isolated_through_the_whole_pipeline(self):
        n = 8000
        speed = np.concatenate([np.full(3000, 4.0), np.full(2000, 4.0), np.full(3000, 0.6)])
        t = scene(speed, n=n)
        both = predict.predict_codes(t, ON, "decoder", [(0, 3000), (5000, 8000)])[0]
        alone = predict.predict_codes(t, ON, "decoder", [(5000, 8000)])[0]
        np.testing.assert_array_equal(both[5000:], alone[5000:])
        self.assertTrue((both[3000:5000] == decoder.X).all())

    def test_written_csv_still_follows_the_submission_contract(self):
        t = scene(profile(3000, [(0, 3000, 3.5), (1000, 1600, 0.7)]))
        codes = predict.predict_codes(t, ON, "decoder", [(0, 3000)])[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = predict.write_submission(os.path.join(tmp, "x.csv"), codes)
            with open(path) as fh:
                lines = fh.read().splitlines()
        self.assertEqual(lines[0], "frame,label")
        self.assertEqual(len(lines), 3001)
        self.assertTrue(set(l.split(",")[1] for l in lines[1:]) <= set("HADX"))


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
PHASE_4_2_SHA256 = "50f32a07fd2989f7c2a4e8e9b75f13acf133582f52f3b4f20e2d8f5f3417bba3"


@unittest.skipUnless(os.path.isfile(REAL_RAW), "tactical ATD not installed")
class TestPhase42Preserved(unittest.TestCase):
    def test_default_pipeline_reproduces_the_phase_4_2_csv_byte_for_byte(self):
        tracking = data.load_tracking()
        codes, _, _ = predict.predict_codes(tracking, DEFAULT, "decoder")
        with tempfile.TemporaryDirectory() as tmp:
            path = predict.write_submission(os.path.join(tmp, "p42.csv"), codes)
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(digest, PHASE_4_2_SHA256)


if __name__ == "__main__":
    unittest.main()
