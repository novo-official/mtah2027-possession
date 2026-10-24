"""Phase 4.5 tests: decoder tie correctness, dead-ball duration models and transition-cost safety.

Everything here is synthetic and checks implementation properties (symmetry, determinism, no
invented transitions, exactness against brute force). Nothing is evidence of accuracy on a match.
Two real-data tests (skipped without the data) pin the Phase 4.2 and 4.3 outputs through the named
legacy presets.

Run with:   .venv/bin/python -m unittest discover -s tests -v
"""

import hashlib
import itertools
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from solution import data, decoder, duration, predict  # noqa: E402
from solution.config import DEFAULT, PRESETS  # noqa: E402

H, A, D = decoder.H, decoder.A, decoder.D
LEGACY = DEFAULT
CORRECTED = DEFAULT.replace(decoder_late_ties=True)


def segment(state, n, strength=2.0):
    """(own, dead) evidence of length n supporting one state."""
    own, dead = np.zeros(n), np.zeros(n)
    if state == H:
        own[:] = strength
    elif state == A:
        own[:] = -strength
    elif state == D:
        dead[:] = strength
    return own, dead


def gap_case(a, b, gap=300, n=100):
    """Evidence for state a, an evidence-free gap, evidence for state b."""
    o1, d1 = segment(a, n)
    o2, d2 = segment(b, n)
    return np.concatenate([o1, np.zeros(gap), o2]), np.concatenate([d1, np.zeros(gap), d2])


def run(own, dead, config, blocks=None):
    blocks = blocks or [(0, len(own))]
    return decoder.decode_blocks(own, dead, blocks, config)


def switches(codes):
    return (np.flatnonzero(np.diff(codes) != 0) + 1).tolist()


def mirror(codes):
    return np.where(codes == H, A, np.where(codes == A, H, codes))


class TestTieCorrectness(unittest.TestCase):
    PAIRS = [(H, A), (A, H), (H, D), (A, D), (D, H), (D, A)]

    def test_root_cause_legacy_places_a_switch_by_the_index_of_the_state_entered(self):
        # first maximum wins: a switch INTO a higher-index state lands late, INTO a lower-index state early
        expected = {(H, A): 400, (A, H): 100, (H, D): 400, (A, D): 400, (D, H): 100, (D, A): 100}
        for a, b in self.PAIRS:
            own, dead = gap_case(a, b)
            self.assertEqual(switches(run(own, dead, LEGACY)), [expected[(a, b)]], "legacy %d->%d" % (a, b))

    def test_corrected_decoder_places_every_state_pair_at_the_end_of_the_evidence_free_gap(self):
        for a, b in self.PAIRS:
            own, dead = gap_case(a, b)
            self.assertEqual(switches(run(own, dead, CORRECTED)), [400], "corrected %d->%d" % (a, b))

    def test_home_away_mirror_symmetry_for_all_pairs_in_the_corrected_decoder(self):
        for a, b in self.PAIRS:
            own, dead = gap_case(a, b)
            normal = run(own, dead, CORRECTED)
            flipped = run(-own, dead, CORRECTED)               # swap the teams' evidence
            np.testing.assert_array_equal(mirror(normal), flipped)

    def test_legacy_is_not_mirror_symmetric_which_is_why_the_correction_exists(self):
        own, dead = gap_case(H, A)
        self.assertNotEqual(switches(run(own, dead, LEGACY)), switches(run(-own, dead, LEGACY)))

    def test_evidence_free_prefix_takes_the_first_supported_state_without_an_invented_switch(self):
        for state in (H, A, D):
            o, d = segment(state, 100)
            own, dead = np.concatenate([np.zeros(150), o]), np.concatenate([np.zeros(150), d])
            for config in (LEGACY, CORRECTED):
                codes = run(own, dead, config)
                self.assertEqual(switches(codes), [], "prefix before %d" % state)
                self.assertTrue((codes == state).all())

    def test_evidence_free_suffix_carries_the_last_state(self):
        for state in (H, A, D):
            o, d = segment(state, 100)
            own, dead = np.concatenate([o, np.zeros(150)]), np.concatenate([d, np.zeros(150)])
            for config in (LEGACY, CORRECTED):
                codes = run(own, dead, config)
                self.assertTrue((codes == state).all())

    def test_fully_evidence_free_block_is_x_when_corrected_and_h_when_legacy(self):
        own = dead = np.zeros(200)
        self.assertTrue((run(own, dead, CORRECTED) == decoder.X).all())
        self.assertTrue((run(own, dead, LEGACY) == H).all())       # the lowest-index artefact, documented
        # and an evidence-free half does not disturb the other half
        both = np.concatenate([np.zeros(100), np.full(100, 2.0)])
        codes = run(both, np.zeros(200), CORRECTED, blocks=[(0, 100), (100, 200)])
        self.assertTrue((codes[:100] == decoder.X).all() and (codes[100:] == H).all())

    def test_weak_supporting_evidence_inside_the_gap_is_followed_not_delayed(self):
        # tiny evidence for the incoming team inside the gap breaks the tie: switch at the START
        own, dead = gap_case(H, A)
        own[100:400] = -1e-3
        for config in (LEGACY, CORRECTED):
            sw = switches(run(own, dead, config))
            self.assertEqual(len(sw), 1)
            self.assertLessEqual(sw[0], 101)
        # weak evidence for the outgoing team: switch at the END
        own, dead = gap_case(H, A)
        own[100:400] = 1e-3
        for config in (LEGACY, CORRECTED):
            sw = switches(run(own, dead, config))
            self.assertEqual(len(sw), 1)
            self.assertGreaterEqual(sw[0], 399)

    def test_conflicting_weak_evidence_does_not_create_switches(self):
        own, dead = gap_case(H, H)
        own[100:400] = np.where(np.arange(300) % 2 == 0, 5e-3, -5e-3)
        for config in (LEGACY, CORRECTED):
            self.assertEqual(switches(run(own, dead, config)), [])

    def test_ties_are_exact_only_nonzero_evidence_is_never_treated_as_a_tie(self):
        # away -> home across a gap. Evidence of 1e-6 per frame is far below any switch cost but is
        # not zero, so it is followed in both decoders: it decides where the switch lands.
        own, dead = gap_case(A, H)
        own[100:400] = 1e-6                                 # favours the incoming team (home)
        for config in (LEGACY, CORRECTED):
            sw = switches(run(own, dead, config))
            self.assertEqual(len(sw), 1)
            self.assertLessEqual(sw[0], 101)                # moved to the START: not delayed to the end
        own[100:400] = -1e-6                                # favours the outgoing team (away)
        for config in (LEGACY, CORRECTED):
            sw = switches(run(own, dead, config))
            self.assertEqual(len(sw), 1)
            self.assertGreaterEqual(sw[0], 399)             # held until the end

    def test_halftime_isolation_holds_with_the_corrected_decoder(self):
        own = np.concatenate([np.full(100, 1.0), np.full(100, 0.0), np.full(100, -0.05)])
        dead = np.zeros(300)
        both = run(own, dead, CORRECTED, blocks=[(0, 100), (200, 300)])
        alone = run(own[200:], dead[200:], CORRECTED, blocks=[(0, 100)])
        np.testing.assert_array_equal(both[200:], alone)
        self.assertTrue((both[100:200] == decoder.X).all())

    def test_repeated_runs_and_state_enumeration_do_not_change_the_result(self):
        rng = np.random.RandomState(8)
        own, dead = rng.randn(4000), np.abs(rng.randn(4000)) * (rng.rand(4000) > 0.9)
        a, b = run(own, dead, CORRECTED), run(own, dead, CORRECTED)
        np.testing.assert_array_equal(a, b)
        # tie-free data: corrected equals legacy exactly
        np.testing.assert_array_equal(a, run(own, dead, LEGACY))


class TestSoftDurationDecoder(unittest.TestCase):
    cost = decoder.transition_costs(DEFAULT)
    init = np.zeros(3)

    def brute(self, em, K, lam):
        best, path = -1e18, None
        for p in itertools.product(range(3), repeat=len(em)):
            s, run_len = em[0, p[0]], (K if p[0] == D else 0)
            for t in range(1, len(p)):
                i, j = p[t - 1], p[t]
                s += em[t, j] - self.cost[i, j]
                if j == D:
                    run_len = 1 if i != D else run_len + 1
                if i == D and j != D:
                    s -= lam * max(0.0, 1.0 - min(run_len, K) / K)
            if s > best + 1e-12:
                best, path = s, p
        return best

    def score(self, em, p, K, lam):
        s, run_len = em[0, p[0]], (K if p[0] == D else 0)
        for t in range(1, len(p)):
            i, j = p[t - 1], p[t]
            s += em[t, j] - self.cost[i, j]
            if j == D:
                run_len = 1 if i != D else run_len + 1
            if i == D and j != D:
                s -= lam * max(0.0, 1.0 - min(run_len, K) / K)
        return s

    def test_matches_brute_force_enumeration(self):
        rng = np.random.RandomState(0)
        for _ in range(80):
            em = rng.randn(7, 3) * 1.5
            path = duration.viterbi_min_duration(em, self.cost, self.init, 3, 5.0)
            self.assertAlmostEqual(self.score(em, tuple(path.tolist()), 3, 5.0), self.brute(em, 3, 5.0), places=9)

    def test_zero_penalty_equals_the_first_order_decoder(self):
        rng = np.random.RandomState(1)
        for _ in range(50):
            em = rng.randn(80, 3) * 2
            np.testing.assert_array_equal(decoder.viterbi(em, self.cost, self.init),
                                          duration.viterbi_min_duration(em, self.cost, self.init, 9, 0.0))

    def test_a_brief_weakly_supported_stoppage_is_discouraged_but_a_strongly_supported_one_survives(self):
        n = 200
        own = np.full(n, 2.0)                                  # live play supported throughout (+1 nat per frame for home)
        weak = np.zeros(n)
        weak[90:102] = 2.5                                     # 12 frames: 1.5 nats/frame net, 18 vs enter+exit 16
        strong = np.zeros(n)
        strong[90:102] = 5.0                                   # 12 frames: 4 nats/frame net
        soft = DEFAULT.replace(duration_model="soft")
        self.assertTrue((run(own, weak, DEFAULT) == D).any())              # the first-order decoder accepts it
        self.assertFalse((run(own, weak, soft) == D).any())                # soft prior: not worth it
        self.assertGreaterEqual(int((run(own, strong, soft) == D).sum()), 10)   # no hard minimum duration

    def test_dead_intervals_touching_the_block_edges_are_not_penalised(self):
        n = 120
        start = np.zeros(n); start[:8] = 2.0
        end = np.zeros(n); end[-8:] = 2.0
        soft = DEFAULT.replace(duration_model="soft")
        for dead in (start, end):
            np.testing.assert_array_equal(run(np.zeros(n), dead, soft) == D, run(np.zeros(n), dead, DEFAULT) == D)

    def test_long_dead_interval_is_unaffected_by_the_penalty(self):
        n = 400
        dead = np.zeros(n); dead[100:300] = 1.0
        soft = DEFAULT.replace(duration_model="soft")
        np.testing.assert_array_equal(run(np.zeros(n), dead, soft), run(np.zeros(n), dead, DEFAULT))

    def test_symmetric_in_the_two_teams_and_isolated_between_halves(self):
        rng = np.random.RandomState(5)
        own, dead = rng.randn(600), np.abs(rng.randn(600)) * (rng.rand(600) > 0.8)
        soft = DEFAULT.replace(duration_model="soft", decoder_late_ties=True)
        a = run(own, dead, soft)
        b = run(-own, dead, soft)
        np.testing.assert_array_equal(mirror(a), b)
        both = run(own, dead, soft, blocks=[(0, 250), (350, 600)])
        alone = run(own, dead, soft, blocks=[(350, 600)])
        np.testing.assert_array_equal(both[350:], alone[350:])

    def test_soft_decoder_refuses_time_varying_costs_and_short_chains(self):
        soft = DEFAULT.replace(duration_model="soft")
        z = np.zeros(10)
        with self.assertRaises(ValueError):
            decoder.decode_blocks(z + 1, z, [(0, 10)], soft, switch_extra=(z, z))
        with self.assertRaises(ValueError):
            duration.viterbi_min_duration(np.zeros((5, 3)), self.cost, self.init, 1, 1.0)

    def test_evidence_free_input_with_the_soft_decoder_is_x_when_corrected(self):
        soft = DEFAULT.replace(duration_model="soft", decoder_late_ties=True)
        self.assertTrue((run(np.zeros(60), np.zeros(60), soft) == decoder.X).all())


class TestHysteresis(unittest.TestCase):
    def test_armed_runs_need_the_enter_level_and_last_until_the_exit_level(self):
        x = np.zeros(400)
        x[100:140] = 0.5                                      # sustained: arms
        x[300:303] = 0.5                                      # lone spike: smoothed mean stays below enter
        armed = duration.armed_mask(x, 37, 0.20, 0.05)
        self.assertTrue(armed[100:140].all())
        self.assertFalse(armed[300:303].any())
        self.assertFalse(armed[0:40].any())

    def test_exit_level_must_not_exceed_enter_level(self):
        with self.assertRaises(ValueError):
            duration.armed_mask(np.zeros(10), 3, 0.1, 0.5)

    def test_conditioning_attenuates_unarmed_positives_and_armed_negatives_only(self):
        cfg = DEFAULT.replace(duration_model="hysteresis")
        x = np.zeros(400)
        x[100:160] = 0.6
        x[125] = -0.4                                         # a dip inside the stoppage
        x[300] = 0.6                                          # an isolated spike elsewhere
        y = duration.hysteresis_dead(x, [(0, 400)], cfg)
        self.assertAlmostEqual(y[110], 0.6)                   # armed positive kept
        self.assertAlmostEqual(y[125], -0.4 * cfg.armed_negative_scale)
        self.assertAlmostEqual(y[300], 0.6 * cfg.unarmed_positive_scale)
        self.assertEqual(y[50], 0.0)

    def test_blocks_do_not_leak_and_the_conditioning_has_no_team_dependence(self):
        cfg = DEFAULT.replace(duration_model="hysteresis")
        x = np.zeros(600)
        x[100:160] = 0.6
        x[400:460] = 0.6
        both = duration.hysteresis_dead(x, [(0, 300), (300, 600)], cfg)
        a = duration.hysteresis_dead(x, [(0, 300)], cfg)
        b = duration.hysteresis_dead(x, [(300, 600)], cfg)
        np.testing.assert_allclose(both[:300], a[:300])
        np.testing.assert_allclose(both[300:], b[300:])

    def test_hysteresis_stops_a_flickering_score_from_fragmenting_a_stoppage(self):
        n = 400
        dead = np.zeros(n)
        dead[100:300] = np.where(np.arange(200) % 12 < 9, 0.9, -0.9)       # a stoppage with periodic dips
        plain = run(np.zeros(n), dead, DEFAULT)
        cond = run(np.zeros(n), duration.hysteresis_dead(dead, [(0, n)], DEFAULT.replace(duration_model="hysteresis")), DEFAULT)
        pieces = lambda c: int(np.sum(np.diff((c == D).astype(int)) == 1))
        self.assertLessEqual(pieces(cond), pieces(plain))
        self.assertGreater((cond[100:300] == D).mean(), 0.8)


class TestTransitionCostSafety(unittest.TestCase):
    def test_defaults_are_safe_and_the_soft_penalty_only_widens_the_margin(self):
        base = duration.check_transition_safety(DEFAULT)
        soft = duration.check_transition_safety(DEFAULT.replace(duration_model="soft"))
        self.assertTrue(base["safe"] and soft["safe"])
        self.assertGreaterEqual(soft["margin"], base["margin"])

    def test_an_unsafe_cost_structure_is_detected(self):
        self.assertFalse(duration.check_transition_safety(DEFAULT.replace(switch_cost=30.0))["safe"])

    def test_the_detour_loophole_is_real_when_the_direct_switch_is_too_expensive(self):
        # H evidence then A evidence with no dead evidence at all: a direct switch is the only honest answer
        own = np.concatenate([np.full(120, 1.0), np.full(120, -1.0)])
        dead = np.zeros(240)
        honest = run(own, dead, DEFAULT)
        self.assertFalse((honest == D).any())
        expensive = DEFAULT.replace(switch_cost=30.0)         # direct 30 > enter 10 + exit 6
        phantom = run(own, dead, expensive)
        self.assertTrue((phantom == D).any())                  # the exploit: change team through an invented stoppage

    def test_no_phantom_dead_state_without_dead_evidence_on_random_ownership(self):
        rng = np.random.RandomState(11)
        configs = [LEGACY, CORRECTED, DEFAULT.replace(duration_model="soft"),
                   DEFAULT.replace(duration_model="soft", decoder_late_ties=True)]
        for seed in range(15):
            own = rng.randn(1500) * rng.choice([0.5, 1.0, 3.0])
            for config in configs:
                self.assertFalse((run(own, np.zeros(1500), config) == D).any(), "seed %d" % seed)

    def test_the_soft_penalty_cannot_open_a_cheaper_indirect_route(self):
        # across a sweep of penalties the direct switch stays the cheapest route without dead evidence
        own = np.concatenate([np.full(80, 1.0), np.full(80, -1.0)])
        for penalty in (0.0, 3.0, 6.0, 12.0, 30.0):
            soft = DEFAULT.replace(duration_model="soft", short_dead_penalty=penalty)
            self.assertFalse((run(own, np.zeros(160), soft) == D).any())

    def test_strong_dead_evidence_still_justifies_home_dead_away(self):
        own = np.concatenate([np.full(100, 1.0), np.zeros(80), np.full(100, -1.0)])
        dead = np.concatenate([np.zeros(100), np.full(80, 1.5), np.zeros(100)])
        for config in (LEGACY, CORRECTED, DEFAULT.replace(duration_model="soft")):
            codes = run(own, dead, config)
            self.assertTrue((codes[:95] == H).all() and (codes[-95:] == A).all())
            self.assertGreaterEqual(int((codes[100:180] == D).sum()), 70)

    def test_completely_missing_ownership_evidence_yields_no_invented_possession_changes(self):
        dead = np.zeros(600)
        dead[200:400] = 1.0
        for config in (LEGACY, CORRECTED, DEFAULT.replace(duration_model="soft")):
            codes = run(np.zeros(600), dead, config)
            alive = codes[codes != D]
            self.assertEqual(len(set(alive.tolist())) <= 1, True)      # one team label at most, never a switch


class TestMissingBallInvariantsWithDurationModels(unittest.TestCase):
    def test_hidden_ball_creates_no_dead_calls_under_each_duration_model(self):
        import test_dead_ball as tdb
        n = 3000
        mask = np.zeros(n, bool)
        mask[1000:2200] = True
        for model in ("none", "hysteresis", "soft", "both"):
            cfg = tdb.ON.replace(duration_model=model)
            visible = tdb.decode(tdb.scene(np.full(n, 3.0)), cfg)
            hidden = tdb.decode(tdb.scene(np.full(n, 3.0), mask_ball=mask), cfg)
            self.assertFalse((visible == D).any(), model)
            self.assertFalse((hidden == D).any(), model)

    def test_a_supported_stoppage_with_the_ball_missing_is_still_decoded_as_dead(self):
        import test_dead_ball as tdb
        n = 3000
        speed = tdb.profile(n, [(0, n, 4.0), (1200, 1900, 0.7)])
        for model in ("none", "soft"):
            cfg = tdb.ON.replace(duration_model=model)
            t = tdb.scene(speed, ball="absent")
            codes = tdb.decode(t, cfg)
            self.assertGreater((codes[1250:1850] == D).mean(), 0.5, model)
            self.assertFalse((codes[:1000] == D).any(), model)


class TestRemainingScenarios(unittest.TestCase):
    """Decoder-level scenarios from the Phase 4.5 robustness list, under every duration model."""
    MODELS = ("none", "hysteresis", "soft", "both")

    def configs(self):
        for late in (False, True):
            for model in self.MODELS:
                yield DEFAULT.replace(decoder_late_ties=late, duration_model=model)

    def decode_with_conditioning(self, own, dead, config):
        if config.duration_model in ("hysteresis", "both"):
            dead = duration.hysteresis_dead(dead, [(0, len(own))], config)
        return run(own, dead, config)

    def test_a_genuine_short_possession_survives_every_configuration(self):
        own = np.concatenate([np.full(150, 2.0), np.full(30, -2.0), np.full(150, 2.0)])      # a 1.2 s away possession
        for config in self.configs():
            codes = self.decode_with_conditioning(own, np.zeros(len(own)), config)
            self.assertGreaterEqual(int((codes[150:180] == A).sum()), 28)
            self.assertFalse((codes == D).any())

    def test_a_sudden_restart_after_a_long_stoppage_ends_dead_promptly(self):
        own = np.concatenate([np.full(100, 2.0), np.zeros(600), np.full(200, -2.0)])
        dead = np.concatenate([np.zeros(100), np.full(600, 1.0), np.full(200, -1.0)])
        for config in self.configs():
            codes = self.decode_with_conditioning(own, dead, config)
            last_dead = int(np.flatnonzero(codes == D).max())
            self.assertGreaterEqual(last_dead, 650)
            self.assertLessEqual(last_dead, 700 + 40)                 # live evidence ends it within ~1.6 s
            self.assertTrue((codes[760:] == A).all())

    def test_weak_dead_evidence_during_active_play_never_creates_a_stoppage(self):
        rng = np.random.RandomState(2)
        own = np.where(np.arange(1500) % 400 < 300, 2.0, -2.0)        # alternating live possessions
        dead = 0.12 + 0.2 * rng.randn(1500)                           # weak, noisy, mildly positive
        for config in self.configs():
            codes = self.decode_with_conditioning(own, dead, config)
            self.assertLess(float((codes == D).mean()), 0.02, config.duration_model)

    def test_strong_ownership_evidence_outweighs_a_temporary_slowdown(self):
        own = np.full(600, 3.0)                                       # a team clearly in control throughout
        dead = np.zeros(600)
        dead[200:330] = 0.6                                           # players slow for ~5 s, max movement score
        for config in self.configs():
            self.assertFalse((self.decode_with_conditioning(own, dead, config) == D).any())

    def test_a_long_stopped_period_is_one_interval_not_many(self):
        own = np.concatenate([np.full(100, 2.0), np.zeros(900), np.full(100, 2.0)])
        dead = np.concatenate([np.zeros(100), np.full(900, 0.8) + 0.3 * np.sin(np.arange(900)), np.zeros(100)])
        for config in self.configs():
            codes = self.decode_with_conditioning(own, dead, config)
            pieces = int(np.sum(np.diff((codes == D).astype(int)) == 1))
            self.assertLessEqual(pieces, 2)
            self.assertGreater(float((codes[150:950] == D).mean()), 0.9)

    def test_a_supported_stoppage_starts_and_ends_near_its_evidence(self):
        own = np.full(600, 1.0)
        dead = np.zeros(600)
        dead[200:400] = 3.0                                           # 8 s of strong dead evidence
        for config in self.configs():
            codes = self.decode_with_conditioning(own, dead, config)
            idx = np.flatnonzero(codes == D)
            self.assertLessEqual(abs(int(idx.min()) - 200), 30)
            self.assertLessEqual(abs(int(idx.max()) - 399), 30)


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
SHA_P42 = "50f32a07fd2989f7c2a4e8e9b75f13acf133582f52f3b4f20e2d8f5f3417bba3"
SHA_P43 = "367b7415e3f2f700115c1846f7ba3ce366b34203732539f4a81d37af79f148a2"


@unittest.skipUnless(os.path.isfile(REAL_RAW), "tactical ATD not installed")
class TestLegacyPresetsReproduceEarlierPhases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracking = data.load_tracking()

    def sha(self, config):
        codes, _, _ = predict.predict_codes(self.tracking, config, "decoder")
        with tempfile.TemporaryDirectory() as tmp:
            path = predict.write_submission(os.path.join(tmp, "x.csv"), codes)
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()

    def test_legacy_p42_preset(self):
        self.assertEqual(self.sha(PRESETS["legacy_p42"]), SHA_P42)

    def test_legacy_p43_preset(self):
        self.assertEqual(self.sha(PRESETS["legacy_p43"]), SHA_P43)

    def test_corrected_and_combined_presets_are_deterministic_and_graded_frames_are_never_x(self):
        for name in ("corrected", "combined"):
            a, b = self.sha(PRESETS[name]), self.sha(PRESETS[name])
            self.assertEqual(a, b)
        codes, _, _ = predict.predict_codes(self.tracking, PRESETS["combined"], "decoder")
        graded = np.zeros(len(codes), bool)
        for s, e in predict.block_ranges():
            graded[s:e] = True
        self.assertTrue((codes[graded] != decoder.X).all() and (codes[~graded] == decoder.X).all())


if __name__ == "__main__":
    unittest.main()
