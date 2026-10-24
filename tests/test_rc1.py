"""Phase 4.8 tests: the release-candidate (`rc1`) behaviour and the production inference path.

Synthetic tests check implementation properties of the frozen configuration (symmetry, half
isolation, missing-observation handling, transition safety, determinism, output format). They are
not evidence of accuracy. The real-data tests are skipped without the restricted tracking and report
only pass / fail.

Run with:   .venv/bin/python -m unittest discover -s tests -v
"""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from solution import data, decoder, features, model_registry, predict  # noqa: E402
from solution.config import DEFAULT, PRESETS, parse_overrides  # noqa: E402
from solution.duration import check_transition_safety  # noqa: E402
from test_solution import scene  # noqa: E402

H, A, D, X = decoder.H, decoder.A, decoder.D, decoder.X
RC1 = PRESETS["rc1"]
LEGACY = PRESETS["legacy_p42"]
PY = sys.executable


def play(parts):
    """Tracking from consecutive parts (n, side, ball_visible). side 'H'/'A': a player of that team stands next to
    the ball; 'N': nobody is near; 'OUT': nobody near and the ball is 5 m outside the touchline. Both teams always
    have 11 slots, so parts concatenate."""
    balls, players = [], []
    for n, side, visible in parts:
        ball = (-5.0, 34.0) if side == "OUT" else (30.0 if side == "H" else 60.0, 34.0)
        near_home = [[ball[0] + 1.0, 34.0]] if side == "H" else [[10.0, 62.0]]
        near_away = [[ball[0] + 1.0, 34.0]] if side == "A" else [[95.0, 62.0]]
        t = scene(n, ball, home_near=near_home, away_near=near_away)
        b = t.ball_xy.copy()
        if not visible:
            b[:] = np.nan
        balls.append(b)
        players.append(t.player_xy)
    base = scene(1, (30.0, 34.0), home_near=[[31.0, 34.0]], away_near=[[95.0, 62.0]])
    return data.Tracking(np.concatenate(players), np.zeros((sum(p.shape[0] for p in players), base.n_slots), np.float32),
                         np.concatenate(balls), base.slot_team, base.slot_is_goalkeeper, base.pitch, base.fps)


def run(tracking, config=RC1, blocks=None):
    blocks = [(0, tracking.n_frames)] if blocks is None else blocks
    codes, _, evidence = predict.predict_codes(tracking, config, "decoder", blocks)
    return codes, evidence


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def swap_teams(tracking):
    return data.Tracking(tracking.player_xy, tracking.player_speed, tracking.ball_xy, 1 - tracking.slot_team,
                         tracking.slot_is_goalkeeper, tracking.pitch, tracking.fps)


def switches(codes):
    return [int(i) for i in np.flatnonzero(np.diff(codes.astype(int)) != 0) + 1]


class TestRC1Definition(unittest.TestCase):
    def test_rc1_is_the_corrected_decoder_with_nothing_experimental(self):
        self.assertTrue(RC1.decoder_late_ties)
        self.assertEqual((RC1.use_dead_ball_module, RC1.dead_model, RC1.use_restart, RC1.ownership_v2, RC1.use_handover,
                          RC1.duration_model, RC1.emission_mode), (False, "legacy", False, False, False, "none", "legacy"))
        self.assertTrue(RC1.use_outside_cue)
        changed = {f for f in RC1.__dataclass_fields__ if getattr(RC1, f) != getattr(LEGACY, f)}
        self.assertEqual(changed, {"decoder_late_ties"})
        self.assertTrue(check_transition_safety(RC1)["safe"])

    def test_rc1_does_not_alias_an_older_preset_object(self):
        self.assertIsNot(PRESETS["rc1"], PRESETS["corrected"])
        self.assertEqual(model_registry.get("rc1").fingerprint, model_registry.get("corrected").fingerprint)

    def test_overrides_are_strict_booleans(self):
        self.assertTrue(parse_overrides(RC1, ["use_restart=on"]).use_restart)
        self.assertFalse(parse_overrides(RC1, ["decoder_late_ties=off"]).decoder_late_ties)
        with self.assertRaises(ValueError):
            parse_overrides(RC1, ["use_restart=maybe"])
        with self.assertRaises(KeyError):
            parse_overrides(RC1, ["no_such_field=1"])


class TestRC1Behaviour(unittest.TestCase):
    def test_possession_follows_the_nearest_team_and_changes_once(self):
        codes, _ = run(play([(600, "H", True), (600, "A", True)]))
        self.assertTrue((codes[:590] == H).all() and (codes[610:] == A).all())
        self.assertEqual(len(switches(codes)), 1)
        self.assertFalse((codes == D).any() or (codes == X).any())

    def test_team_symmetry(self):
        t = play([(400, "H", True), (50, "OUT", True), (150, "N", False), (400, "A", True), (200, "A", False), (300, "H", True)])
        a, _ = run(t)
        b, _ = run(swap_teams(t))
        np.testing.assert_array_equal(np.where(a == H, A, np.where(a == A, H, a)), b)

    def test_output_is_not_a_constant_or_hard_coded(self):
        codes, _ = run(play([(500, "H", True), (500, "A", True), (500, "H", True)]))
        self.assertEqual(sorted(set(codes.tolist())), [H, A])
        self.assertEqual(len(switches(codes)), 2)

    def test_determinism(self):
        t = play([(300, "H", True), (50, "OUT", True), (300, "N", False), (300, "A", True)])
        np.testing.assert_array_equal(run(t)[0], run(t)[0])

    def test_possession_is_carried_through_a_ball_gap_and_switches_only_when_evidence_returns(self):
        for first, second, code_a, code_b in (("H", "A", H, A), ("A", "H", A, H)):
            t = play([(300, first, True), (400, first, False), (400, second, True)])
            codes, _ = run(t)
            self.assertTrue((codes[:700] == code_a).all(), "state must be carried through the 400-frame gap")
            self.assertTrue((codes[705:] == code_b).all())
            self.assertGreaterEqual(switches(codes)[0], 700)      # not at the start of the unobserved stretch
            self.assertLessEqual(switches(codes)[0], 702)

    def test_legacy_tie_rule_moved_an_unsupported_switch_to_the_gap_start_rc1_does_not(self):
        t = play([(300, "A", True), (400, "A", False), (400, "H", True)])
        legacy, _ = run(t, LEGACY)
        rc1, _ = run(t, RC1)
        self.assertEqual(switches(legacy)[0], 300)               # the A->H switch sits at the start of the gap
        self.assertGreaterEqual(switches(rc1)[0], 700)

    def test_hidden_ball_alone_never_creates_a_stoppage(self):
        codes, _ = run(play([(300, "H", True), (1500, "H", False), (300, "H", True)]))
        self.assertTrue((codes == H).all())

    def test_missing_players_are_not_evidence(self):
        t = play([(300, "H", True), (300, "H", True), (300, "H", True)])
        xy = t.player_xy.copy()
        xy[300:600, t.slot_team == 1] = np.nan                      # the whole away team untracked for 12 s
        codes, evidence = run(data.Tracking(xy, t.player_speed, t.ball_xy, t.slot_team, t.slot_is_goalkeeper, t.pitch, t.fps))
        self.assertTrue((codes == H).all())
        self.assertTrue(np.isfinite(evidence.own_llr).all() and np.isfinite(evidence.dead_llr).all())

    def test_all_missing_block_is_unanswered_not_home(self):
        codes, _ = run(play([(500, "H", False)]))
        self.assertTrue((codes == X).all())

    def test_out_of_play_ball_is_dead_and_the_state_is_carried_through_the_following_gap(self):
        """Documents the accepted RC1 risk: D entered through the out-of-play cue is held through an
        evidence-free stretch and left only when evidence returns (never earlier, never much later)."""
        t = play([(300, "H", True), (60, "OUT", True), (400, "N", False), (300, "H", True)])
        codes, _ = run(t)
        self.assertTrue((codes[360:760] == D).all())
        self.assertTrue((codes[:290] == H).all())
        exit_at = int(np.flatnonzero(codes[760:] != D)[0]) + 760
        self.assertLessEqual(exit_at - 760, 20)
        self.assertTrue((codes[exit_at + 10:] == H).all())
        legacy, _ = run(t, LEGACY)
        self.assertLess(int(np.flatnonzero(legacy[360:] != D)[0]) + 360, 600)    # legacy left D inside the gap

    def test_halves_are_isolated_and_other_frames_are_x(self):
        t = play([(600, "H", True), (300, "A", True), (600, "A", True)])
        blocks = [(0, 600), (900, 1500)]
        base, _ = run(t, blocks=blocks)
        flipped = play([(600, "A", True), (300, "A", True), (600, "A", True)])
        other, _ = run(flipped, blocks=blocks)
        np.testing.assert_array_equal(base[900:], other[900:])      # nothing crosses halftime
        self.assertTrue((base[600:900] == X).all())
        self.assertNotEqual(int((base[:600] == H).sum()), int((other[:600] == H).sum()))

    def test_no_nan_or_inf_in_the_evidence_for_degenerate_input(self):
        t = play([(200, "N", False)])
        codes, evidence = run(t)
        for array in (evidence.own_llr, evidence.own_weight, evidence.dead_llr):
            self.assertTrue(np.isfinite(array).all())


class TestOutputContract(unittest.TestCase):
    def codes(self, n=games.GAMES[games.GAME]["n_video_frames"]):
        return np.random.RandomState(0).randint(0, 4, size=n).astype(np.int8)

    def test_file_matches_the_official_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = predict.write_submission(os.path.join(tmp, "nested", "x.csv"), self.codes())
            raw = read_bytes(path)
            table = pm.read_label_csv(path)
        self.assertTrue(raw.startswith(b"frame,label\n") and raw.endswith(b"\n") and b"\r" not in raw)
        raw.decode("utf-8")
        self.assertEqual(raw.count(b"\n"), 199319 + 1)
        self.assertIsNone(pm.validate_submission(table, 199319))

    def test_validate_codes_rejects_wrong_length_and_values(self):
        predict.validate_codes(self.codes())
        for bad in (self.codes(10), self.codes() + 5, self.codes().astype(float), np.zeros((3, 3), np.int8)):
            with self.assertRaises(ValueError):
                predict.validate_codes(bad)

    def test_write_is_atomic_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.csv")
            predict.write_submission(path, np.array([0, 1, 2, 3], dtype=np.int8))
            before = read_bytes(path)
            with mock.patch("solution.predict.os.replace", side_effect=OSError("disk gone")):
                with self.assertRaises(OSError):
                    predict.write_submission(path, np.array([3, 3, 3, 3], dtype=np.int8))
            self.assertEqual(read_bytes(path), before)       # the earlier complete file survives
            self.assertEqual(os.listdir(tmp), ["x.csv"])           # and no .tmp_submission_* debris

    def test_writing_over_protected_locations_is_refused(self):
        targets = [os.path.join(games.DATA_DIR, "tactical", "zz_prediction.csv"),
                   os.path.join(ROOT, "evaluation", "results", "baseline_zz_test.csv"),
                   os.path.join(ROOT, "evaluation", "gt", "zz_possession_gt.csv"),
                   os.path.join(ROOT, "starter", "zz.csv")]
        try:
            for target in targets:
                with self.assertRaises(ValueError, msg=target):
                    predict.write_submission(target, np.zeros(4, np.int8))
                self.assertFalse(os.path.exists(target), target)
        finally:
            for target in targets:
                if os.path.exists(target):
                    os.remove(target)

    def test_other_files_in_the_results_folder_are_not_protected(self):
        self.assertIsNone(predict._protected(os.path.join(ROOT, "evaluation", "results", "my_entry.csv")))
        self.assertIsNone(predict._protected(os.path.join(ROOT, "outputs", "rc1", "possession_rc1.csv")))


class TestCommandLine(unittest.TestCase):
    def cli(self, *args, env=None, cwd=None):
        environment = dict(os.environ, PYTHONPATH=ROOT, **(env or {}))
        return subprocess.run([PY, "-m", "solution.predict", *args], capture_output=True, text=True, env=environment,
                              cwd=cwd or ROOT, timeout=300)

    def test_describe_and_list_need_no_data(self):
        with tempfile.TemporaryDirectory() as empty:
            out = self.cli("--describe", "rc1", env={"MTAH_DATA_DIR": empty})
            self.assertEqual(out.returncode, 0)
            self.assertIn("release-candidate", out.stdout)
            self.assertIn(model_registry.get("rc1").fingerprint, out.stdout)
            listing = self.cli("--list-presets", env={"MTAH_DATA_DIR": empty})
        self.assertEqual(listing.returncode, 0)
        self.assertIn("rc1", listing.stdout)

    def test_missing_data_fails_clearly_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryDirectory() as out_dir:
            target = os.path.join(out_dir, "x.csv")
            result = self.cli("--preset", "rc1", "--out", target, env={"MTAH_DATA_DIR": empty})
            self.assertEqual(result.returncode, 2)
            self.assertIn("match data not found", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(os.listdir(out_dir), [])

    def test_protected_output_is_refused_before_any_work(self):
        result = self.cli("--preset", "rc1", "--out", os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing", result.stderr)
        self.assertNotIn("configuration:", result.stdout)

    def test_bad_override_is_a_clean_error(self):
        result = self.cli("--preset", "rc1", "--set", "use_restart=maybe", "--out", os.path.join(tempfile.gettempdir(), "zz.csv"))
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")


@unittest.skipUnless(os.path.isfile(REAL_RAW), "tactical ATD not installed")
class TestRC1ReleasePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = []
        for i in (1, 2):
            # independent executions, from a foreign working directory: no reliance on cwd or on outputs/
            path = os.path.join(cls.tmp.name, "rc1_{}.csv".format(i))
            result = subprocess.run([PY, "-m", "solution.predict", "--preset", "rc1", "--out", path], capture_output=True,
                                    text=True, cwd=cls.tmp.name, env=dict(os.environ, PYTHONPATH=ROOT), timeout=300)
            assert result.returncode == 0, result.stderr
            cls.paths.append(path)
        cls.blocks = predict.block_ranges()
        cls.codes = pm.label_codes(pm.read_label_csv(cls.paths[0]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_two_independent_runs_are_byte_identical_and_match_the_pinned_hash(self):
        a, b = (predict.file_sha256(p) for p in self.paths)
        self.assertEqual(a, b)
        self.assertEqual(a, model_registry.get("rc1").reference_sha256)
        self.assertEqual(a, model_registry.get("corrected").reference_sha256)

    def test_the_official_validator_accepts_the_file(self):
        result = subprocess.run([PY, os.path.join(ROOT, "evaluation", "validate_submission.py"), self.paths[0]],
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("199319 rows, frame 0..199318", result.stdout)

    def test_every_frame_once_in_order_and_only_valid_labels(self):
        table = pm.read_label_csv(self.paths[0])
        self.assertIsNone(pm.validate_submission(table, 199319))
        with open(self.paths[0], encoding="utf-8") as handle:
            lines = handle.read().split("\n")
        self.assertEqual(lines[0], "frame,label")
        self.assertEqual(lines[-1], "")
        self.assertEqual(len(lines), 199319 + 2)
        frames = np.array([int(line.split(",")[0]) for line in lines[1:-1]])
        np.testing.assert_array_equal(frames, np.arange(199319))
        self.assertTrue(all(line.split(",")[1] in ("H", "A", "D", "X") and line.count(",") == 1 for line in lines[1:-1]))

    def test_graded_frames_are_answered_and_the_rest_are_x(self):
        graded = np.zeros(len(self.codes), bool)
        for start, stop in self.blocks:
            graded[start:stop] = True
        self.assertEqual(int(graded.sum()), 158857)
        self.assertTrue((self.codes[graded] != X).all())
        self.assertTrue((self.codes[~graded] == X).all())

    def test_no_state_change_rests_on_a_missing_ball(self):
        tracking = data.load_tracking()
        feats = features.compute_features(tracking, self.blocks, DEFAULT)
        filled = pm.fill_abstentions(self.codes, self.blocks)
        transitions = pm.transitions(filled, self.blocks)
        self.assertGreater(len(transitions), 100)
        for frame, old, new in transitions:
            if old < 2 and new < 2:                         # a hand-over needs a ball observation within one second
                self.assertTrue(feats.ball_valid[max(0, frame - 25): frame + 26].any())
            self.assertLess(feats.gap_length[frame], 125)   # and no change lands inside a >= 5 s ball gap

    def test_both_halves_decode_independently_on_real_data(self):
        tracking = data.load_tracking()
        both, _, _ = predict.predict_codes(tracking, RC1, "decoder", self.blocks)
        second_only, _, _ = predict.predict_codes(tracking, RC1, "decoder", self.blocks[1:])
        s, e = self.blocks[1]
        np.testing.assert_array_equal(both[s:e], second_only[s:e])

    def test_public_scoring_runs_and_reports_no_abstention(self):
        gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
        result = pm.evaluate(gt, self.codes)
        self.assertEqual(result["n_abstain_on_graded"], 0)
        self.assertGreater(result["spell_f1_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
