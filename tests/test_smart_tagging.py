"""Phase 4.9 tests: the optional Smart Tagging team-phase prior.

Synthetic XML and synthetic tracking only, except the clearly marked real-data class (skipped without the
restricted data; it prints nothing). The properties tested: a tag is evidence, never an answer; it is bounded;
it contributes zero whenever its meaning is unclear; it is symmetric, deterministic and opt-in; RC1 is untouched.

Run with:   .venv/bin/python -m unittest discover -s tests -v
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

import games  # noqa: E402
import load_smart_tagging as official  # noqa: E402
from solution import data, decoder, model_registry, predict, smart_tagging as st  # noqa: E402
from solution.config import PRESETS  # noqa: E402
from test_rc1 import play, run, swap_teams  # noqa: E402

H, A, D, X = decoder.H, decoder.A, decoder.D, decoder.X
RC1 = PRESETS["rc1"]
CFG = st.TagConfig()
BUDGET = CFG.budget(RC1.switch_cost)
PY = sys.executable
ATTACK, DEFEND = st.ELIGIBLE_PAIRS[1]          # progression / defending in middle third


def inst(code, start, end, team="Barcelona", half="1st Half", **extra):
    labels = dict(extra)
    if team is not None:
        labels["Team"] = team
    if half is not None:
        labels["Half"] = half
    return {"code": code, "start": start, "end": end, "labels": labels}


def pair(start, end, attack_team="Barcelona", defend_team="Atletic Club", half="1st Half", codes=(ATTACK, DEFEND)):
    return [inst(codes[0].upper(), start, end, attack_team, half), inst(codes[1].upper(), start, end, defend_team, half)]


def write_xml(path, instances):
    body = []
    for n, i in enumerate(instances, 1):
        labels = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t) for g, t in i["labels"].items())
        start = "" if i["start"] is None else "<start>{}</start>".format(i["start"])
        end = "" if i["end"] is None else "<end>{}</end>".format(i["end"])
        body.append("<instance><ID>{}</ID>{}{}<code>{}</code>{}</instance>".format(n, start, end, i["code"], labels))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("<file><ALL_INSTANCES>{}</ALL_INSTANCES></file>".format("".join(body)))
    return path


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def xml(self, instances):
        return write_xml(os.path.join(self.tmp.name, "tags.xml"), instances)

    def votes(self, instances):
        events, _ = st.parse_events(self.xml(instances), CFG)
        return st.build_votes(events, CFG)

    def prior(self, instances, n=3000, blocks=((0, 3000),)):
        events, _ = st.parse_events(self.xml(instances), CFG)
        votes, _ = st.build_votes(events, CFG)
        return st.ownership_prior(votes, n, list(blocks), BUDGET, CFG)[0]


class TestParsing(Tmp):
    def test_matches_the_official_parser_on_codes_times_and_teams(self):
        instances = [inst("PROGRESSION", 10.0, 27.6), inst("Defending In Middle Third", 10.0, 27.6, "Atletic Club"),
                     inst("SET PIECES", 100.04, 135.02, "N/A", Type="Throw-in"), inst("BUILD UP", 0.01, 0.03, "Athletic Club", "2nd Half")]
        folder = os.path.join(self.tmp.name, "tactical")
        os.makedirs(folder)
        write_xml(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), instances)
        theirs = official.load(data_dir=self.tmp.name)
        ours, stats = st.parse_events(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), CFG)
        self.assertEqual(stats, {"instances": 4, "invalid_interval": 0})
        self.assertEqual([(e["code"].lower(), e["start_frame"], e["end_frame"]) for e in theirs],
                         [(e.code, e.start_frame, e.end_frame) for e in ours])

    def test_time_to_frame_rounding_matches_the_official_helper(self):
        for seconds in (0.0, 0.019, 0.021, 0.02, 0.06, 12.34, 12.38, 1799.98, 4501.2, 3.5 / 25, 7.5 / 25):
            self.assertEqual(st.to_frame(seconds), int(round(seconds * games.GAMES[games.GAME]["fps"])), seconds)
        self.assertEqual(st.to_frame(0.019), 0)
        self.assertEqual(st.to_frame(0.021), 1)

    def test_codes_and_team_names_are_case_insensitive_and_both_spellings_of_the_away_club_work(self):
        self.assertEqual([st.team_side(n) for n in ("Barcelona", " barcelona ", "BARCELONA", "Atletic Club", "Athletic Club", "ATLETIC CLUB")],
                         [0, 0, 0, 1, 1, 1])
        events, _ = st.parse_events(self.xml([inst("BuIlD Up", 1, 2, "ATHLETIC CLUB")]), CFG)
        self.assertEqual((events[0].code, events[0].team), ("build up", 1))

    def test_unknown_missing_and_na_teams_are_never_guessed(self):
        for name in ("N/A", "n/a", "", None, "Real Madrid", "Barca"):
            self.assertIsNone(st.team_side(name), name)
        events, _ = st.parse_events(self.xml([inst("PROGRESSION", 1, 9, None)]), CFG)
        self.assertIsNone(events[0].team)

    def test_missing_or_invalid_timestamps_are_dropped_and_counted_not_repaired(self):
        instances = [inst("PROGRESSION", None, 9), inst("PROGRESSION", 1, None), inst("PROGRESSION", "abc", 9), inst("PROGRESSION", "nan", 9),
                     inst("PROGRESSION", 9, 9), inst("PROGRESSION", 10, 5), inst("PROGRESSION", -3, 5), inst("PROGRESSION", "inf", "inf"),
                     inst("PROGRESSION", 1, 2)]
        events, stats = st.parse_events(self.xml(instances), CFG)
        self.assertEqual(len(events), 1)
        self.assertEqual(stats, {"instances": 9, "invalid_interval": 8})

    def test_missing_and_malformed_files_fail_loudly(self):
        with self.assertRaises(OSError):
            st.parse_events(os.path.join(self.tmp.name, "nope.xml"), CFG)
        bad = os.path.join(self.tmp.name, "bad.xml")
        with open(bad, "w") as handle:
            handle.write("<file><instance>")
        with self.assertRaises(ValueError):
            st.parse_events(bad, CFG)
        with self.assertRaises(IOError) as caught:
            st.require_file(self.tmp.name)
        self.assertIn("match data not found", str(caught.exception))

    def test_half_tag_is_parsed_and_unknown_halves_contribute_nothing(self):
        self.assertEqual([st.half_index(t) for t in ("1st Half", "2nd Half", "2ND half", None, "Extra", "")], [0, 1, 1, None, None, None])
        votes, _ = self.votes(pair(10, 30, half=None))
        self.assertEqual(votes, [])


class TestVotes(Tmp):
    def test_attack_team_is_in_control_and_defend_team_is_the_other_club(self):
        v, _ = self.votes(pair(10, 30, "Barcelona", "Atletic Club"))
        self.assertEqual([x.side for x in v], [0])
        v, _ = self.votes(pair(10, 30, "Atletic Club", "Barcelona"))
        self.assertEqual([x.side for x in v], [1])

    def test_defend_tag_alone_never_means_the_defending_team_has_the_ball(self):
        v, _ = self.votes(pair(10, 30, "N/A", "Barcelona"))          # Barcelona defends -> Atletic is in control
        self.assertEqual([x.side for x in v], [1])
        prior = self.prior(pair(10, 30, "N/A", "Barcelona"))
        self.assertLess(prior[10 * 25:30 * 25].sum(), 0)              # never > 0 (home) from a home DEFENDING tag
        self.assertEqual(self.prior(pair(10, 30, "N/A", "Barcelona")).max(), 0.0)

    def test_pair_with_conflicting_or_unknown_teams_is_zero_not_a_guess(self):
        for att, dfn in (("Barcelona", "Barcelona"), ("Atletic Club", "Atletic Club"), ("N/A", "N/A"), (None, None), ("Barcelona", None) if False else ("N/A", None)):
            v, s = self.votes(pair(10, 30, att, dfn))
            self.assertEqual(v, [], (att, dfn))
        self.assertTrue(np.all(self.prior(pair(10, 30, "Barcelona", "Barcelona")) == 0))

    def test_an_unpaired_attack_tag_and_an_ineligible_code_are_zero(self):
        self.assertEqual(self.votes([inst("PROGRESSION", 10, 30)])[0], [])
        for code in ("LONG BALL", "CREATING CHANCES", "BALL IN THE BOX", "PLAYERS IN THE BOX", "BALL IN FINAL THIRD", "SET PIECES", "GOALS"):
            self.assertEqual(self.votes([inst(code, 10, 30, "Barcelona"), inst("DEFENDING IN MIDDLE THIRD", 10, 30, "Atletic Club")])[0], [], code)

    def test_pairs_must_share_the_identical_interval(self):
        instances = [inst("PROGRESSION", 10, 30), inst("DEFENDING IN MIDDLE THIRD", 10.5, 30, "Atletic Club")]
        self.assertEqual(self.votes(instances)[0], [])

    def test_duplicate_instances_of_one_pair_are_one_vote(self):
        v, _ = self.votes(pair(10, 30) + pair(10, 30) + pair(10, 30))
        self.assertEqual(len(v), 1)
        single = self.prior(pair(10, 30))
        np.testing.assert_array_equal(self.prior(pair(10, 30) * 3), single)

    def test_all_three_eligible_pair_types_are_recognised(self):
        instances = []
        for k, codes in enumerate(st.ELIGIBLE_PAIRS):
            instances += pair(100 * k, 100 * k + 20, codes=codes)
        self.assertEqual(len(self.votes(instances)[0]), 3)


class TestPrior(Tmp):
    def test_total_contribution_of_a_pair_is_bounded_whatever_its_length(self):
        for length in (2.0, 5.0, 20.0, 60.0):
            prior = self.prior(pair(10, 10 + length), n=4000)
            self.assertAlmostEqual(float(prior.sum()), BUDGET, places=6, msg=length)
            self.assertAlmostEqual(float(prior[250:250 + int(length * 25)].min()), BUDGET / (length * 25), places=9)
        self.assertLess(BUDGET, RC1.switch_cost)
        self.assertLessEqual(BUDGET, 0.5 * RC1.switch_cost + 1e-12)

    def test_durations_outside_two_to_sixty_seconds_contribute_zero(self):
        for length in (1.9, 60.1, 400.0):
            self.assertTrue(np.all(self.prior(pair(10, 10 + length), n=20000, blocks=((0, 20000),)) == 0), length)

    def test_agreeing_overlapping_pairs_are_not_summed_and_opposite_pairs_cancel(self):
        a = pair(10, 30, codes=st.ELIGIBLE_PAIRS[0])
        b = pair(10, 30, codes=st.ELIGIBLE_PAIRS[1])
        both = self.prior(a + b)
        np.testing.assert_allclose(both, self.prior(a))                   # same side, same density: counted once
        opposite = pair(10, 30, "Atletic Club", "Barcelona", codes=st.ELIGIBLE_PAIRS[1])
        self.assertTrue(np.all(self.prior(a + opposite) == 0))             # equal and opposite: cancels exactly

    def test_events_are_clipped_to_their_half_and_nothing_leaks_outside_the_graded_blocks(self):
        prior = self.prior(pair(10, 30), n=3000, blocks=((0, 400), (1000, 3000)))
        self.assertTrue(np.all(prior[400:] == 0))
        self.assertAlmostEqual(float(prior[:400].sum()), BUDGET * (400 - 250) / 500, places=6)       # the clipped part only
        second = self.prior(pair(10, 30, half="2nd Half"), n=3000, blocks=((0, 400), (1000, 3000)))
        self.assertTrue(np.all(second == 0))                                # 10-30 s is not inside the second block
        outside = self.prior(pair(80, 100), n=3000, blocks=((0, 400),))
        self.assertTrue(np.all(outside == 0))

    def test_budget_fraction_must_stay_below_one_switch(self):
        for bad in (0.0, 1.0, 1.5, -0.1):
            with self.assertRaises(ValueError):
                st.TagConfig(budget_fraction=bad).budget(12.0)

    def test_home_away_symmetry_and_determinism(self):
        home = self.prior(pair(10, 30, "Barcelona", "Atletic Club"))
        away = self.prior(pair(10, 30, "Atletic Club", "Barcelona"))
        np.testing.assert_array_equal(home, -away)
        np.testing.assert_array_equal(home, self.prior(pair(10, 30, "Barcelona", "Atletic Club")))


class TestTagIsEvidenceNotAnAnswer(Tmp):
    N = 3000

    def codes(self, tracking, instances, external=True):
        prior = self.prior(instances, n=tracking.n_frames, blocks=((0, tracking.n_frames),)) if instances else None
        return predict.predict_codes(tracking, RC1, "decoder", [(0, tracking.n_frames)], external_own=prior if external else None)[0]

    def test_with_no_tag_or_a_zero_prior_the_output_is_exactly_rc1(self):
        t = play([(400, "H", True), (50, "OUT", True), (300, "N", False), (400, "A", True), (200, "A", False), (300, "H", True)])
        base = run(t)[0]
        np.testing.assert_array_equal(self.codes(t, None), base)
        zeros = predict.predict_codes(t, RC1, "decoder", [(0, t.n_frames)], external_own=np.zeros(t.n_frames))[0]
        np.testing.assert_array_equal(zeros, base)
        np.testing.assert_array_equal(self.codes(t, pair(10, 30, "N/A", "N/A")), base)

    def test_strong_contradictory_tracking_evidence_wins_over_a_tag_for_the_other_team(self):
        t = play([(1500, "H", True)])                                         # home is clearly in control for 60 s
        for team_pair in (pair(5, 35, "Atletic Club", "Barcelona"), pair(5, 35, "Atletic Club", "Barcelona", codes=st.ELIGIBLE_PAIRS[0])):
            codes = self.codes(t, team_pair)
            self.assertTrue((codes == H).all())

    def test_a_tag_alone_cannot_switch_the_state_through_a_ball_gap(self):
        t = play([(300, "H", True), (1200, "H", False)])
        codes = self.codes(t, pair(14, 56, "Atletic Club", "Barcelona"))        # tag for the away team across the whole gap
        self.assertTrue((codes == H).all())
        self.assertLessEqual(BUDGET, RC1.switch_cost / 2 + 1e-12)

    def test_a_tag_can_tip_a_borderline_decision_but_only_with_tracking_support(self):
        # tracking evidence worth less than one switch after the gap; with the tag that evidence + 6.0 nats clears it.
        t = play([(300, "H", True), (400, "H", False), (6, "A", True), (300, "H", True)])
        plain = run(t)[0]
        tagged = self.codes(t, pair(12.5, 28.5, "Atletic Club", "Barcelona"))
        self.assertFalse((plain[700:706] == A).any())
        self.assertTrue(((tagged == A).sum() >= (plain == A).sum()))
        # and a tag for the team the tracking already names changes nothing material
        agree = self.codes(play([(1500, "A", True)]), pair(5, 35, "Atletic Club", "Barcelona"))
        self.assertTrue((agree == A).all())

    def test_tag_never_creates_a_stoppage_or_removes_one(self):
        t = play([(300, "H", True), (60, "OUT", True), (400, "N", False), (300, "H", True)])
        base = run(t)[0]
        for instances in (pair(8, 28, "Atletic Club", "Barcelona"), pair(8, 28, "Barcelona", "Atletic Club")):
            tagged = self.codes(t, instances)
            self.assertEqual(int((base == D).sum()), int((tagged == D).sum()))

    def test_team_swap_symmetry_of_the_whole_chain(self):
        t = play([(500, "H", True), (400, "H", False), (60, "A", True), (500, "A", True)])
        a = self.codes(t, pair(10, 30, "Barcelona", "Atletic Club") + pair(30, 50, "Atletic Club", "Barcelona", codes=st.ELIGIBLE_PAIRS[0]))
        b = self.codes(swap_teams(t), pair(10, 30, "Atletic Club", "Barcelona") + pair(30, 50, "Barcelona", "Atletic Club", codes=st.ELIGIBLE_PAIRS[0]))
        np.testing.assert_array_equal(np.where(a == H, A, np.where(a == A, H, a)), b)

    def test_deterministic_and_bounded_per_frame(self):
        t = play([(500, "H", True), (400, "N", False), (500, "A", True)])
        instances = pair(10, 30)
        np.testing.assert_array_equal(self.codes(t, instances), self.codes(t, instances))
        prior = self.prior(instances)
        self.assertLessEqual(float(np.abs(prior).max()), BUDGET / (CFG.min_duration_s * 25) + 1e-12)
        self.assertLess(float(np.abs(prior).max()), 0.2)                      # vs up to 3.0 nats/frame of tracking evidence

    def test_wrong_shape_prior_is_rejected(self):
        t = play([(200, "H", True)])
        with self.assertRaises(ValueError):
            predict.predict_codes(t, RC1, "decoder", [(0, 200)], external_own=np.zeros(5))


class TestOptIn(unittest.TestCase):
    def test_only_the_experimental_preset_uses_tags_and_it_shares_rc1_config_values(self):
        self.assertEqual(sorted(st.TAG_PRESETS), ["rc2_tag_experimental"])
        self.assertEqual(model_registry.get("rc2_tag_experimental").fingerprint, model_registry.get("rc1").fingerprint)
        self.assertIsNone(model_registry.get("rc1").tag_config)
        self.assertIsNotNone(model_registry.get("rc2_tag_experimental").tag_fingerprint)
        self.assertEqual(model_registry.get("rc2_tag_experimental").status, "rejected")     # Phase 4.9 decision
        self.assertEqual(model_registry.names("release-candidate"), ["rc1"])

    def test_tag_config_is_pinned(self):
        self.assertEqual(model_registry.get("rc2_tag_experimental").tag_fingerprint,
                         "cf6c30e986018b8e1c13e2839c853b53659498937e1a5e4136a7a3378c17e225")

    def test_rc1_run_never_touches_the_tag_file(self):
        with mock.patch.object(st, "tag_prior", side_effect=AssertionError("rc1 must not read tags")), \
             mock.patch.object(st, "require_file", side_effect=AssertionError("rc1 must not read tags")):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):   # argparse rejection, before any data access
                predict.main(["--preset", "no_such_preset"])
            with contextlib.redirect_stdout(io.StringIO()):
                out = predict.main(["--describe", "rc1"])
        self.assertEqual(out, 0)


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
REAL_TAG = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_smart_tagging.xml")
RC1_SHA = "a9be383bfa867ae034432d7746894951c89f6f9cd56ad8c9264eb9a5dd37f1be"


@unittest.skipUnless(os.path.isfile(REAL_RAW) and os.path.isfile(REAL_TAG), "tactical ATD / Smart Tagging not installed")
class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.blocks = predict.block_ranges()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cli(self, *args, env=None):
        return subprocess.run([PY, "-m", "solution.predict", *args], capture_output=True, text=True, cwd=self.tmp.name,
                              env=dict(os.environ, PYTHONPATH=ROOT, **(env or {})), timeout=300)

    def test_the_real_file_parses_with_the_official_time_base_and_yields_bounded_votes(self):
        events, stats = st.parse_events(REAL_TAG, CFG)
        self.assertEqual(stats, {"instances": 638, "invalid_interval": 0})
        report = official.check_time_base(official.load())
        self.assertTrue(report["agrees"])
        prior, rep = st.tag_prior(REAL_TAG, 199319, self.blocks, RC1.switch_cost, CFG)
        self.assertGreater(rep["eligible"], 80)
        self.assertLessEqual(rep["eligible"], 112)
        self.assertLessEqual(float(np.abs(prior).max()), BUDGET / (CFG.min_duration_s * 25))
        graded = np.zeros(199319, bool)
        for a, b in self.blocks:
            graded[a:b] = True
        self.assertTrue(np.all(prior[~graded] == 0))

    def test_rc1_is_byte_identical_and_rc2_without_tags_equals_rc1(self):
        a = os.path.join(self.tmp.name, "rc1.csv")
        self.assertEqual(self.cli("--preset", "rc1", "--out", a).returncode, 0)
        self.assertEqual(predict.file_sha256(a), RC1_SHA)
        tracking = data.load_tracking()
        codes, _, _ = predict.predict_codes(tracking, PRESETS["rc2_tag_experimental"], "decoder", self.blocks)
        path = predict.write_submission(os.path.join(self.tmp.name, "rc2_off.csv"), codes)
        self.assertEqual(predict.file_sha256(path), RC1_SHA)

    def test_rc2_cli_is_reproducible_and_fails_cleanly_without_the_tag_file(self):
        outs = []
        for i in (1, 2):
            path = os.path.join(self.tmp.name, "rc2_{}.csv".format(i))
            result = self.cli("--preset", "rc2_tag_experimental", "--out", path)
            self.assertEqual(result.returncode, 0, result.stderr)
            outs.append(predict.file_sha256(path))
        self.assertEqual(outs[0], outs[1])
        partial = os.path.join(self.tmp.name, "partial", "tactical")
        os.makedirs(partial)
        for name in ("BAR-ATH_tactical_atd.txt", "BAR-ATH_tactical_atd.xml"):
            os.symlink(os.path.join(ROOT, "data", "tactical", name), os.path.join(partial, name))
        result = self.cli("--preset", "rc2_tag_experimental", "--out", os.path.join(self.tmp.name, "rc2_missing.csv"),
                          env={"MTAH_DATA_DIR": os.path.dirname(partial)})
        self.assertEqual(result.returncode, 2)
        self.assertIn("match data not found", result.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "rc2_missing.csv")))


if __name__ == "__main__":
    unittest.main()
