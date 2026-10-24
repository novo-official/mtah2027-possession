"""Phase 4.8 tests: the model registry and the historical-preset regression pins.

The fingerprints below freeze every named configuration. Editing a historical preset (or any
`Config` default it inherits) changes its fingerprint and fails here, instead of silently redefining
an old result. The real-data tests re-derive the recorded SHA-256 of every preset's full-match
output; they are skipped when the restricted tracking is not installed and print nothing but
pass / fail.

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

from solution import data, model_registry, predict  # noqa: E402
from solution.config import DEFAULT, PRESETS  # noqa: E402
from solution.duration import check_transition_safety  # noqa: E402

FINGERPRINTS = {
    "rc1": "1e2dbb95abbe735a2e307e263c38828b36dca2b5eb51372be3a5a96570d728b0",
    "legacy_p42": "5e37a555bb30b179dae3f2af32f432eb3a340beb4f1904fde08469b0a830f5fe",
    "legacy_p43": "637526c56ac83785ccce20b87e0081ed1dee702745f1cc0b90bd7b7e6e0ce551",
    "corrected": "1e2dbb95abbe735a2e307e263c38828b36dca2b5eb51372be3a5a96570d728b0",
    "corrected_dead": "da110422489c8bede96fdbc7e6869bba84b072664d0a29cc37fbcf7bfbaee6f9",
    "corrected_dead_hysteresis": "c80cf24d56509f13a33ca7e3198fb7f2d53a6ce5d425601c3533177825f32ebc",
    "duration_only": "13e72405310a2f8555419ee9c8dd0f41569e272bcb379c42a390f72a85d7222d",
    "combined": "2c04386f203efd301d04fd528c50060394c2a5d80e3c25598bb988567566e7e4",
    "oa_only": "f4de99b00aa7456a0d11dd1d853863f86c6b7e71b032f2720069b6662f5ed22c",
    "oa_restart": "dea8f1e2f221852341776d789b66436fae44ac4773c912555e0a62e1dd5e3fe0",
    "oa_entry": "4b74e8f0cdb054bd925434984de8940e0f7200d1f0678d88fae7cec9812f4930",
    "oa_full": "f621b6af862cb468c8a9e2ae50fd42d65cd9a2142c229390484a2056b8374198",
    "oa_mixture_null_rejected": "11402497a9cf3cf9a9192941b73fa01e98d2029565f137d7417f0ae8bb64331f",
    # Phase 4.9: same Config values as rc1 (the tag prior lives in smart_tagging.TagConfig, pinned in test_smart_tagging.py)
    "rc2_tag_experimental": "1e2dbb95abbe735a2e307e263c38828b36dca2b5eb51372be3a5a96570d728b0",
}
STATUSES = {"release-candidate", "reference", "experimental", "rejected"}


class TestRegistryStructure(unittest.TestCase):
    def test_every_preset_is_registered_and_every_registration_is_a_preset(self):
        self.assertEqual(set(PRESETS), set(model_registry.REGISTRY))

    def test_fingerprints_are_pinned_for_every_preset(self):
        self.assertEqual(set(FINGERPRINTS), set(PRESETS))
        for name, expected in FINGERPRINTS.items():
            self.assertEqual(model_registry.get(name).fingerprint, expected, name)

    def test_fingerprint_detects_any_single_field_change(self):
        base = model_registry.config_fingerprint(PRESETS["rc1"])
        self.assertNotEqual(base, model_registry.config_fingerprint(PRESETS["rc1"].replace(switch_cost=12.5)))
        self.assertNotEqual(base, model_registry.config_fingerprint(PRESETS["rc1"].replace(decoder_late_ties=False)))
        self.assertEqual(base, model_registry.config_fingerprint(PRESETS["rc1"].replace()))

    def test_exactly_one_release_candidate_and_it_is_rc1(self):
        self.assertEqual(model_registry.names("release-candidate"), ["rc1"])
        self.assertEqual(model_registry.RELEASE_CANDIDATE, "rc1")
        self.assertNotIn("rc1", ("legacy_p42", "legacy_p43", "corrected", "corrected_dead"))

    def test_every_entry_is_documented(self):
        for name, spec in model_registry.REGISTRY.items():
            self.assertIn(spec.status, STATUSES, name)
            self.assertTrue(spec.description and spec.enabled and spec.disabled is not None and spec.limitations, name)
            self.assertIn("--preset " + name, spec.reproduce)
            text = model_registry.describe(name)
            self.assertIn(spec.fingerprint, text)
        with self.assertRaises(KeyError):
            model_registry.get("no_such_preset")

    def test_recorded_reference_hashes_are_well_formed_and_unique_per_distinct_output(self):
        refs = {n: s.reference_sha256 for n, s in model_registry.REGISTRY.items() if s.reference_sha256}
        for name, digest in refs.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$", name)
        # presets that are known to be output-identical share a hash; nothing else may
        by_hash = {}
        for name, digest in refs.items():
            by_hash.setdefault(digest, []).append(name)
        for names in by_hash.values():
            self.assertLessEqual(len(names), 2)
            if len(names) == 2:
                self.assertEqual(set(names), {"rc1", "corrected"})

    def test_historical_presets_keep_their_definitions(self):
        self.assertEqual(PRESETS["legacy_p42"], DEFAULT)
        self.assertEqual(PRESETS["legacy_p43"], DEFAULT.replace(use_dead_ball_module=True))
        self.assertEqual(PRESETS["corrected"], DEFAULT.replace(decoder_late_ties=True))
        self.assertEqual(PRESETS["corrected_dead"], DEFAULT.replace(decoder_late_ties=True, use_dead_ball_module=True))
        self.assertEqual(PRESETS["corrected_dead_hysteresis"], PRESETS["corrected_dead"].replace(duration_model="hysteresis"))

    def test_every_registered_preset_keeps_a_direct_switch_cheaper_than_the_dead_detour(self):
        for name in PRESETS:
            self.assertTrue(check_transition_safety(PRESETS[name])["safe"], name)

    def test_official_baselines_are_unchanged(self):
        for name, digest in model_registry.OFFICIAL_BASELINES.items():
            path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
            if not os.path.isfile(path):
                self.skipTest("committed baselines not present")
            self.assertEqual(predict.file_sha256(path), digest, name)
        self.assertEqual(model_registry.FALLBACK_BASELINE["sha256"], model_registry.OFFICIAL_BASELINES["baseline_smooth50"])


REAL_RAW = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")


@unittest.skipUnless(os.path.isfile(REAL_RAW), "tactical ATD not installed")
class TestHistoricalReproduction(unittest.TestCase):
    """Every recorded preset output is regenerated byte for byte."""

    @classmethod
    def setUpClass(cls):
        cls.tracking = data.load_tracking()
        cls.blocks = predict.block_ranges()

    def sha(self, config):
        codes, _, _ = predict.predict_codes(self.tracking, config, "decoder", self.blocks)
        with tempfile.TemporaryDirectory() as tmp:
            return predict.file_sha256(predict.write_submission(os.path.join(tmp, "x.csv"), codes))

    def test_every_recorded_reference_output_is_reproduced_exactly(self):
        for name, spec in model_registry.REGISTRY.items():
            if spec.reference_sha256 is None:
                continue
            with self.subTest(preset=name):
                self.assertEqual(self.sha(spec.config), spec.reference_sha256)

    def test_unrecorded_preset_is_at_least_deterministic(self):
        config = PRESETS["oa_mixture_null_rejected"]
        self.assertEqual(self.sha(config), self.sha(config))


if __name__ == "__main__":
    unittest.main()
