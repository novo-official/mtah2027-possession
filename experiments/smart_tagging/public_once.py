"""The ONE pre-registered public-window evaluation of the Smart Tagging candidate (Phase 4.9).

    .venv/bin/python -m experiments.smart_tagging.public_once

Scores the already-generated candidate CSV and the frozen RC1 CSV with the official scorer. A lock file records
that the evaluation has been done; the script refuses to run again (no sweeps, no re-tuning after seeing a score).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.rc1_selection.compare import fmt_public  # noqa: E402
from experiments.rc1_selection.diagnostics import pm, public_gt, public_metrics  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "rc2_tag_experimental")
LOCK = os.path.join(OUT, "public_eval.done")


def main():
    if os.path.exists(LOCK):
        sys.exit("refusing: the single public-window evaluation of this candidate has already been run (see {})".format(LOCK))
    gt = public_gt()
    rows = {}
    for name, path in (("RC1", os.path.join(ROOT, "outputs", "rc1", "possession_rc1.csv")),
                       ("rc2_tag_experimental", os.path.join(OUT, "possession.csv"))):
        rows[name] = public_metrics(gt, pm.label_codes(pm.read_label_csv(path)), 1000)
        print(fmt_public(name, rows[name]))
    with open(LOCK, "w") as handle:
        handle.write("done\n")


if __name__ == "__main__":
    main()
