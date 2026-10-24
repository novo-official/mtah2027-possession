"""Reading the Automatic Tracking Data (ATD) into plain NumPy arrays, in metres.

Conventions fixed here and relied on everywhere else:

* Row i of every array is **video frame i** of the tactical video. The ATD counts `frame_id` from
  1, so `video_frame = frame_id - 1`; the loader checks the ids are exactly 1..N and refuses
  anything else rather than guessing an alignment.
* Positions are in **metres** with the origin at a pitch corner. The file stores 0-1 fractions;
  they are scaled by the real pitch of `games.GAMES` (105.3 x 68.0), axis by axis, because the
  pitch is not square.
* A missing or implausible observation is **NaN**, never zero. A coordinate that is non-finite or
  further than `coord_margin_m` outside the pitch is a tracker error and also becomes NaN.
* Team 0 is home, team 1 is away. The first `<Team>` of the metadata is home (FIFA/EPTS
  convention, as in starter/baseline.py). Slots are tracklets, not people.

Only the public repository helpers are used for paths and constants (games.py).
"""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402

from .config import DEFAULT  # noqa: E402

HOME, AWAY = 0, 1


def tokens_per_record(n_slots):
    """frame id + (x, y, speed) per slot + (x, y, speed) for the ball."""
    return 1 + 3 * n_slots + 3


@dataclass(frozen=True)
class Tracking:
    """One recording's tracking, aligned so row i is video frame i."""
    player_xy: np.ndarray        # (N, S, 2) float32 metres, NaN = missing/invalid
    player_speed: np.ndarray     # (N, S) float32 native speed, NaN = missing (kept, not trusted)
    ball_xy: np.ndarray          # (N, 2) float32 metres
    slot_team: np.ndarray        # (S,) int8, 0 home / 1 away
    slot_is_goalkeeper: np.ndarray  # (S,) bool
    pitch: tuple                 # (length_m, width_m)
    fps: int

    @property
    def n_frames(self):
        return self.ball_xy.shape[0]

    @property
    def n_slots(self):
        return self.player_xy.shape[1]

    def with_ball_masked(self, mask):
        """Copy with the ball set to missing where `mask` is True (robustness experiments)."""
        ball = self.ball_xy.copy()
        ball[mask] = np.nan
        return Tracking(self.player_xy, self.player_speed, ball, self.slot_team,
                        self.slot_is_goalkeeper, self.pitch, self.fps)


def sanitize(xy_fraction, pitch, margin_m):
    """0-1 pitch fractions -> metres; non-finite or implausibly far outside the pitch -> NaN.

    Both coordinates of a point are invalidated together: half a position is no position.
    """
    length, width = pitch
    out = np.empty(xy_fraction.shape, dtype=np.float32)
    out[..., 0] = xy_fraction[..., 0] * length
    out[..., 1] = xy_fraction[..., 1] * width
    bad = ~np.isfinite(out).all(axis=-1)
    with np.errstate(invalid="ignore"):
        bad |= (out[..., 0] < -margin_m) | (out[..., 0] > length + margin_m)
        bad |= (out[..., 1] < -margin_m) | (out[..., 1] > width + margin_m)
    out[bad] = np.nan
    return out


def parse_metadata(meta_path):
    """(slot_team int8[S], slot_is_goalkeeper bool[S]) from the ATD metadata XML."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    players = list(root.iter("Player"))
    if not players:
        raise ValueError("{}: no Player entries".format(meta_path))
    team = np.array([HOME if p.get("teamId") == team_ids[0] else AWAY for p in players],
                    dtype=np.int8)
    keeper = np.zeros(len(players), dtype=bool)
    for i, player in enumerate(players):
        for parameter in player.iter("ProviderParameter"):
            if (parameter.findtext("Name") == "position_type"
                    and (parameter.findtext("Value") or "").strip().lower() == "goalkeeper"):
                keeper[i] = True
    return team, keeper


def parse_raw(raw_path, n_slots, expected_frames=None):
    """(frame_ids int64[N], players float64[N, S, 3], ball float64[N, 3]) straight from the text.

    Every line is `<id>:<x,y,s>;<x,y,s>;...:<x,y,s>`; NaN tokens parse as NaN. The whole file is
    parsed in one vectorised pass and the token count is checked, so a malformed line fails
    loudly instead of shifting every later column.
    """
    with open(raw_path, "rb") as handle:
        text = handle.read()
    n_lines = text.count(b"\n") + (0 if text.endswith(b"\n") or not text else 1)
    flat = text.replace(b"\n", b",").replace(b";", b",").replace(b":", b",")
    values = np.fromstring(flat, sep=",", dtype=np.float64)  # noqa: NPY201 (text mode, not deprecated)
    per = tokens_per_record(n_slots)
    if values.size != n_lines * per:
        raise ValueError("{}: {} tokens for {} lines, expected {} per line".format(
            raw_path, values.size, n_lines, per))
    table = values.reshape(n_lines, per)
    if expected_frames is not None and n_lines != expected_frames:
        raise ValueError("{}: {} records, expected {}".format(raw_path, n_lines, expected_frames))
    frame_ids = table[:, 0].astype(np.int64)
    if not np.array_equal(frame_ids, np.arange(1, n_lines + 1)):
        raise ValueError("{}: frame ids are not exactly 1..{} in order".format(raw_path, n_lines))
    players = table[:, 1:1 + 3 * n_slots].reshape(n_lines, n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:]
    return frame_ids, players, ball


def build_tracking(frame_players, frame_ball, slot_team, slot_keeper, pitch, fps, margin_m):
    """Tracking from raw 0-1 arrays (N,S,3) and (N,3), applying unit conversion and sanitising."""
    return Tracking(
        player_xy=sanitize(frame_players[..., :2], pitch, margin_m),
        player_speed=frame_players[..., 2].astype(np.float32),
        ball_xy=sanitize(frame_ball[..., :2], pitch, margin_m),
        slot_team=slot_team,
        slot_is_goalkeeper=slot_keeper,
        pitch=tuple(pitch),
        fps=fps,
    )


def load_tracking(game=games.GAME, data_dir=None, source="tactical", config=DEFAULT):
    """The tactical (default) or broadcast ATD as a Tracking. Paths come from games.asset_paths."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, source)
    slot_team, slot_keeper = parse_metadata(meta_path)
    _, players, ball = parse_raw(raw_path, len(slot_team), games.source_frames(source, game))
    return build_tracking(players, ball, slot_team, slot_keeper, games.GAMES[game]["pitch"],
                          config.fps, config.coord_margin_m)


def half_blocks(game=games.GAME):
    """[(half, first_video_frame, stop_video_frame)] with `stop` exclusive.

    The published graded halves, from games.half_video_blocks (inclusive ends there). Decoding
    is done inside these blocks independently; nothing is carried across the gap between them.
    """
    return [(half, first, last + 1) for half, first, last in games.half_video_blocks(game)]
