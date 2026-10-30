"""
utils.py

Reusable helper functions for the CSGO Engagement Dataset Builder.

These functions are shared between the engagement extractor and
feature extractor.
"""

from __future__ import annotations

from bisect import bisect_left
from math import sqrt
from typing import Any


# ============================================================
# Distance Functions
# ============================================================

def distance_2d(
    x1: float,
    y1: float,
    x2: float,
    y2: float
) -> float:
    """
    Calculates horizontal (XY) distance.
    """

    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def distance_3d(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float
) -> float:
    """
    Calculates full 3D Euclidean distance.
    """

    return sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2 +
        (z1 - z2) ** 2
    )


def calculate_distance(
    player1: dict,
    player2: dict
) -> float:
    """
    Calculates the 3D distance between two players.
    """

    return distance_3d(
        player1["x"],
        player1["y"],
        player1["z"],
        player2["x"],
        player2["y"],
        player2["z"]
    )


# ============================================================
# Angle Functions
# ============================================================

def normalize_angle(angle: float) -> float:
    """
    Normalizes an angle to the range [-180, 180].
    """

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def angle_difference(
    angle1: float,
    angle2: float
) -> float:
    """
    Returns the smallest difference between two angles.
    """

    return abs(normalize_angle(angle1 - angle2))


# ============================================================
# Frame Lookup
# ============================================================

def get_frame_at_tick(
    round_data: dict,
    tick: int
) -> dict | None:
    """
    Returns the first frame at or after a given tick.
    """

    for frame in round_data["frames"]:

        if frame["tick"] >= tick:
            return frame

    return None


def get_previous_frame(
    frames: list[dict],
    tick: int,
    frame_offset: int = 1
) -> dict | None:
    """
    Returns the frame a specified number of frames before a tick.

    Parameters
    ----------
    frames
        List of round frames.

    tick
        Engagement start tick.

    frame_offset
        Number of frames before the engagement.

    Returns
    -------
    dict | None
    """

    if not frames:
        return None

    index = None

    for i, frame in enumerate(frames):

        if frame["tick"] >= tick:
            index = i
            break

    if index is None:
        return frames[-1]

    target = max(0, index - frame_offset)

    return frames[target]


def get_frame_index(
    round_data: dict,
    tick: int
) -> int:
    """
    Returns the frame index closest to a tick.
    """

    frames = round_data["frames"]

    ticks = [frame["tick"] for frame in frames]

    index = bisect_left(ticks, tick)

    if index >= len(frames):
        return len(frames) - 1

    return index


def get_round_end_time(
    round_data: dict
) -> float:
    """
    Returns the timestamp of the final frame in the round.
    """

    frames = round_data.get("frames", [])

    if not frames:
        return 0.0

    return frames[-1]["seconds"]


# ============================================================
# Player Lookup
# ============================================================

def find_player(
    frame: dict,
    steam_id: int
) -> dict | None:
    """
    Finds a player in a frame using SteamID.
    """

    if frame is None:
        return None

    for player in frame["t"]["players"]:

        if player["steamID"] == steam_id:
            return player

    for player in frame["ct"]["players"]:

        if player["steamID"] == steam_id:
            return player

    return None


def player_alive(
    frame: dict,
    steam_id: int
) -> bool:
    """
    Returns whether the player is alive.
    """

    player = find_player(frame, steam_id)

    return player is not None and player["isAlive"]


def player_side(
    frame: dict,
    steam_id: int
) -> str | None:
    """
    Returns the player's side.
    """

    player = find_player(frame, steam_id)

    if player is None:
        return None

    return player["side"]


# ============================================================
# Team Helpers
# ============================================================

def get_teammates(
    frame: dict,
    steam_id: int
) -> list[dict]:
    """
    Returns all teammates.
    """

    player = find_player(frame, steam_id)

    if player is None:
        return []

    if player["side"] == "T":
        return frame["t"]["players"]

    return frame["ct"]["players"]


def get_opponents(
    frame: dict,
    steam_id: int
) -> list[dict]:
    """
    Returns all opponents.
    """

    player = find_player(frame, steam_id)

    if player is None:
        return []

    if player["side"] == "T":
        return frame["ct"]["players"]

    return frame["t"]["players"]


def alive_teammates(
    frame: dict,
    steam_id: int
) -> list[dict]:
    """
    Returns living teammates.
    """

    return [
        p
        for p in get_teammates(frame, steam_id)
        if p["isAlive"]
    ]


def alive_opponents(
    frame: dict,
    steam_id: int
) -> list[dict]:
    """
    Returns living opponents.
    """

    return [
        p
        for p in get_opponents(frame, steam_id)
        if p["isAlive"]
    ]


# ============================================================
# Engagement Helpers
# ============================================================

def engagement_duration(
    engagement: dict
) -> float:
    """
    Returns engagement duration.
    """

    return engagement["endTime"] - engagement["startTime"]


def engagement_event_count(
    engagement: dict
) -> int:
    """
    Returns number of events.
    """

    return len(engagement["events"])


# ============================================================
# Geometry Helpers
# ============================================================

def player_distance(
    player1: dict,
    player2: dict
) -> float:
    """
    Returns 3D distance between two players.
    """

    return calculate_distance(player1, player2)


def player_horizontal_distance(
    player1: dict,
    player2: dict
) -> float:
    """
    Returns horizontal distance between two players.
    """

    return distance_2d(
        player1["x"],
        player1["y"],
        player2["x"],
        player2["y"]
    )


# ============================================================
# Demo Helpers
# ============================================================

def iter_rounds(
    demo: dict
):
    """
    Generator over demo rounds.
    """

    yield from demo["gameRounds"]


def valid_round(
    round_data: dict
) -> bool:
    """
    Returns True if the round should be processed.
    """

    if round_data.get("isWarmup", False):
        return False

    if not round_data.get("frames"):
        return False

    return True