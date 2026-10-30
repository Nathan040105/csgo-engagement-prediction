"""
nav_data.py

Loads parsed CS:GO nav mesh JSON (produced by parser.py, the fixed
pynav script) and exposes lookup functions for use as engagement
features: area identity, named callout region, walkable graph
distance between two positions, area "openness" (connectivity
degree), and proximity to hiding-spot/cover data.

Nav mesh geometry is static per map, so each map's parsed JSON is
loaded and cached once per process, not once per row.

Coordinate lookups use a brute-force scan over all areas in a map
(~1,000-2,000 areas depending on map). This is intentionally simple
rather than spatially indexed: at dataset sizes in the low thousands
of rows, a KD-tree or grid index would save milliseconds that don't
matter, at the cost of real complexity that does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from config import NAV_DATA_DIR, HIDING_SPOT_PROXIMITY_UNITS

# ==========================================================
# Hiding spot flag bits
#
# CAUTION: these bit -> name mappings are the commonly cited
# community convention (Source engine bot navigation code), but
# could not be independently confirmed against an authoritative
# Valve source during this project. Treat hiding_spot_flags (the
# raw integer) as the defensible feature; treat the decoded named
# booleans below as a best-effort convenience that should be
# verified against Valve's Nav Mesh Editing documentation
# (developer.valvesoftware.com/wiki/Nav_Mesh_Editing) before being
# cited as fact in the final report.
# ==========================================================

HIDING_SPOT_FLAG_IN_COVER = 0x01
HIDING_SPOT_FLAG_GOOD_SNIPER_SPOT = 0x02
HIDING_SPOT_FLAG_IDEAL_SNIPER_SPOT = 0x04
HIDING_SPOT_FLAG_EXPOSED = 0x08


# ==========================================================
# Map Cache
# ==========================================================

_MAP_CACHE: dict[str, dict[str, Any]] = {}


def _load_map(map_name: str) -> dict[str, Any] | None:
    """
    Loads and caches a map's parsed nav JSON.

    Builds two lookup structures on first load:
        areas       : area_id -> area dict (flattened out of
                      the JSON's place-nested structure)
        area_place  : area_id -> place name string

    Returns None if the parsed JSON file does not exist for this
    map, rather than raising, so callers can fall back to sentinel
    values the same way the rest of the pipeline already does for
    missing weapon-dict entries.
    """

    if map_name in _MAP_CACHE:
        return _MAP_CACHE[map_name]

    json_path = Path(NAV_DATA_DIR) / "parsed" / f"{map_name}.json"

    if not json_path.exists():
        return None

    with open(json_path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    areas: dict[int, dict[str, Any]] = {}
    area_place: dict[int, str] = {}

    for place in raw["places"]:
        for area in place["areas"]:
            areas[area["id"]] = area
            area_place[area["id"]] = place["name"]

    loaded = {
        "areas": areas,
        "area_place": area_place,
    }

    _MAP_CACHE[map_name] = loaded

    return loaded


# ==========================================================
# Geometry Helpers
# ==========================================================


def _area_bounds(area: dict[str, Any]) -> tuple[float, float, float, float]:
    """
    Returns (min_x, max_x, min_y, max_y) for an area, normalizing
    north_west/south_east into explicit min/max rather than
    assuming a fixed corner ordering.
    """

    nw = area["north_west"]
    se = area["south_east"]

    min_x = min(nw["x"], se["x"])
    max_x = max(nw["x"], se["x"])
    min_y = min(nw["y"], se["y"])
    max_y = max(nw["y"], se["y"])

    return min_x, max_x, min_y, max_y


def _area_avg_z(area: dict[str, Any]) -> float:
    """
    Returns the average of the area's four corner heights, used to
    disambiguate vertically stacked areas that overlap in x/y
    (stairs, catwalks, balconies over tunnels, etc.).
    """

    return (
        area["north_west"]["z"]
        + area["south_east"]["z"]
        + area["north_east_z"]
        + area["south_west_z"]
    ) / 4.0


def _area_centroid(area: dict[str, Any]) -> tuple[float, float, float]:
    """
    Returns the (x, y, z) centroid of an area, used as the vertex
    position for graph-distance calculations.
    """

    min_x, max_x, min_y, max_y = _area_bounds(area)

    return (
        (min_x + max_x) / 2.0,
        (min_y + max_y) / 2.0,
        _area_avg_z(area),
    )


def _distance_3d(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:

    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + (p1[2] - p2[2]) ** 2
    )


# ==========================================================
# Area Lookup
# ==========================================================


def get_area(
    map_name: str,
    x: float,
    y: float,
    z: float,
) -> dict[str, Any] | None:
    """
    Finds the nav area containing a given position.

    Checks 2D (x, y) bounding-box containment against every area in
    the map. If more than one area contains the point (vertically
    stacked geometry, e.g. a catwalk over a tunnel), the area whose
    average height is closest to z wins.

    If no area's 2D bounds contain the point at all (e.g. the
    player is airborne slightly outside the nearest walkable
    footprint, or right at a boundary edge), falls back to the
    area with the closest centroid, so a feature is still produced
    rather than silently dropped - matches this pipeline's existing
    convention of degrading gracefully rather than returning
    nothing.

    Returns None only if the map itself has no parsed nav data.
    """

    nav = _load_map(map_name)

    if nav is None:
        return None

    candidates = []

    for area in nav["areas"].values():

        min_x, max_x, min_y, max_y = _area_bounds(area)

        if min_x <= x <= max_x and min_y <= y <= max_y:
            candidates.append(area)

    if candidates:

        if len(candidates) == 1:
            return candidates[0]

        return min(
            candidates,
            key=lambda a: abs(_area_avg_z(a) - z),
        )

    # Fallback: no bounding box contains the point - use nearest
    # centroid instead of returning nothing.
    return min(
        nav["areas"].values(),
        key=lambda a: _distance_3d(_area_centroid(a), (x, y, z)),
    )


def get_area_id(map_name: str, x: float, y: float, z: float) -> int:
    """
    Convenience wrapper returning just the area id.

    Returns -1 if the map has no parsed nav data, matching this
    pipeline's existing sentinel convention for unavailable
    features (see attacker_magazine_pct / attacker_reserve_pct).
    """

    area = get_area(map_name, x, y, z)

    if area is None:
        return -1

    return area["id"]


def get_callout(map_name: str, x: float, y: float, z: float) -> str:
    """
    Returns the named place (callout) containing a position, e.g.
    "BombsiteA", "LongA", "Mid".

    Returns "Unknown" if the map has no parsed nav data, or if the
    resolved area has no assigned place name - matching the
    "Unknown" convention already used for active_weapon_class in
    feature_extractor.py.
    """

    nav = _load_map(map_name)

    if nav is None:
        return "Unknown"

    area = get_area(map_name, x, y, z)

    if area is None:
        return "Unknown"

    return nav["area_place"].get(area["id"], "Unknown")


# ==========================================================
# Graph Distance
# ==========================================================


def nav_path_distance(
    map_name: str,
    area_id_1: int,
    area_id_2: int,
) -> float:
    """
    Shortest walkable path distance between two areas' centroids,
    following the nav mesh's actual connectivity graph (Dijkstra),
    rather than a straight line that could pass through walls.

    Returns -1 if the map has no parsed nav data, either area id
    is invalid, or no path exists between them.
    """

    nav = _load_map(map_name)

    if nav is None:
        return -1

    areas = nav["areas"]

    if area_id_1 not in areas or area_id_2 not in areas:
        return -1

    if area_id_1 == area_id_2:
        return 0.0

    import heapq

    centroids = {
        area_id: _area_centroid(area)
        for area_id, area in areas.items()
    }

    distances = {area_id_1: 0.0}
    visited = set()
    heap = [(0.0, area_id_1)]

    while heap:

        current_dist, current_id = heapq.heappop(heap)

        if current_id in visited:
            continue

        visited.add(current_id)

        if current_id == area_id_2:
            return current_dist

        current_area = areas[current_id]

        for connection in current_area["connections"]:

            neighbor_id = connection["target_area"]

            if neighbor_id not in areas or neighbor_id in visited:
                continue

            edge_weight = _distance_3d(
                centroids[current_id],
                centroids[neighbor_id],
            )

            new_dist = current_dist + edge_weight

            if new_dist < distances.get(neighbor_id, float("inf")):
                distances[neighbor_id] = new_dist
                heapq.heappush(heap, (new_dist, neighbor_id))

    # Exhausted the graph without reaching area_id_2 - disconnected
    # component (shouldn't normally happen on a single connected
    # map, but guarded rather than assumed).
    return -1


def area_connectivity_degree(map_name: str, area_id: int) -> int:
    """
    Number of outgoing connections from an area - a rough proxy
    for how open/junction-like (high degree) versus how much of a
    chokepoint/corner (low degree) a position is.

    Returns -1 if the map has no parsed nav data or the area id is
    invalid.
    """

    nav = _load_map(map_name)

    if nav is None:
        return -1

    area = nav["areas"].get(area_id)

    if area is None:
        return -1

    return len(area["connections"])


# ==========================================================
# Hiding Spot / Cover Proximity
# ==========================================================


def nearest_hiding_spot(
    map_name: str,
    x: float,
    y: float,
    z: float,
) -> tuple[float, int]:
    """
    Finds the closest hiding spot to a position, searched across
    the whole map (not just the player's current area), since a
    relevant piece of cover may sit just across an area boundary.

    Returns (distance, flags) for the nearest hiding spot, or
    (-1, -1) if the map has no parsed nav data or contains no
    hiding spots at all.
    """

    nav = _load_map(map_name)

    if nav is None:
        return -1, -1

    best_distance = float("inf")
    best_flags = -1

    for area in nav["areas"].values():

        for spot in area["hiding_spots"]:

            loc = spot["location"]
            distance = _distance_3d((x, y, z), (loc["x"], loc["y"], loc["z"]))

            if distance < best_distance:
                best_distance = distance
                best_flags = spot["flags"]

    if best_flags == -1:
        return -1, -1

    return best_distance, best_flags


def is_near_cover(
    map_name: str,
    x: float,
    y: float,
    z: float,
    proximity_units: float = HIDING_SPOT_PROXIMITY_UNITS,
) -> bool:
    """
    Whether a position is within proximity_units of any hiding
    spot flagged IN_COVER.

    See the module-level caution above: the flag-bit interpretation
    used here is a commonly cited convention, not independently
    confirmed against an authoritative Valve source.
    """

    distance, flags = nearest_hiding_spot(map_name, x, y, z)

    if distance == -1:
        return False

    return (
        distance <= proximity_units
        and bool(flags & HIDING_SPOT_FLAG_IN_COVER)
    )


def is_exposed(
    map_name: str,
    x: float,
    y: float,
    z: float,
    proximity_units: float = HIDING_SPOT_PROXIMITY_UNITS,
) -> bool:
    """
    Whether a position is within proximity_units of any hiding
    spot flagged EXPOSED.

    See the module-level caution above regarding flag-bit
    interpretation confidence.
    """

    distance, flags = nearest_hiding_spot(map_name, x, y, z)

    if distance == -1:
        return False

    return (
        distance <= proximity_units
        and bool(flags & HIDING_SPOT_FLAG_EXPOSED)
    )
