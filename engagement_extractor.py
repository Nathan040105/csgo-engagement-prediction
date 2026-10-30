"""
engagement_extractor.py

Extracts 1v1 engagements from an AWPy parsed CSGO demo.

Engagements begin when the first damage occurs between two players and
continue until one of the following:

    • One player dies
    • Either player begins fighting someone else
    • No interaction occurs for TIMEOUT_SECONDS
    • The round ends

Returns a list of engagement dictionaries that can be passed directly to
feature_extractor.py.
"""

from __future__ import annotations

from typing import Dict, List, Any, FrozenSet

from config import TIMEOUT_SECONDS
from utils import get_round_end_time

# ==========================================================
# Type Aliases
# ==========================================================

Event = Dict[str, Any]
Engagement = Dict[str, Any]

# ==========================================================
# Event Construction
# ==========================================================


def build_event_list(round_data: Dict[str, Any]) -> List[Event]:
    """
    Convert damage and kill events into a single chronological list.

    Parameters
    ----------
    round_data : dict

    Returns
    -------
    List[Event]
    """

    events: List[Event] = []

    # ------------------------------------------------------
    # Damage Events
    # ------------------------------------------------------

    for dmg in round_data.get("damages", []):

        events.append({
            "type": "damage",

            "tick": dmg["tick"],
            "seconds": dmg["seconds"],

            "attacker": dmg["attackerSteamID"],
            "attackerName": dmg["attackerName"],

            "victim": dmg["victimSteamID"],
            "victimName": dmg["victimName"],

            "hpDamage": dmg["hpDamageTaken"]
        })

    # ------------------------------------------------------
    # Kill Events
    # ------------------------------------------------------

    for kill in round_data.get("kills", []):

        events.append({
            "type": "kill",

            "tick": kill["tick"],
            "seconds": kill["seconds"],

            "attacker": kill["attackerSteamID"],
            "attackerName": kill["attackerName"],

            "victim": kill["victimSteamID"],
            "victimName": kill["victimName"]
        })

    # ------------------------------------------------------
    # Chronological Order
    # ------------------------------------------------------

    events.sort(key=lambda e: e["tick"])

    return events


# ==========================================================
# Helper Functions
# ==========================================================


def create_engagement(event: Event) -> Engagement:
    """
    Create a new engagement from the first event.
    """

    return {

        "player1": event["attacker"],
        "player2": event["victim"],

        "players": frozenset([
            event["attacker"],
            event["victim"]
        ]),

        "events": [event],

        "status": "ongoing",

        "startTick": event["tick"],
        "startTime": event["seconds"],

        "lastTick": event["tick"]
    }


def finish_engagement(
    engagement: Engagement,
    status: str,
    end_tick: int,
    end_time: float
) -> None:
    """
    Mark an engagement as finished.
    """

    engagement["status"] = status
    engagement["endTick"] = end_tick
    engagement["endTime"] = end_time



def merge_mutual_kills(
    engagements: List[Engagement]
) -> List[Engagement]:
    """
    Detects trades / mutual kills.

    The per-event extraction loop represents a trade as two
    separate one-sided "death" engagements, since a kill event
    only carries one attacker and one victim at a time. A dead
    player cannot register a kill afterward, so any two "death"
    engagements sharing the same undirected player pair within a
    round can only be the two halves of the same trade - no
    tick-distance threshold needed.
    """

    merged: List[Engagement] = []
    used: set = set()

    death_statuses = ("attacker_won", "victim_won")

    for i, eng in enumerate(engagements):

        if i in used:
            continue

        if eng["status"] not in death_statuses:
            merged.append(eng)
            continue

        pair = eng["players"]
        match_index = None

        for j in range(i + 1, len(engagements)):

            if j in used:
                continue

            if (
                engagements[j]["status"] in death_statuses
                and engagements[j]["players"] == pair
            ):
                match_index = j
                break

        if match_index is None:
            merged.append(eng)
            continue

        other = engagements[match_index]
        used.add(match_index)

        merged_engagement = {
            **eng,
            "events": sorted(
                eng["events"] + other["events"],
                key=lambda e: e["tick"]
            ),
            "status": "mutual_kill",
            "endTick": max(eng["endTick"], other["endTick"]),
            "endTime": max(eng["endTime"], other["endTime"]),
        }

        merged_engagement.pop("winner", None)
        merged_engagement.pop("loser", None)

        merged.append(merged_engagement)

    return merged


def resolve_interrupted_engagement(
    old_pair: FrozenSet[int],
    active_engagements: Dict[FrozenSet[int], Engagement],
    player_to_pair: Dict[int, FrozenSet[int]],
    engagements: List[Engagement],
    tick: int,
    seconds: float,
) -> None:
    """
    Ends an active engagement because one of its two players has
    just started a new engagement elsewhere.

    Same-tick special case: if the old engagement was itself only
    just created on this exact tick (startTick == tick), it never
    had any real elapsed duration. This happens when a single tick
    carries several damage events for one attacker against
    alternating victims (e.g. splash/multi-hit damage landing on
    more than one player at once) - the pair gets created,
    immediately "interrupted" by the next event, created again,
    interrupted again, and so on, all within one tick. Recording
    each of those as a distinct "interrupted" engagement produces
    duplicate, zero-duration rows that don't correspond to anything
    a human reviewing the demo would call a separate 1v1 - so
    they're discarded here (removed from the active/lookup state
    without being appended to the output) rather than finished and
    recorded. A genuine interrupt - where the old engagement's
    startTick is strictly earlier than the interrupting tick, i.e.
    real time actually elapsed - is unaffected and is recorded
    exactly as before.
    """

    if old_pair not in active_engagements:
        return

    old_engagement = active_engagements[old_pair]

    if old_engagement["startTick"] != tick:

        finish_engagement(
            engagement=old_engagement,
            status="interrupted",
            end_tick=tick,
            end_time=seconds
        )

        engagements.append(old_engagement)

    p1, p2 = tuple(old_pair)

    del active_engagements[old_pair]

    player_to_pair.pop(p1, None)
    player_to_pair.pop(p2, None)


# ==========================================================
# Public API
# ==========================================================


def extract_engagements(
    round_data: Dict[str, Any],
    tick_rate: int
) -> List[Engagement]:
    """
    Extract Version 2 engagements from a single round.

    Parameters
    ----------
    round_data : dict

    tick_rate : int

    Returns
    -------
    List[Engagement]
    """

    timeout_ticks = int(TIMEOUT_SECONDS * tick_rate)

    events = build_event_list(round_data)

    engagements: List[Engagement] = []

    # ------------------------------------------------------
    # Active Engagements
    #
    # pair -> engagement
    # ------------------------------------------------------

    active_engagements: Dict[
        FrozenSet[int],
        Engagement
    ] = {}

    # ------------------------------------------------------
    # Player Lookup
    #
    # player -> pair
    # ------------------------------------------------------

    player_to_pair: Dict[
        int,
        FrozenSet[int]
    ] = {}

    # ======================================================
    # Main Version 2 Extraction Loop
    # ======================================================

    for event in events:

        tick = event["tick"]

        attacker = event["attacker"]
        victim = event["victim"]

        if attacker is None or victim is None:
            continue

        if attacker == victim:
            continue

        pair = frozenset([attacker, victim])

                # --------------------------------------------------
        # Step 1
        # Timeout inactive engagements
        # --------------------------------------------------

        timed_out_pairs = []

        for active_pair, engagement in active_engagements.items():

            if tick - engagement["lastTick"] > timeout_ticks:

                finish_engagement(
                    engagement=engagement,
                    status="timedout",
                    end_tick=engagement["lastTick"],
                    end_time=engagement["events"][-1]["seconds"]
                )

                engagements.append(engagement)
                timed_out_pairs.append(active_pair)

        # Remove timed-out engagements

        for active_pair in timed_out_pairs:

            p1, p2 = tuple(active_pair)

            del active_engagements[active_pair]

            player_to_pair.pop(p1, None)
            player_to_pair.pop(p2, None)

        # --------------------------------------------------
        # Step 2
        # Existing engagement?
        # --------------------------------------------------

        if pair in active_engagements:

            engagement = active_engagements[pair]

            engagement["events"].append(event)
            engagement["lastTick"] = tick

        else:

            # ----------------------------------------------
            # Interrupt attacker's previous engagement
            # ----------------------------------------------

            if attacker in player_to_pair:

                resolve_interrupted_engagement(
                    old_pair=player_to_pair[attacker],
                    active_engagements=active_engagements,
                    player_to_pair=player_to_pair,
                    engagements=engagements,
                    tick=tick,
                    seconds=event["seconds"],
                )

            # ----------------------------------------------
            # Interrupt victim's previous engagement
            # ----------------------------------------------

            if victim in player_to_pair:

                resolve_interrupted_engagement(
                    old_pair=player_to_pair[victim],
                    active_engagements=active_engagements,
                    player_to_pair=player_to_pair,
                    engagements=engagements,
                    tick=tick,
                    seconds=event["seconds"],
                )

            # ----------------------------------------------
            # Start new engagement
            # ----------------------------------------------

            engagement = create_engagement(event)

            active_engagements[pair] = engagement

            player_to_pair[attacker] = pair
            player_to_pair[victim] = pair

        # --------------------------------------------------
        # Step 3
        # Kill ends engagement
        # --------------------------------------------------
        if event["type"] == "kill":

            engagement = active_engagements[pair]

            engagement["winner"] = event["attackerName"]
            engagement["loser"] = event["victimName"]

            # Compare against this engagement's own initiating
            # event (events[0]) rather than the kill event itself,
            # since the person who lands the kill is not always
            # the person who started this engagement instance
            # (e.g. the victim of the first exchange turns the
            # fight around and gets the kill). SteamID is used
            # rather than name to avoid ambiguity from
            # duplicate/renamed player names.
            initiating_event = engagement["events"][0]

            if event["attacker"] == initiating_event["attacker"]:
                result_status = "attacker_won"
            else:
                result_status = "victim_won"

            finish_engagement(
                engagement=engagement,
                status=result_status,
                end_tick=tick,
                end_time=event["seconds"]
            )

            engagements.append(engagement)

            del active_engagements[pair]

            player_to_pair.pop(attacker, None)
            player_to_pair.pop(victim, None)
    # ======================================================
    # Round Finished
    # ======================================================

    round_end_tick = round_data["frames"][-1]["tick"]
    round_end_time = get_round_end_time(round_data)

    for engagement in active_engagements.values():

        finish_engagement(
            engagement=engagement,
            status="round_end",
            end_tick=round_end_tick,
            end_time=round_end_time
        )

        engagements.append(engagement)

    return merge_mutual_kills(engagements)


# ==========================================================
# Debugging
# ==========================================================

def print_engagements(
    engagements: List[Engagement]
) -> None:
    """
    Pretty-print extracted engagements.

    Useful for validating extractor behaviour on individual demos.
    """

    print(f"\nFound {len(engagements)} engagements\n")

    for i, engagement in enumerate(engagements, start=1):

        first = engagement["events"][0]

        print("=" * 60)
        print(f"Engagement {i}")
        print("=" * 60)

        print(
            f"Players : "
            f"{first['attackerName']} vs {first['victimName']}"
        )

        print(f"Start   : {engagement['startTime']:.2f}s")
        print(f"End     : {engagement['endTime']:.2f}s")

        duration = engagement["endTime"] - engagement["startTime"]

        print(f"Duration: {duration:.2f}s")
        print(f"Events  : {len(engagement['events'])}")
        print(f"Status  : {engagement['status']}")

        if "winner" in engagement:
            print(f"Winner  : {engagement['winner']}")
        else:
            print("Winner  : None")

        print("\nTimeline")

        for event in engagement["events"]:

            if event["type"] == "damage":

                print(
                    f"{event['seconds']:6.2f}s | "
                    f"{event['attackerName']} damaged "
                    f"{event['victimName']} "
                    f"({event['hpDamage']} HP)"
                )

            else:

                print(
                    f"{event['seconds']:6.2f}s | "
                    f"{event['attackerName']} killed "
                    f"{event['victimName']}"
                )

        print()


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    import json
    import lzma

    demo_path = "sample_demo.json.xz"

    with lzma.open(
        demo_path,
        "rt",
        encoding="utf-8"
    ) as file:

        demo = json.load(file)

    first_round = demo["gameRounds"][0]

    engagements = extract_engagements(
        first_round,
        demo["tickRate"]
    )

    print_engagements(engagements)