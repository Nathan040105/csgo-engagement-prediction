"""
feature_extractor.py

Extracts machine learning features for engagements detected by
engagement_extractor.py.

The extractor receives an engagement and returns a dictionary
containing all features required for dataset generation.

Only the state BEFORE the engagement begins is used so that the
dataset contains no information from the future.
"""

from __future__ import annotations

import math
from typing import Any

from config import FRAME_OFFSET,WEAPON_MAGAZINE_CAPACITY,WEAPON_RESERVE_AMMO
import nav_data
from utils import (
    calculate_distance,
    find_player,
    get_previous_frame,
)


class FeatureExtractor:
    """
    Extracts features describing the game state immediately before an
    engagement begins.
    """

    def __init__(self, frame_offset: int = FRAME_OFFSET):
        """
        Parameters
        ----------
        frame_offset
            Number of frames before the engagement start used for
            feature extraction.
        """
        self.frame_offset = frame_offset
        

    # ==========================================================
    # Frame Selection
    # ==========================================================

    def _get_feature_frame(
        self,
        round_data: dict,
        engagement: dict
    ) -> dict | None:
        """
        Returns the frame used for feature extraction.

        By default this is one frame before the engagement begins
        to avoid leaking information from the engagement itself.
        """

        start_tick = engagement["startTick"]

        return get_previous_frame(
            round_data["frames"],
            start_tick,
            self.frame_offset
        )

    # ==========================================================
    # Player Lookup
    # ==========================================================

    def _get_players(
        self,
        frame: dict,
        engagement: dict
    ) -> tuple[dict, dict] | tuple[None, None]:
        """
        Returns the attacker and victim player dictionaries for
        the engagement from the selected feature frame.

        Returns
        -------
        tuple
            (attacker, victim) or (None, None) if either player
            cannot be found.
        """

        first_event = engagement["events"][0]

        attacker = find_player(
            frame,
            first_event["attacker"]
        )

        victim = find_player(
            frame,
            first_event["victim"]
        )

        return attacker, victim

    # ==========================================================
    # Team Lookup
    # ==========================================================

    def _get_teammates(
        self,
        frame: dict,
        player: dict
    ) -> list[dict]:
        """
        Returns all living teammates excluding the player.
        """

        if player["side"] == "T":
            players = frame["t"]["players"]
        else:
            players = frame["ct"]["players"]

        teammates = []

        for teammate in players:

            if teammate["steamID"] == player["steamID"]:
                continue

            if teammate["isAlive"]:
                teammates.append(teammate)

        return teammates

    def _get_opponents(
        self,
        frame: dict,
        player: dict
    ) -> list[dict]:
        """
        Returns all living opponents.
        """

        if player["side"] == "T":
            players = frame["ct"]["players"]
        else:
            players = frame["t"]["players"]

        return [
            opponent
            for opponent in players
            if opponent["isAlive"]
        ]

    # ==========================================================
    # Utility Methods
    # ==========================================================

    def _closest_distance(
        self,
        player: dict,
        players: list[dict]
    ) -> float:
        """
        Returns the distance to the closest player in a list.

        Returns -1.0 if the list is empty.
        """

        if not players:
            return -1.0

        return min(
            calculate_distance(player, other)
            for other in players
        )

    def _average_distance(
        self,
        player: dict,
        players: list[dict]
    ) -> float:
        """
        Returns the average distance to a list of players.

        Returns -1.0 if the list is empty.
        """

        if not players:
            return -1.0

        distances = [
            calculate_distance(player, other)
            for other in players
        ]

        return sum(distances) / len(distances)
    
    

    # ==========================================================
    # Feature Extraction
    # ==========================================================

    def extract_features(
        self,
        round_data: dict,
        engagement: dict,
        map_name: str,
        is_lan: bool = False,
        match_id: str = "",
        round_num: int = -1
    ) -> dict[str, Any]:
        """
        Extract features describing the game state immediately
        before the engagement begins.

        Parameters
        ----------
        round_data
            Round dictionary from the parsed demo.

        engagement
            Engagement dictionary returned by
            engagement_extractor.py.

        match_id
            Stable per-demo identifier (see build_dataset.py's
            process_demo). Written to the row as-is so downstream
            code can group by it for match-aware train/test
            splitting - it is metadata, not a model feature, and
            should be dropped (or passed separately as `groups`)
            before fitting anything.

        round_num
            The round's number within its demo. Metadata for
            traceability, same caveat as match_id - not intended
            as a model input.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all extracted ML features.

            Returns an empty dictionary if the required frame or
            either player cannot be found.
        """

        frame = self._get_feature_frame(
            round_data,
            engagement
        )

        if frame is None:
            return {}

        attacker, victim = self._get_players(
            frame,
            engagement
        )

        if attacker is None or victim is None:
            return {}
        
        if not attacker.get("isAlive", True) or not victim.get("isAlive", True):
            return {}

        attacker_teammates = self._get_teammates(
            frame,
            attacker
        )

        victim_teammates = self._get_teammates(
            frame,
            victim
        )

        attacker_opponents = self._get_opponents(
            frame,
            attacker
        )

        victim_opponents = self._get_opponents(
            frame,
            victim
        )

        features = {}
        # ======================================================
        # Basic Engagement Information
        # ======================================================

        first_event = engagement["events"][0]

        features["match_id"] = match_id
        features["round_num"] = round_num

        features["start_tick"] = engagement["startTick"]
        features["start_time"] = engagement["startTime"]

        features["attacker_side"] = attacker["side"]
        features["victim_side"] = victim["side"]
        features["map_name"] = map_name
        features["is_lan"] = is_lan

        features["attacker_name"] = first_event["attackerName"]
        features["victim_name"] = first_event["victimName"]

        features["attacker_steamid"] = first_event["attacker"]
        features["victim_steamid"] = first_event["victim"]

        features["engagement_status"] = engagement["status"]

        # ======================================================
        # Player Health
        # ======================================================

        features["attacker_hp"] = attacker["hp"]
        features["victim_hp"] = victim["hp"]

        features["attacker_armor"] = attacker["armor"]
        features["victim_armor"] = victim["armor"]

        features["attacker_has_helmet"] = attacker["hasHelmet"]
        features["victim_has_helmet"] = victim["hasHelmet"]

        # ======================================================
        # Weapons
        # ======================================================

        features["attacker_active_weapon"] = attacker["activeWeapon"]
        features["victim_active_weapon"] = victim["activeWeapon"]

        features["attacker_scoped"] = attacker.get("isScoped", False)
        features["victim_scoped"] = victim.get("isScoped", False)

        features["attacker_zoom_level"] = attacker.get("zoomLevel", 0)
        features["victim_zoom_level"] = victim.get("zoomLevel", 0)

        # ======================================================
        # Economy
        # ======================================================

        features["attacker_cash"] = attacker["cash"]
        features["victim_cash"] = victim["cash"]

        features["attacker_equipment"] = attacker["equipmentValue"]
        features["victim_equipment"] = victim["equipmentValue"]

        features["attacker_total_utility"] = attacker["totalUtility"]
        features["victim_total_utility"] = victim["totalUtility"]

        # ======================================================
        # Bomb Information
        # ======================================================

        features["attacker_has_bomb"] = attacker.get("hasBomb", False)
        features["victim_has_bomb"] = victim.get("hasBomb", False)

        features["bomb_planted"] = frame.get("bombPlanted", False)

        # ======================================================
        # Movement
        # ======================================================

        features["attacker_airborne"] = attacker.get("isAirborne", False)
        features["victim_airborne"] = victim.get("isAirborne", False)

        features["attacker_ducking"] = attacker.get("isDucking", False)
        features["victim_ducking"] = victim.get("isDucking", False)

        features["attacker_walking"] = attacker.get("isWalking", False)
        features["victim_walking"] = victim.get("isWalking", False)

        features["attacker_reloading"] = attacker.get("isReloading", False)
        features["victim_reloading"] = victim.get("isReloading", False)

        features["attacker_blinded"] = attacker.get("isBlinded", False)
        features["victim_blinded"] = victim.get("isBlinded", False)

        # ======================================================
        # Velocity
        # ======================================================

        attacker_speed = (
            attacker["velocityX"] ** 2 +
            attacker["velocityY"] ** 2 +
            attacker["velocityZ"] ** 2
        ) ** 0.5

        victim_speed = (
            victim["velocityX"] ** 2 +
            victim["velocityY"] ** 2 +
            victim["velocityZ"] ** 2
        ) ** 0.5

        features["attacker_speed"] = attacker_speed
        features["victim_speed"] = victim_speed

        # ======================================================
        # Position
        # ======================================================

        features["attacker_x"] = attacker["x"]
        features["attacker_y"] = attacker["y"]
        features["attacker_z"] = attacker["z"]

        features["victim_x"] = victim["x"]
        features["victim_y"] = victim["y"]
        features["victim_z"] = victim["z"]

        features["delta_x"] = attacker["x"] - victim["x"]
        features["delta_y"] = attacker["y"] - victim["y"]
        features["delta_z"] = attacker["z"] - victim["z"]

        features["height_advantage"] = (
            attacker["z"] - victim["z"]
        )

        # ======================================================
        # View Angles
        # ======================================================

        features["attacker_view_x"] = attacker["viewX"]
        features["attacker_view_y"] = attacker["viewY"]

        features["victim_view_x"] = victim["viewX"]
        features["victim_view_y"] = victim["viewY"]

        horizontal_diff = abs(
            attacker["viewX"] - victim["viewX"]
        )

        horizontal_diff = min(
            horizontal_diff,
            360 - horizontal_diff
        )

        features["horizontal_angle_difference"] = horizontal_diff

        features["vertical_angle_difference"] = abs(
            attacker["viewY"] - victim["viewY"]
        )

        # ======================================================
        # Geometry
        # ======================================================

        engagement_distance = calculate_distance(
            attacker,
            victim
        )

        features["engagement_distance"] = engagement_distance

        attacker_other_enemies = [
            enemy
            for enemy in attacker_opponents
            if enemy["steamID"] != victim["steamID"]
        ]

        victim_other_enemies = [
            enemy
            for enemy in victim_opponents
            if enemy["steamID"] != attacker["steamID"]
        ]

        features["attacker_closest_teammate"] = (
            self._closest_distance(
                attacker,
                attacker_teammates
            )
        )

        features["victim_closest_teammate"] = (
            self._closest_distance(
                victim,
                victim_teammates
            )
        )

        features["attacker_closest_enemy"] = (
            self._closest_distance(
                attacker,
                attacker_other_enemies
            )
        )

        features["victim_closest_enemy"] = (
            self._closest_distance(
                victim,
                victim_other_enemies
            )
        )

        features["attacker_average_teammate_distance"] = (
            self._average_distance(
                attacker,
                attacker_teammates
            )
        )

        features["victim_average_teammate_distance"] = (
            self._average_distance(
                victim,
                victim_teammates
            )
        )

        features["attacker_average_enemy_distance"] = (
            self._average_distance(
                attacker,
                attacker_other_enemies
            )
        )

        features["victim_average_enemy_distance"] = (
            self._average_distance(
                victim,
                victim_other_enemies
            )
        )

        # ======================================================
        # Nav Mesh Features
        #
        # map_name is already passed into extract_features; the
        # geometry-mismatch exclusion (see config.py,
        # NAV_MISMATCH_EXCLUDED_MAPS) is applied upstream in
        # build_dataset.py, which skips excluded-map demos before
        # any engagement ever reaches this function. So by the
        # time execution gets here, map_name is already known to
        # be a map whose nav data is considered trustworthy.
        # ======================================================

        attacker_area_id = nav_data.get_area_id(
            map_name, attacker["x"], attacker["y"], attacker["z"]
        )

        victim_area_id = nav_data.get_area_id(
            map_name, victim["x"], victim["y"], victim["z"]
        )

        features["attacker_nav_area_id"] = attacker_area_id
        features["victim_nav_area_id"] = victim_area_id

        attacker_callout = nav_data.get_callout(
            map_name, attacker["x"], attacker["y"], attacker["z"]
        )

        victim_callout = nav_data.get_callout(
            map_name, victim["x"], victim["y"], victim["z"]
        )

        features["attacker_callout"] = attacker_callout
        features["victim_callout"] = victim_callout

        features["same_callout"] = (
            attacker_callout == victim_callout
            and attacker_callout != "Unknown"
        )

        if attacker_area_id != -1 and victim_area_id != -1:

            path_distance = nav_data.nav_path_distance(
                map_name, attacker_area_id, victim_area_id
            )

        else:

            path_distance = -1

        features["nav_path_distance"] = path_distance

        # Path-distance-to-Euclidean-distance ratio: an indirect
        # line-of-sight proxy. Close to 1 suggests a fairly open,
        # direct route between the two players; much greater than
        # 1 suggests the shortest walkable path has to go around
        # something, an indirect signal of a possible obstruction.
        # This is an approximation, not a verified raycast - see
        # nav_data.py and the accompanying methodology notes.
        if path_distance > 0 and engagement_distance > 0:
            features["nav_los_proxy_ratio"] = (
                path_distance / engagement_distance
            )
        else:
            features["nav_los_proxy_ratio"] = -1

        features["attacker_area_connectivity"] = (
            nav_data.area_connectivity_degree(map_name, attacker_area_id)
        )

        features["victim_area_connectivity"] = (
            nav_data.area_connectivity_degree(map_name, victim_area_id)
        )

        features["attacker_near_cover"] = nav_data.is_near_cover(
            map_name, attacker["x"], attacker["y"], attacker["z"]
        )

        features["victim_near_cover"] = nav_data.is_near_cover(
            map_name, victim["x"], victim["y"], victim["z"]
        )

        features["attacker_exposed"] = nav_data.is_exposed(
            map_name, attacker["x"], attacker["y"], attacker["z"]
        )

        features["victim_exposed"] = nav_data.is_exposed(
            map_name, victim["x"], victim["y"], victim["z"]
        )

        # ======================================================
        # Team State
        # ======================================================

        features["t_alive"] = frame["t"]["alivePlayers"]
        features["ct_alive"] = frame["ct"]["alivePlayers"]

        features["t_equipment"] = frame["t"]["teamEqVal"]
        features["ct_equipment"] = frame["ct"]["teamEqVal"]

        features["t_total_utility"] = frame["t"]["totalUtility"]
        features["ct_total_utility"] = frame["ct"]["totalUtility"]

        features["alive_difference"] = (
            frame["ct"]["alivePlayers"] -
            frame["t"]["alivePlayers"]
        )

        # ======================================================
        # Round Context
        # ======================================================

        features["clock_time"] = frame["clockTime"]
        features["frame_tick"] = frame["tick"]

        # ======================================================
        # Teammate Information
        # ======================================================

        features["attacker_alive_teammates"] = len(attacker_teammates)
        features["victim_alive_teammates"] = len(victim_teammates)

        features["attacker_alive_enemies"] = len(attacker_opponents)
        features["victim_alive_enemies"] = len(victim_opponents)

        # ======================================================
        # Player State
        # ======================================================

        features["attacker_ping"] = attacker.get("ping", -1)
        features["victim_ping"] = victim.get("ping", -1)

        features["attacker_standing"] = attacker.get("isStanding", True)
        features["victim_standing"] = victim.get("isStanding", True)

        features["attacker_planting"] = attacker.get("isPlanting", False)
        features["victim_planting"] = victim.get("isPlanting", False)

        features["attacker_defusing"] = attacker.get("isDefusing", False)
        features["victim_defusing"] = victim.get("isDefusing", False)

        features["attacker_buyzone"] = attacker.get("isInBuyZone", False)
        features["victim_buyzone"] = victim.get("isInBuyZone", False)

        features["attacker_bombzone"] = attacker.get("isInBombZone", False)
        features["victim_bombzone"] = victim.get("isInBombZone", False)

        # ======================================================
        # Inventory
        # ======================================================

        def parse_inventory(player):
            """
            Extracts inventory information into ML-friendly features.
            Compatible with AWPy 1.3.1.
            """

            inventory = player.get("inventory") or []

            primary = "No Primary"
            secondary = "No Secondary"

            flash = 0
            smoke = 0
            he = 0
            molotov = 0
            incendiary = 0
            decoy = 0
            zeus = 0

            active_magazine = -1
            active_reserve = -1

            active_weapon = player.get("activeWeapon", "")
            active_weapon_class = "Unknown"

            for item in inventory:

                name = item.get("weaponName", "")
                weapon_class = item.get("weaponClass", "")

                # ----------------------------
                # Active weapon ammunition
                # ----------------------------
                if name == active_weapon:
                    active_magazine = item.get("ammoInMagazine", -1)
                    active_reserve = item.get("ammoInReserve", -1)
                    active_weapon_class = weapon_class

                # ----------------------------
                # Primary weapon
                # ----------------------------
                if weapon_class in (
                    "Rifle",
                    "Rifles",
                    "SMG",
                    "SMGs",
                    "Heavy",
                    "Sniper Rifle",
                    "Sniper Rifles",
                ):
                    primary = name

                # ----------------------------
                # Secondary weapon (pistol)
                # ----------------------------
                elif weapon_class in ("Pistol", "Pistols"):
                    secondary = name

                # ----------------------------
                # Grenades
                # ----------------------------
                elif weapon_class == "Grenade":

                    lname = name.lower()

                    if "flash" in lname:
                        flash += 1

                    elif "smoke" in lname:
                        smoke += 1

                    elif "molotov" in lname:
                        molotov += 1

                    elif "incendiary" in lname:
                        incendiary += 1

                    elif "decoy" in lname:
                        decoy += 1

                    elif lname.startswith("he") or "high explosive" in lname:
                        he += 1

                # ----------------------------
                # Zeus
                # ----------------------------
                elif "zeus" in name.lower():
                    zeus = 1

            return {
                "primary": primary,
                "secondary": secondary,

                "flash": flash,
                "smoke": smoke,
                "he": he,
                "molotov": molotov,
                "incendiary": incendiary,
                "decoy": decoy,
                "zeus": zeus,

                "has_flash": flash > 0,
                "has_smoke": smoke > 0,
                "has_he": he > 0,
                "has_molotov": molotov > 0,
                "has_incendiary": incendiary > 0,
                "has_decoy": decoy > 0,

                "active_magazine": active_magazine,
                "active_reserve": active_reserve,
                "active_weapon_class": active_weapon_class,

                "inventory_size": len(inventory),
            }


        attacker_inv = parse_inventory(attacker)
        victim_inv = parse_inventory(victim)

        attacker_mag_size = WEAPON_MAGAZINE_CAPACITY.get(attacker["activeWeapon"])
        attacker_reserve_size = WEAPON_RESERVE_AMMO.get(attacker["activeWeapon"])

        victim_mag_size = WEAPON_MAGAZINE_CAPACITY.get(victim["activeWeapon"])
        victim_reserve_size = WEAPON_RESERVE_AMMO.get(victim["activeWeapon"])

        # ======================================================
        # Attacker Inventory Features
        # ======================================================

        features["attacker_primary_weapon"] = attacker_inv["primary"]
        features["attacker_secondary_weapon"] = attacker_inv["secondary"]
        features["attacker_active_weapon_class"] = attacker_inv["active_weapon_class"]

        features["attacker_flashbangs"] = attacker_inv["flash"]
        features["attacker_smokes"] = attacker_inv["smoke"]
        features["attacker_hegrenades"] = attacker_inv["he"]
        features["attacker_molotovs"] = attacker_inv["molotov"]
        features["attacker_incendiaries"] = attacker_inv["incendiary"]
        features["attacker_decoys"] = attacker_inv["decoy"]

        features["attacker_has_flash"] = attacker_inv["has_flash"]
        features["attacker_has_smoke"] = attacker_inv["has_smoke"]
        features["attacker_has_he"] = attacker_inv["has_he"]
        features["attacker_has_molotov"] = attacker_inv["has_molotov"]
        features["attacker_has_incendiary"] = attacker_inv["has_incendiary"]
        features["attacker_has_decoy"] = attacker_inv["has_decoy"]

        features["attacker_has_zeus"] = attacker_inv["zeus"]

        if (attacker_mag_size is not None and attacker_inv["active_magazine"] >= 0 and attacker_inv["active_reserve"] >= 0):
            features["attacker_magazine_pct"] = min(attacker_inv["active_magazine"] / attacker_mag_size, 1.0)
        else:
            features["attacker_magazine_pct"] = -1

        if (attacker_reserve_size is not None and attacker_inv["active_reserve"] >= 0):
            features["attacker_reserve_pct"] = min(attacker_inv["active_reserve"] / attacker_reserve_size, 1.0)
        else:
            features["attacker_reserve_pct"] = -1

        features["attacker_inventory_size"] = attacker_inv["inventory_size"]

        # ======================================================
        # Victim Inventory Features
        # ======================================================

        features["victim_primary_weapon"] = victim_inv["primary"]
        features["victim_secondary_weapon"] = victim_inv["secondary"]
        features["victim_active_weapon_class"] = victim_inv["active_weapon_class"]

        features["victim_flashbangs"] = victim_inv["flash"]
        features["victim_smokes"] = victim_inv["smoke"]
        features["victim_hegrenades"] = victim_inv["he"]
        features["victim_molotovs"] = victim_inv["molotov"]
        features["victim_incendiaries"] = victim_inv["incendiary"]
        features["victim_decoys"] = victim_inv["decoy"]

        features["victim_has_flash"] = victim_inv["has_flash"]
        features["victim_has_smoke"] = victim_inv["has_smoke"]
        features["victim_has_he"] = victim_inv["has_he"]
        features["victim_has_molotov"] = victim_inv["has_molotov"]
        features["victim_has_incendiary"] = victim_inv["has_incendiary"]
        features["victim_has_decoy"] = victim_inv["has_decoy"]

        features["victim_has_zeus"] = victim_inv["zeus"]

        if (victim_mag_size is not None and victim_inv["active_magazine"] >= 0 and victim_inv["active_reserve"] >= 0):
            features["victim_magazine_pct"] = min(victim_inv["active_magazine"] / victim_mag_size, 1.0)
        else:
            features["victim_magazine_pct"] = -1

        if (victim_reserve_size is not None and victim_inv["active_reserve"] >= 0):
            features["victim_reserve_pct"] = min(victim_inv["active_reserve"] / victim_reserve_size, 1.0)
        else:
            features["victim_reserve_pct"] = -1

        features["victim_inventory_size"] = victim_inv["inventory_size"]

        # ======================================================
        # Relative Advantage
        # ======================================================

        features["hp_difference"] = (
            attacker["hp"] - victim["hp"]
        )

        features["armor_difference"] = (
            attacker["armor"] - victim["armor"]
        )

        features["equipment_difference"] = (
            attacker["equipmentValue"]
            - victim["equipmentValue"]
        )

        features["utility_difference"] = (
            attacker["totalUtility"]
            - victim["totalUtility"]
        )

        features["speed_difference"] = (
            attacker_speed - victim_speed
        )

        # ======================================================
        # Outcome Label
        # ======================================================

        if engagement["status"] == "attacker_won":

            features["winner_name"] = engagement["winner"]
            features["attacker_won"] = True

        elif engagement["status"] == "victim_won":

            features["winner_name"] = engagement["winner"]
            features["attacker_won"] = False

        else:
            features["winner_name"] = ""
            features["attacker_won"] = None

        # ======================================================
        # Data Validation
        # ======================================================

        for key, value in list(features.items()):

            # Replace NaN values
            if isinstance(value, float):

                if math.isnan(value):
                    features[key] = -1.0

            # Replace None values with empty strings
            elif value is None:

                features[key] = ""

        # ======================================================
        # Return Feature Dictionary
        # ======================================================

        return features