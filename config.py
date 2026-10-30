"""
config.py

Configuration settings for the CSGO Engagement Dataset Builder.

Modify values here rather than throughout the codebase.
"""

# ============================================================
# Dataset Paths
# ============================================================

# Root directory containing all ESTA demos.
# Expected structure:
#
# demos/
#     lan/
#     online/
#
from pathlib import Path

# Project root (folder containing this file)
PROJECT_ROOT = Path(__file__).resolve().parent

# ESTA demos
DATASET_DIR = PROJECT_ROOT / "esta"

# Output
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "engagement_dataset.csv"

# Engagement extraction
TIMEOUT_SECONDS = 5

# ============================================================
# Frame Selection
# ============================================================

# Number of frames before the engagement start to use for
# feature extraction.
#
# 0 = exact engagement frame
# 1 = previous frame (~0.5 sec earlier)
FRAME_OFFSET = 1

# ============================================================
# Processing
# ============================================================

# Continue processing if a demo cannot be parsed.
SKIP_CORRUPTED_DEMOS = True

# Print progress while processing demos.
SHOW_PROGRESS = True

# ============================================================
# Random Seed
# ============================================================

# Used later if any randomized processing is added.
RANDOM_SEED = 42

# ============================================================
# Weapon Magazine Capacities (CS:GO) - January 2021 to May 2022
#
# Weapon magazine capacities were obtained from Liquipedia's
# Counter-Strike weapon reference. Historical values were
# verified against official Valve CS:GO patch notes where
# weapon capacities changed during or after the dataset period
# (e.g. the M4A1-S magazine size).
#
# Sources:
# https://liquipedia.net/counterstrike/Portal:Weapons
# https://blog.counter-strike.net/category/updates/
#
# These values are used to normalize weapon ammunition features
# during feature extraction.
# ============================================================

WEAPON_MAGAZINE_CAPACITY = {

    # Pistols
    "Glock-18": 20,
    "USP-S": 12,
    "P2000": 13,
    "Dual Berettas": 30,
    "P250": 13,
    "Five-SeveN": 20,
    "Tec-9": 18,
    "CZ75 Auto": 12,
    "Desert Eagle": 7,
    "R8 Revolver": 8,

    # SMGs
    "MAC-10": 30,
    "MP9": 30,
    "MP7": 30,
    "MP5-SD": 30,
    "UMP-45": 25,
    "P90": 50,
    "PP-Bizon": 64,

    # Rifles
    "AK-47": 30,
    "M4A4": 30,
    "M4A1-S": 25,
    "M4A1": 25,  # Alias used by AWPy/demo parser
    "FAMAS": 25,
    "Galil AR": 35,
    "AUG": 30,
    "SG 553": 30,

    # Snipers
    "SSG 08": 10,
    "AWP": 10,
    "SCAR-20": 20,
    "G3SG1": 20,

    # Heavy
    "Nova": 8,
    "XM1014": 7,
    "MAG-7": 5,
    "Sawed-Off": 7,
    "M249": 100,
    "Negev": 150,
}

# ============================================================
# Weapon Reserve Ammo Capacities (CS:GO) - January 2021 to May 2022
#
# "Reserve ammo" = maximum spare ammunition carried outside the
# active magazine (i.e. total carryable ammo = magazine size +
# reserve ammo). Cross-verified against Valve's official CS:GO
# patch notes and the Valve Developer Community's SDK entity
# documentation (developer.valvesoftware.com/wiki/Weapon_<name>).
#
# Two values required historical correction from a first-pass
# reference:
#   - M4A1-S: 75 (buffed from 40 on 13 Mar 2019; not nerfed again
#     to 80 until 15 Jun 2022, after this dataset's window closes)
#   - Negev: 300 (one secondary source listed 200; Valve's own SDK
#     docs and an independent total-ammo-pool figure both confirm
#     300)
#
# KNOWN UNRESOLVED DISCREPANCY - AWP: used as 30 here (majority-
# corroborated value); one secondary source implies 25. Noted as
# a sourcing caveat rather than silently resolved.
#
# Sources:
# https://developer.valvesoftware.com/wiki/Weapon_ak47
# https://developer.valvesoftware.com/wiki/Weapon_awp
# https://developer.valvesoftware.com/wiki/Weapon_m249
# https://developer.valvesoftware.com/wiki/Weapon_negev
# https://www.hltv.org/news/34061/csgo-update-brings-changes-to-m4a1-s-and-ancient
# https://liquipedia.net/counterstrike/Portal:Weapons
# https://blog.counter-strike.net/category/updates/
# ============================================================

WEAPON_RESERVE_AMMO = {

    # Pistols
    "Glock-18": 120,
    "USP-S": 24,
    "P2000": 52,
    "Dual Berettas": 120,
    "P250": 26,
    "Five-SeveN": 100,
    "Tec-9": 120,
    "CZ75 Auto": 12,
    "Desert Eagle": 35,
    "R8 Revolver": 8,

    # SMGs
    "MAC-10": 100,
    "MP9": 120,
    "MP7": 120,
    "MP5-SD": 120,
    "UMP-45": 100,
    "P90": 100,
    "PP-Bizon": 120,

    # Rifles
    "AK-47": 90,
    "M4A4": 90,
    "M4A1-S": 75,
    "M4A1": 75,  # Alias used by AWPy/demo parser
    "FAMAS": 90,
    "Galil AR": 90,
    "AUG": 90,
    "SG 553": 90,

    # Snipers
    "SSG 08": 90,
    "AWP": 30,  # see "KNOWN UNRESOLVED DISCREPANCY" note above
    "SCAR-20": 90,
    "G3SG1": 90,

    # Heavy
    "Nova": 32,
    "XM1014": 32,
    "MAG-7": 32,
    "Sawed-Off": 32,
    "M249": 200,
    "Negev": 300,
}

# ============================================================
# Nav Mesh Data
# ============================================================
 
# Root directory for nav mesh data. Expects a "parsed" subfolder
# containing <map_name>.json files produced by parser.py.
NAV_DATA_DIR = PROJECT_ROOT / "nav_data"
 
# Distance (in game units) within which a hiding spot is
# considered "near" a player, for is_near_cover() / is_exposed().
# This threshold was not derived from any authoritative source -
# it's a starting estimate and should be sanity-checked against
# real engagement distances before being treated as final.
HIDING_SPOT_PROXIMITY_UNITS = 150


# ============================================================
# Nav Mesh Geometry Mismatch - Excluded Maps
# ============================================================
NAV_MISMATCH_EXCLUDED_MAPS = {
    "de_vertigo",
}