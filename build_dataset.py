"""
build_dataset.py

Main driver for building the CSGO engagement dataset.

This script recursively searches the ESTA demo directory,
extracts engagements from every round, computes machine
learning features, and writes one row per engagement to
a CSV file.

Pipeline

    Demo
        ↓
    Rounds
        ↓
    EngagementExtractor
        ↓
    FeatureExtractor
        ↓
    CSV Dataset
"""

from __future__ import annotations

import csv
import json
import lzma
from pathlib import Path
from typing import Iterator

from config import (
    DATASET_DIR,
    OUTPUT_CSV,
    SHOW_PROGRESS,
    SKIP_CORRUPTED_DEMOS,
    NAV_MISMATCH_EXCLUDED_MAPS,
)

from engagement_extractor import extract_engagements
from feature_extractor import FeatureExtractor


class DatasetBuilder:
    """
    Builds the engagement dataset from all ESTA demos.
    """

    def __init__(self) -> None:

        self.feature_extractor = FeatureExtractor()

        self.total_demos = 0
        self.total_rounds = 0
        self.total_engagements = 0
        self.failed_demos = 0
        self.excluded_map_demos = 0

        self.csv_writer = None
        self.csv_file = None

    # ==========================================================
    # Demo Discovery
    # ==========================================================

    def find_demo_files(self) -> list[Path]:
        """
        Recursively finds every .json.xz demo inside
        DATASET_DIR.

        Returns
        -------
        list[Path]
            Sorted list of demo paths.
        """

        demo_files = sorted(
            DATASET_DIR.rglob("*.json.xz")
        )

        return demo_files

    # ==========================================================
    # Demo Loading
    # ==========================================================

    def load_demo(
        self,
        demo_path: Path
    ) -> dict:
        """
        Loads a compressed ESTA demo.

        Parameters
        ----------
        demo_path
            Path to the .json.xz demo.

        Returns
        -------
        dict
            Parsed demo dictionary.
        """

        with lzma.open(
            demo_path,
            "rt",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # ==========================================================
    # CSV Handling
    # ==========================================================

    def open_csv(self) -> None:
        """
        Opens the output CSV for writing.

        The header is written automatically using the
        first extracted feature dictionary.
        """

        self.csv_file = open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.csv_writer = None

    def close_csv(self) -> None:
        """
        Closes the output CSV.
        """

        if self.csv_file is not None:
            self.csv_file.close()

    def write_feature_row(
        self,
        features: dict
    ) -> None:
        """
        Writes one engagement feature dictionary to the CSV.

        The first row determines the CSV header.
        """

        if not features:
            return

        if self.csv_writer is None:

            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=list(features.keys())
            )

            self.csv_writer.writeheader()

        self.csv_writer.writerow(features)

    # ==========================================================
    # Progress Reporting
    # ==========================================================

    def print_progress(
        self,
        current_demo: int,
        total_demos: int,
        demo_path: Path
    ) -> None:
        """
        Prints progress while processing demos.
        """

        if not SHOW_PROGRESS:
            return

        print(
            f"[{current_demo}/{total_demos}] "
            f"{demo_path.name}"
        )

    # ==========================================================
    # Generator
    # ==========================================================

    def iter_rounds(
        self,
        demo: dict
    ) -> Iterator[tuple[int, dict]]:
        """
        Yields (round_index, round_data) for every round contained
        within a demo.

        round_index is the round's position in demo["gameRounds"]
        (0-based, includes warmup/empty rounds in the count so it
        stays stable regardless of what process_demo later skips).
        It exists as a guaranteed-present fallback for round_num -
        see process_demo, which prefers round_data["roundNum"] when
        the field is actually present in the parsed JSON and only
        falls back to this index otherwise.
        """

        for round_index, round_data in enumerate(demo["gameRounds"]):
            yield round_index, round_data
    
    # ==========================================================
    # Round Processing
    # ==========================================================

    def process_round(
        self,
        round_data: dict,
        tick_rate,
        map_name: str,
        is_lan: bool,
        match_id: str,
        round_num: int,
    ) -> None:
        """
        Extracts engagements from a single round and writes
        one CSV row per valid engagement.

        Parameters
        ----------
        round_data
            One round from the parsed demo.

        match_id
            Stable identifier for the demo this round came from
            (see process_demo). Written to every engagement row so
            downstream code (model_tuning.py's `groups` parameter)
            can do match-aware train/test splitting instead of
            leaking rounds from the same match across the split.

        round_num
            This round's number within its demo. Not intended as
            a model feature - it's here for traceability/debugging
            (e.g. tying a row back to the exact round in the demo).
        """

        self.total_rounds += 1

        # ----------------------------------------------
        # Extract engagements
        # ----------------------------------------------

        engagements = extract_engagements(
            round_data,
            tick_rate
        )

        # ----------------------------------------------
        # Extract features for each engagement
        # ----------------------------------------------

        for engagement in engagements:

            features = self.feature_extractor.extract_features(
                round_data,
                engagement,
                map_name,
                is_lan,
                match_id,
                round_num
            )

            if not features:
                continue

            self.write_feature_row(features)

            self.total_engagements += 1

    # ==========================================================
    # Demo Processing
    # ==========================================================

    def process_demo(
        self,
        demo_path: Path
    ) -> None:
        """
        Processes one ESTA demo.

        Parameters
        ----------
        demo_path
            Path to the demo file.
        """

        try:

            demo = self.load_demo(demo_path)
            map_name = demo.get("mapName", "Unknown")
            is_lan = "lan" in str(demo_path)

            # ------------------------------------------------------
            # Match Identifier
            #
            # Uses the demo's filename stem rather than demo["matchID"]
            # (or similar), because the filename is guaranteed to exist
            # and be unique per file regardless of what fields a given
            # demo's JSON does or doesn't carry - the pipeline's existing
            # convention (see WEAPON_MAGAZINE_CAPACITY lookups, nav_data's
            # -1 sentinels) is to prefer a value that can't silently be
            # missing over one that's technically "more correct" but
            # fragile. This is the group key model_tuning.py's `groups`
            # parameter expects, so that a train/test split never puts
            # rounds from the same demo on both sides.
            # ------------------------------------------------------

            match_id = demo_path.stem

            print(demo_path)
            print(is_lan)

        except Exception as error:

            self.failed_demos += 1

            if SKIP_CORRUPTED_DEMOS:

                print(
                    f"Failed to load "
                    f"{demo_path.name}"
                )

                print(error)

                return

            raise

        if map_name in NAV_MISMATCH_EXCLUDED_MAPS:

            self.excluded_map_demos += 1

            if SHOW_PROGRESS:
                print(
                    f"Skipped (excluded map: {map_name}): "
                    f"{demo_path.name}"
                )

            return

        self.total_demos += 1

        # ----------------------------------------------
        # Process every round
        # ----------------------------------------------
        tick_rate = demo["tickRate"]
        for round_index, round_data in self.iter_rounds(demo):

            # Skip warmup rounds

            if round_data.get("isWarmup", False):
                continue

            # Skip empty rounds

            if not round_data.get("frames"):
                continue

            # Prefer the round's own roundNum field when the parsed
            # JSON actually has one (matches what you'd see reviewing
            # the demo in-game); fall back to the enumeration index
            # from iter_rounds otherwise, so this is never missing.
            round_num = round_data.get("roundNum", round_index)

            self.process_round(
                round_data,
                tick_rate,
                map_name,
                is_lan,
                match_id,
                round_num
            )

    # ==========================================================
    # Demo Loop
    # ==========================================================

    def process_all_demos(
        self,
        demo_files: list[Path]
    ) -> None:
        """
        Processes every demo found in the dataset.

        Parameters
        ----------
        demo_files
            List of demo paths.
        """

        total = len(demo_files)

        for index, demo_path in enumerate(
            demo_files,
            start=1
        ):

            self.print_progress(
                index,
                total,
                demo_path
            )

            self.process_demo(demo_path)
    
        # ==========================================================
    # Build Dataset
    # ==========================================================

    def build(self):

        demo_files = self.find_demo_files()

        if not demo_files:
            print("No demos found.")
            return

        print(f"Found {len(demo_files)} demos.")

        self.open_csv()

        try:
            self.process_all_demos(demo_files)
        finally:
            self.close_csv()

        print("\nDataset complete.")
        print(f"Demos processed : {self.total_demos}")
        print(f"Rounds processed: {self.total_rounds}")
        print(f"Engagements     : {self.total_engagements}")
        print(f"Failed demos    : {self.failed_demos}")
        print(f"Excluded (map)  : {self.excluded_map_demos}")
        print(f"CSV saved to    : {OUTPUT_CSV}")


# ==============================================================
# Main Entry Point
# ==============================================================

def main() -> None:
    """
    Entry point for dataset generation.
    """

    builder = DatasetBuilder()
    builder.build()


if __name__ == "__main__":
    main()