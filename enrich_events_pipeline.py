"""Backend enrichment pipeline for the GoalUnit event viewer.

Reads the raw Wyscout `events.parquet`, precomputes every tactical
classification/flag the Streamlit frontend (`st_events.py`) needs, and writes
a curated, dtype-optimized `events_enriched.parquet`. The goal is for
`st_events.py` to become a thin reader of precomputed boolean/category
columns instead of recomputing temporal joins and spatial masks per request.

Reuses the vectorized conventions already established in
`player_evaluation_pipeline.py` (`build_continuous_timeline`,
`_is_penalty_area`, `build_key_passes`) with one deliberate change: all
spatial masks here use the attack-normalized `*Adj*` coordinates rather than
the raw `start/endCoordinatesX/Y` columns.

Why: raw coordinates are tied to the PHYSICAL pitch, not to attacking
direction. Verified empirically on this dataset -- GOAL events cluster at
BOTH ends in raw coordinates (`startCoordinatesX` quartiles: -52.5, -52.5,
+52.5, +52.5), because teams switch ends at half time and raw X is fixed to
the stadium frame. Only `startAdjCoordinatesX` clusters exclusively at +52.5
for every goal, confirming it is the attack-normalized column. Reusing raw
coordinates for box-entry/final-third masks (as the evaluation pipeline
currently does) would misclassify roughly half of all events -- whichever
side defended the +X end that period. Every spatial flag below uses Adj
coordinates to avoid that bug.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd


def _load_local_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
_load_local_env_file(PROJECT_ROOT / ".env")

DATA_DIR: Final[Path] = Path(os.getenv("GOALUNIT_DATA_DIR", str(PROJECT_ROOT / "data")))
EVENTS_PARQUET: Final[Path] = Path(
    os.getenv("GOALUNIT_EVENTS_PARQUET", str(DATA_DIR / "events.parquet"))
)
ENRICHED_EVENTS_PARQUET: Final[Path] = Path(
    os.getenv("GOALUNIT_ENRICHED_EVENTS_PARQUET", str(DATA_DIR / "events_enriched.parquet"))
)

# Attack-normalized (Adj) pitch geometry, centred on (0, 0); goal line at +PITCH_HALF_LENGTH.
PITCH_HALF_LENGTH: Final[float] = 52.5
PENALTY_AREA_DEPTH: Final[float] = 16.5
PENALTY_AREA_HALF_WIDTH: Final[float] = 20.16
PENALTY_AREA_X_MIN: Final[float] = PITCH_HALF_LENGTH - PENALTY_AREA_DEPTH
FINAL_THIRD_X: Final[float] = PITCH_HALF_LENGTH / 3
GOAL_POST_ABS_Y: Final[float] = 7.32 / 2
PROGRESSIVE_GAIN_THRESHOLD: Final[float] = 10.0
KEY_PASS_WINDOW_SECONDS: Final[float] = 5.0
DANGEROUS_CROSS_WINDOW_SECONDS: Final[float] = 5.0
DANGEROUS_CROSS_MIN_XG: Final[float] = 0.1

DRIBBLE_SUBTYPES: Final[list[str]] = [
    "CARRY", "ESCAPE_PRESSURE", "PROGRESSIVE_CARRY", "STEPPING_IN", "ONE_VS_ONE",
]
DRIBBLE_VARIANT_ACTIONS: Final[set[str]] = {"DRIBBLE", *DRIBBLE_SUBTYPES}
BOX_ENTRY_ACTIONS: Final[set[str]] = {"PASS", *DRIBBLE_VARIANT_ACTIONS}
WING_LANES: Final[frozenset[str]] = frozenset({"LEFT_WING", "RIGHT_WING"})

# Columns pulled from the raw export; a curated subset, not the full ~150-column schema.
RAW_COLUMNS: Final[list[str]] = [
    "matchId", "seasonId", "seasonName", "label", "dateutc",
    "homeClubId", "homeClubName", "awayClubId", "awayClubName",
    "clubId", "clubName", "attackingClubId",
    "playerId", "playerName", "playerPosition", "passReceiverPlayerId", "duelPlayerId",
    "periodId", "eventNumber", "gameTimeInSec",
    "actionType", "dribbleType", "duelType", "result", "bodyPart", 
    "startLane", "endLane",
    "startPitchPosition", "endPitchPosition",
    "startAdjCoordinatesX", "startAdjCoordinatesY",
    "endAdjCoordinatesX", "endAdjCoordinatesY",
    "SHOT_XG",
    "SHOT_AT_GOAL_OFF_TARGET_NUMBER",
    "SHOT_AT_GOAL_NUMBER_SUCCESS",
    "SHOT_AT_GOAL_NUMBER_SAVED",
    "SHOT_AT_GOAL_NUMBER_CAUGHT",
    "SHOT_AT_GOAL_NUMBER_BLOCKED",
    "ASSISTS",
    "WON_GROUND_DUELS",
    "WON_AERIAL_DUELS",
    "BALL_WIN_NUMBER",
]

# Columns safe to bucket as pandas `category` dtype (low-to-moderate cardinality strings).
CATEGORY_COLUMNS: Final[list[str]] = [
    "seasonName", "label", "homeClubName", "awayClubName", "clubName",
    "playerName", "playerPosition", "actionType", "dribbleType", "duelType",
    "result", "bodyPart", "startLane", "shot_outcome", "match_display",
]
# Always-populated integer identifiers -> plain int32 (no NaNs to preserve).
DENSE_INT_COLUMNS: Final[list[str]] = [
    "matchId", "seasonId", "homeClubId", "awayClubId", "clubId", "playerId",
    "eventNumber",
]
# Sparse integer identifiers -> nullable Int32 (NaNs are meaningful, e.g. no pass receiver).
NULLABLE_INT_COLUMNS: Final[list[str]] = [
    "attackingClubId", "passReceiverPlayerId", "duelPlayerId", "ground_duel_loser_id",
]
FLOAT32_COLUMNS: Final[list[str]] = [
    "gameTimeInSec",
    "startAdjCoordinatesX", "startAdjCoordinatesY",
    "endAdjCoordinatesX", "endAdjCoordinatesY",
    "SHOT_XG",
]
# Raw counter/sub-type columns fully superseded by shot_outcome, is_assist, is_duel_won,
# is_ball_recovery, and the harmonized actionType - dropped once those are derived.
COLUMNS_TO_DROP_AFTER_ENGINEERING: Final[list[str]] = [
    "dribbleType", "duelType",
    "SHOT_AT_GOAL_OFF_TARGET_NUMBER",
    "SHOT_AT_GOAL_NUMBER_SUCCESS",
    "SHOT_AT_GOAL_NUMBER_SAVED",
    "SHOT_AT_GOAL_NUMBER_CAUGHT",
    "SHOT_AT_GOAL_NUMBER_BLOCKED",
    "ASSISTS",
    "WON_GROUND_DUELS",
    "WON_AERIAL_DUELS",
    "BALL_WIN_NUMBER",
]


def _validate_input_path() -> None:
    """Validate the resolved raw-events path before attempting pipeline execution."""
    if not EVENTS_PARQUET.exists():
        raise FileNotFoundError(
            "events.parquet was not found. Configure GOALUNIT_DATA_DIR or "
            "GOALUNIT_EVENTS_PARQUET in .env or your shell environment. "
            f"Resolved path: {EVENTS_PARQUET}"
        )


def load_raw_events() -> pd.DataFrame:
    """Load the curated raw-column subset needed to engineer every flag below."""
    _validate_input_path()
    events_df = pd.read_parquet(EVENTS_PARQUET, columns=RAW_COLUMNS)
    return events_df.reset_index(drop=True)


def harmonize_action_types(events_df: pd.DataFrame) -> pd.DataFrame:
    """Fold dribble sub-types and aerial duels into actionType (mirrors st_events.py).

    2023/2024 has no dribbleType at all, so generic "DRIBBLE" rows persist for
    that season; "periodId"-only dribble flags (is_progressive_carry, etc.)
    are computed from geometry below so they still work for that season.
    """
    events_df = events_df.copy()

    dribble_mask = events_df["actionType"].eq("DRIBBLE") & events_df["dribbleType"].notna()
    events_df.loc[dribble_mask, "actionType"] = events_df.loc[dribble_mask, "dribbleType"]

    # Wyscout has no dedicated aerial-duel event; it tags the header onto whatever
    # action followed (RECEPTION, LOOSE_BALL_REGAIN, PASS, etc.) via duelType instead.
    aerial_mask = events_df["duelType"].eq("AERIAL_DUEL")
    events_df.loc[aerial_mask, "actionType"] = "AERIAL_DUEL"
    return events_df


def _is_penalty_area(coordinate_x: pd.Series, coordinate_y: pd.Series) -> pd.Series:
    """Attacking penalty area in attack-normalized Adj coordinates."""
    return coordinate_x.ge(PENALTY_AREA_X_MIN) & coordinate_y.abs().le(PENALTY_AREA_HALF_WIDTH)


def classify_shot_outcomes(events_df: pd.DataFrame) -> pd.Series:
    """Bucket each SHOT into one of 5 mutually exclusive outcomes (same split as st_events.py)."""
    conditions = [
        events_df["SHOT_AT_GOAL_NUMBER_SUCCESS"].notna(),
        events_df["SHOT_AT_GOAL_NUMBER_SAVED"].notna() | events_df["SHOT_AT_GOAL_NUMBER_CAUGHT"].notna(),
        events_df["SHOT_AT_GOAL_NUMBER_BLOCKED"].notna(),
        events_df["SHOT_AT_GOAL_OFF_TARGET_NUMBER"].notna(),
    ]
    choices = ["GOAL", "SAVED", "BLOCKED", "OFF_TARGET"]
    # Remaining shots (e.g. off the woodwork) are the only ones left uncovered above.
    outcomes = pd.Series(np.select(conditions, choices, default="ON_TARGET"), index=events_df.index)
    return outcomes.where(events_df["actionType"].eq("SHOT"))


def flag_key_passes(events_df: pd.DataFrame) -> pd.Series:
    """Per-event boolean: successful pass whose receiver shoots within 5s, same half.

    Vectorized adaptation of `build_key_passes` in player_evaluation_pipeline.py,
    which aggregates straight to a per-player count; this keeps the row-level
    match so it can be stored as a flag and later filtered/plotted directly.
    """
    passes = events_df.loc[
        events_df["actionType"].eq("PASS") & events_df["result"].eq("SUCCESS"),
        ["matchId", "periodId", "eventNumber", "gameTimeInSec", "attackingClubId", "passReceiverPlayerId"],
    ].rename(columns={"eventNumber": "pass_event_number", "gameTimeInSec": "pass_time_sec"})

    shots = events_df.loc[
        events_df["actionType"].eq("SHOT"),
        ["matchId", "periodId", "attackingClubId", "playerId", "gameTimeInSec"],
    ].rename(columns={"playerId": "shooter_id", "gameTimeInSec": "shot_time_sec"})

    matched = passes.merge(
        shots,
        left_on=["matchId", "periodId", "attackingClubId", "passReceiverPlayerId"],
        right_on=["matchId", "periodId", "attackingClubId", "shooter_id"],
        how="inner",
    )
    elapsed_seconds = matched["shot_time_sec"] - matched["pass_time_sec"]
    matched = matched.loc[elapsed_seconds.gt(0) & elapsed_seconds.le(KEY_PASS_WINDOW_SECONDS)]
    key_pass_keys = matched.drop_duplicates(
        subset=["matchId", "periodId", "pass_event_number"]
    )[["matchId", "periodId", "pass_event_number"]]

    matched_positions = events_df.reset_index().merge(
        key_pass_keys,
        left_on=["matchId", "periodId", "eventNumber"],
        right_on=["matchId", "periodId", "pass_event_number"],
        how="inner",
    )["index"]

    flag = pd.Series(False, index=events_df.index)
    flag.loc[matched_positions] = True
    return flag


def flag_dangerous_crosses(events_df: pd.DataFrame, cross_mask: pd.Series) -> pd.Series:
    """Per-event boolean: cross followed by a high-xG team shot within 5s, same half.

    A cross (as defined by `cross_mask`) is dangerous when its attacking club
    records a SHOT with xG > `DANGEROUS_CROSS_MIN_XG` in the same match period
    within $(0, 5]$ seconds. The shooter is not restricted to the pass receiver,
    so flick-ons and second contacts that produce the chance are credited too.
    Each cross is flagged at most once regardless of how many shots follow.
    """
    crosses = events_df.loc[
        cross_mask, ["matchId", "periodId", "attackingClubId", "gameTimeInSec"]
    ].rename(columns={"gameTimeInSec": "cross_time_sec"})
    crosses["cross_index"] = crosses.index

    shots = events_df.loc[
        events_df["actionType"].eq("SHOT") & events_df["SHOT_XG"].gt(DANGEROUS_CROSS_MIN_XG),
        ["matchId", "periodId", "attackingClubId", "gameTimeInSec"],
    ].rename(columns={"gameTimeInSec": "shot_time_sec"})

    matched = crosses.merge(shots, on=["matchId", "periodId", "attackingClubId"], how="inner")
    elapsed_seconds = matched["shot_time_sec"] - matched["cross_time_sec"]
    matched = matched.loc[elapsed_seconds.gt(0) & elapsed_seconds.le(DANGEROUS_CROSS_WINDOW_SECONDS)]

    flag = pd.Series(False, index=events_df.index)
    flag.loc[matched["cross_index"].unique()] = True
    return flag


def engineer_flags(events_df: pd.DataFrame) -> pd.DataFrame:
    """Attach every precomputed classification/flag column described in the pipeline spec."""
    events_df = harmonize_action_types(events_df)
    action_type = events_df["actionType"]
    is_success = events_df["result"].eq("SUCCESS")

    start_in_box = _is_penalty_area(events_df["startAdjCoordinatesX"], events_df["startAdjCoordinatesY"])
    end_in_box = _is_penalty_area(events_df["endAdjCoordinatesX"], events_df["endAdjCoordinatesY"])
    
    # Euclidean distance to opponent's goal center (+PITCH_HALF_LENGTH, 0)
    dist_start_to_goal = np.sqrt(
        (PITCH_HALF_LENGTH - events_df["startAdjCoordinatesX"]) ** 2
        + (0.0 - events_df["startAdjCoordinatesY"]) ** 2
    )
    dist_end_to_goal = np.sqrt(
        (PITCH_HALF_LENGTH - events_df["endAdjCoordinatesX"]) ** 2
        + (0.0 - events_df["endAdjCoordinatesY"]) ** 2
    )
    distance_closer_to_goal = dist_start_to_goal - dist_end_to_goal

    # Pitch half segmentation in attack-normalized coordinates (X < 0 is own half)
    start_in_own_half = events_df["startAdjCoordinatesX"].lt(0)
    end_in_own_half = events_df["endAdjCoordinatesX"].lt(0)

    # Tiered distance thresholds: 30m (own->own), 15m (own->opp), 10m (opp->opp)
    prog_own_to_own = start_in_own_half & end_in_own_half & distance_closer_to_goal.ge(30.0)
    prog_own_to_opp = start_in_own_half & ~end_in_own_half & distance_closer_to_goal.ge(15.0)
    prog_opp_to_opp = ~start_in_own_half & ~end_in_own_half & distance_closer_to_goal.ge(10.0)
    is_progressive_distance = prog_own_to_own | prog_own_to_opp | prog_opp_to_opp

    events_df["shot_outcome"] = classify_shot_outcomes(events_df)
    events_df["is_key_pass"] = flag_key_passes(events_df)
    events_df["is_assist"] = events_df["ASSISTS"].notna()

    # Progressive pass and carry definitions
    events_df["is_progressive_pass"] = (
        action_type.eq("PASS") & is_progressive_distance
    )
    events_df["is_progressive_carry"] = (
        action_type.isin(DRIBBLE_VARIANT_ACTIONS) & is_progressive_distance
    )

    # Box entries, crosses, duels, and ball recoveries
    events_df["is_box_entry"] = (
        action_type.isin(BOX_ENTRY_ACTIONS) & ~start_in_box & end_in_box
    )
    # Successful pass from a non-central lane into the box that ends beyond the near (first) post.
    # Mirroring by the start side makes the rule identical for left and right crosses.
    start_y = events_df["startAdjCoordinatesY"]
    side = np.sign(start_y)
    passes_first_post = (side * events_df["endAdjCoordinatesY"]).le(GOAL_POST_ABS_Y)
    events_df["is_cross_into_box"] = (
        action_type.eq("PASS")
        #& events_df["result"].eq("SUCCESS")
        & events_df["startLane"].notna()
        & events_df["startLane"].ne("CENTER")
        & events_df["startPitchPosition"].eq("FINAL_THIRD")
        & ~start_in_box
        & end_in_box
        & passes_first_post
    )
    # Completed crosses into the box; crossing accuracy = sum(is_successful_cross) / sum(is_cross_into_box).
    events_df["is_successful_cross"] = events_df["is_cross_into_box"] & is_success
    events_df["is_dangerous_cross"] = flag_dangerous_crosses(
        events_df, events_df["is_cross_into_box"]
    )

    # Passes (crosses, cut-backs) and carries that start in a wing lane outside the box and end inside it.
    # Outcome-agnostic here; consumers gate on `result == SUCCESS` for the offensive quality metric.
    events_df["is_wide_lane_box_entry"] = events_df["is_box_entry"] & events_df["startLane"].isin(WING_LANES)

    events_df["is_ground_duel_won"] = events_df["WON_GROUND_DUELS"].notna()
    # Wyscout never populates LOST_GROUND_DUELS: every won duel names the beaten opponent in
    # `duelPlayerId`, so a player's losses are the won-duel rows that carry their id here.
    events_df["ground_duel_loser_id"] = events_df["duelPlayerId"].where(events_df["is_ground_duel_won"])
    events_df["is_duel_won"] = events_df["is_ground_duel_won"] | events_df["WON_AERIAL_DUELS"].notna()
    events_df["is_ball_recovery"] = (
        action_type.isin(["LOOSE_BALL_REGAIN", "INTERCEPTION"]) | events_df["BALL_WIN_NUMBER"].notna()
    )

    events_df["match_display"] = events_df["dateutc"].str.slice(0, 10) + " — " + events_df["label"]
    return events_df


def optimize_dtypes(events_df: pd.DataFrame) -> pd.DataFrame:
    """Downcast to compact dtypes so Streamlit's parquet reads stay snappy."""
    events_df = events_df.copy()

    for column in CATEGORY_COLUMNS:
        if column in events_df.columns:
            events_df[column] = events_df[column].astype("category")

    for column in DENSE_INT_COLUMNS:
        events_df[column] = pd.to_numeric(events_df[column], downcast="integer")

    for column in NULLABLE_INT_COLUMNS:
        events_df[column] = events_df[column].astype("Int32")

    for column in FLOAT32_COLUMNS:
        events_df[column] = events_df[column].astype("float32")

    events_df["periodId"] = pd.to_numeric(events_df["periodId"], downcast="integer")

    bool_columns = [column for column in events_df.columns if column.startswith("is_")]
    events_df[bool_columns] = events_df[bool_columns].astype(bool)

    return events_df


def main() -> None:
    """Load, enrich, optimize, and export the enriched events parquet."""
    print(f"Loading raw events from {EVENTS_PARQUET} ...")
    events_df = load_raw_events()
    print(f"Loaded {len(events_df):,} raw events, {len(events_df.columns)} columns.")

    events_df = engineer_flags(events_df)
    events_df = events_df.drop(columns=COLUMNS_TO_DROP_AFTER_ENGINEERING)
    events_df = optimize_dtypes(events_df)

    ENRICHED_EVENTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(ENRICHED_EVENTS_PARQUET, engine="pyarrow", compression="snappy", index=False)

    raw_size_mb = EVENTS_PARQUET.stat().st_size / 1_048_576
    enriched_size_mb = ENRICHED_EVENTS_PARQUET.stat().st_size / 1_048_576
    print(f"Wrote {len(events_df):,} rows x {len(events_df.columns)} columns -> {ENRICHED_EVENTS_PARQUET}")
    print(f"Raw source (all ~150 columns): {raw_size_mb:.1f} MB")
    print(f"Enriched output (curated + flags): {enriched_size_mb:.1f} MB")

    flag_columns = [column for column in events_df.columns if column.startswith("is_")] + ["shot_outcome"]
    print("\nFlag counts:")
    for column in flag_columns:
        if column == "shot_outcome":
            print(f"  {column}: {events_df[column].value_counts(dropna=True).to_dict()}")
        else:
            print(f"  {column}: {int(events_df[column].sum()):,}")


if __name__ == "__main__":
    main()
