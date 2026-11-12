"""Player evaluation metrics and role-specific performance indices.

The pipeline reconstructs eligible outfield player-season minutes from Wyscout
event data, engineers possession-adjusted per-90 quality metrics (successful
outcomes only), standardizes each metric within the player's tactical cluster,
and calculates weighted performance indices for the evaluated forward and
defensive roles. Every (seasonId, playerId) pair is an independent observation,
consistent with the player-season clusters exported by `player_profile_pipeline.py`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from enrich_events_pipeline import ENRICHED_EVENTS_PARQUET


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
CLUSTERS_CSV: Final[Path] = Path(
    os.getenv(
        "GOALUNIT_CLUSTERS_CSV",
        str(PROJECT_ROOT / "final_tactical_clusters.csv"),
    )
)
OUTPUT_CSV: Final[Path] = Path(
    os.getenv(
        "GOALUNIT_EVALUATION_OUTPUT_CSV",
        str(PROJECT_ROOT / "player_evaluation_metrics.csv"),
    )
)

DEFENSIVE_COMPARISON_CSV: Final[Path] = Path(
    os.getenv(
        "GOALUNIT_DEFENSIVE_COMPARISON_CSV",
        str(PROJECT_ROOT / "defensive_exponent_comparison.csv"),
    )
)

TARGET_SEASON_IDS: Final[list[int]] = [862, 1773]
MIN_MINUTES_THRESHOLD: Final[int] = 900
PRIMARY_HEADER_ACTIONS: Final[list[str]] = [
    "PASS",
    "SHOT",
    "BLOCK",
    "DRIBBLE",
    "CLEARANCE",
]
# Active ball-control actions; won duels are added separately regardless of actionType.
POSSESSION_ACTION_TYPES: Final[list[str]] = ["PASS", "DRIBBLE", "SHOT"]

PITCH_HALF_LENGTH: Final[float] = 52.5
PENALTY_AREA_DEPTH: Final[float] = 16.5
PENALTY_AREA_HALF_WIDTH: Final[float] = 20.16
PENALTY_AREA_X: Final[float] = PITCH_HALF_LENGTH - PENALTY_AREA_DEPTH
FINAL_THIRD_X: Final[float] = PITCH_HALF_LENGTH / 3

PLAYER_KEYS: Final[list[str]] = ["seasonId", "playerId"]
# Possession-adjusted metrics are re-expressed at this hypothetical possession baseline (%).
NEUTRAL_POSSESSION_PCT: Final[float] = 50.0

MINUTES_COLUMNS: Final[list[str]] = [
    "seasonId",
    "seasonName",
    "matchId",
    "periodId",
    "playerId",
    "gameTimeInSec",
    "actionType",
]
EVALUATION_COLUMNS: Final[list[str]] = [
    "seasonId",
    "matchId",
    "periodId",
    "eventNumber",
    "gameTimeInSec",
    "clubId",
    "attackingClubId",
    "playerId",
    "passReceiverPlayerId",
    "duelPlayerId",
    "actionType",
    "result",
    "bodyPart",
    "dribbleType",
    "dribbleResult",
    "SUCCESSFUL_PASSES",
    "SHOT_XG",
    "SHOT_AT_GOAL_NUMBER",
    "WON_AERIAL_DUELS",
    "WON_GROUND_DUELS",
    "dribbleDistance",
    "startCoordinatesX",
    "startCoordinatesY",
    "endCoordinatesX",
    "endCoordinatesY",
    "startPitchPosition",
    "endPitchPosition",
]
# Event-level flags precomputed by `enrich_events_pipeline.py`; the full-back
# metrics reuse them (restricted to successful outcomes) instead of re-deriving them.
ENRICHED_FLAG_COLUMNS: Final[list[str]] = [
    "seasonId",
    "playerId",
    "result",
    "is_progressive_pass",
    "is_progressive_carry",
    "is_cross_into_box",
    "is_successful_cross",
    "is_dangerous_cross",
    "is_ball_recovery",
    "is_ground_duel_won",
    "is_wide_lane_box_entry",
    "ground_duel_loser_id",
]
# Possession-adjusted (padj) variants drive clustering/index Z-scores; rate
# metrics stay unadjusted since they are already possession-independent.
METRIC_COLUMNS: Final[list[str]] = [
    "npxg_padj_p90",
    "shots_inside_box_padj_p90",
    "touches_inside_box_padj_p90",
    "headers_padj_p90",
    "aerial_duel_success_rate_pct",
    "progressive_carrying_distance_padj_p90",
    "dribbles_carries_into_penalty_area_padj_p90",
    "crosses_into_penalty_area_padj_p90",
    "dangerous_crosses_padj_p90",
    "cross_accuracy_pct",
    "take_ons_padj_p90",
    "successful_take_on_rate_pct",
    "shots_padj_p90",
    "passes_to_final_third_or_penalty_box_padj_p90",
    "key_passes_padj_p90",
    "progressive_carries_padj_p90",
    "progressive_passes_padj_p90",
    "ball_recoveries_padj_p90",
    "ground_duels_won_padj_p90",
    "wide_lane_box_entries_padj_p90",
    "ground_duel_win_rate_pct",
]
# Defensive volumes are scaled by (50 / opponent possession %) raised to this
# exponent: 0 = unadjusted, 1 = full, 0.5 = partial adjustment.
DEFENSIVE_PADJ_EXPONENT: Final[float] = 0.0
DEFENSIVE_PADJ_P90_COLUMNS: Final[frozenset[str]] = frozenset(
    {"ball_recoveries_p90", "ground_duels_won_p90"}
)
DEFENSIVE_EXPONENT_VARIANTS: Final[dict[str, float]] = {
    "unadjusted": 0.0,
    "partial": 0.5,
    "opponent_share": 1.0,
}
FULL_BACK_ROLE: Final[str] = "Progressive Fullbacks"
TARGET_MAN_ROLE: Final[str] = "Target Men"
WINGER_ROLE: Final[str] = "Wingers"  # carries the previous "Inside Forwards" metrics (placeholder)

# Hardcoded role -> K-Means cluster id. K-Means ids are arbitrary, so update this
# mapping by hand (matching CLUSTER_NAMES in the t-SNE plot scripts) after every
# re-run of `player_profile_pipeline.py`.
ROLE_CLUSTERS: Final[dict[str, int]] = {
    FULL_BACK_ROLE: 3,
    TARGET_MAN_ROLE: 8,
    WINGER_ROLE: 5,
}

FORWARD_ROLE_ZSCORE_COLUMNS: Final[dict[str, list[str]]] = {
    TARGET_MAN_ROLE: [
        "npxg_padj_p90_zscore",
        "shots_inside_box_padj_p90_zscore",
        "touches_inside_box_padj_p90_zscore",
        "headers_padj_p90_zscore",
        #"aerial_duel_success_rate_pct_zscore",
    ],
    WINGER_ROLE: [
        "npxg_padj_p90_zscore",
        "shots_padj_p90_zscore",
        "passes_to_final_third_or_penalty_box_padj_p90_zscore",
        "key_passes_padj_p90_zscore",
    ],
}

DEFENSIVE_ROLE_ZSCORE_COLUMNS: Final[dict[str, list[str]]] = {
    FULL_BACK_ROLE: [
        #"wide_lane_box_entries_padj_p90_zscore",
        "key_passes_padj_p90_zscore",
        "dangerous_crosses_padj_p90_zscore",
        "cross_accuracy_pct_zscore",
        "progressive_passes_padj_p90_zscore",
        "ball_recoveries_padj_p90_zscore",
        "ground_duel_win_rate_pct_zscore",
        #"progressive_carries_padj_p90_zscore",
    ],
}

# Explicitly defined weights for each Z-score column within a role's cluster.
FORWARD_ROLE_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    TARGET_MAN_ROLE: {
        "npxg_padj_p90_zscore": 0.45,
        "touches_inside_box_padj_p90_zscore": 0.15,
        "shots_inside_box_padj_p90_zscore": 0.30,
        "headers_padj_p90_zscore": 0.10,
        #"aerial_duel_success_rate_pct_zscore": 0.05,
    },
    WINGER_ROLE: {
        "npxg_padj_p90_zscore": 0.40,
        "shots_padj_p90_zscore": 0.20,
        "passes_to_final_third_or_penalty_box_padj_p90_zscore": 0.15,
        "key_passes_padj_p90_zscore": 0.25,
    },
}

DEFENSIVE_ROLE_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    FULL_BACK_ROLE: {
        "dangerous_crosses_padj_p90_zscore": 0.2,
        "cross_accuracy_pct_zscore": 0.2,
        #"wide_lane_box_entries_padj_p90_zscore": 0.15,
        "progressive_passes_padj_p90_zscore": 0.15,
        "key_passes_padj_p90_zscore": 0.15,
        #"progressive_carries_padj_p90_zscore": 0.2,

        # Defensive contributions 30%
        "ball_recoveries_padj_p90_zscore": 0.2,
        "ground_duel_win_rate_pct_zscore": 0.1,
    },
}

ROLE_ZSCORE_COLUMNS: Final[dict[str, list[str]]] = {
    **FORWARD_ROLE_ZSCORE_COLUMNS,
    **DEFENSIVE_ROLE_ZSCORE_COLUMNS,
}
ROLE_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    **FORWARD_ROLE_WEIGHTS,
    **DEFENSIVE_ROLE_WEIGHTS,
}


def _validate_input_paths() -> None:
    """Validate source-data locations before executing the pipeline."""
    missing_paths = [
        path
        for path in (EVENTS_PARQUET, ENRICHED_EVENTS_PARQUET, CLUSTERS_CSV)
        if not path.exists()
    ]
    if missing_paths:
        formatted_paths = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required input files were not found: {formatted_paths}")


def load_minutes_events(season_ids: Sequence[int]) -> pd.DataFrame:
    """Load event columns needed to reproduce playing-time eligibility."""
    _validate_input_paths()
    events_df = pd.read_parquet(EVENTS_PARQUET, columns=MINUTES_COLUMNS)
    return events_df.loc[events_df["seasonId"].isin(season_ids)].copy()


def load_evaluation_events(season_ids: Sequence[int]) -> pd.DataFrame:
    """Load event columns required for evaluation-metric engineering."""
    _validate_input_paths()
    events_df = pd.read_parquet(EVENTS_PARQUET, columns=EVALUATION_COLUMNS)
    return events_df.loc[events_df["seasonId"].isin(season_ids)].copy()


def load_enriched_flag_events(season_ids: Sequence[int]) -> pd.DataFrame:
    """Load the precomputed event flags produced by `enrich_events_pipeline.py`."""
    _validate_input_paths()
    events_df = pd.read_parquet(ENRICHED_EVENTS_PARQUET, columns=ENRICHED_FLAG_COLUMNS)
    return events_df.loc[events_df["seasonId"].isin(season_ids)].copy()


def identify_goalkeepers(events_df: pd.DataFrame) -> np.ndarray:
    """Return IDs of players with at least one goalkeeper-specific action."""
    return events_df.loc[
        events_df["actionType"].str.startswith("GK_", na=False), "playerId"
    ].dropna().unique()


def build_continuous_timeline(events_df: pd.DataFrame) -> pd.DataFrame:
    """Replace Wyscout's second-half offset with each match's first-half length."""
    h1_end_times = (
        events_df.loc[events_df["periodId"].eq(1)]
        .groupby("matchId", as_index=False)["gameTimeInSec"]
        .max()
        .rename(columns={"gameTimeInSec": "h1_duration"})
    )
    timeline_df = events_df.merge(h1_end_times, on="matchId", how="left")
    timeline_df["continuous_sec"] = np.select(
        [timeline_df["periodId"].eq(1), timeline_df["periodId"].eq(2)],
        [
            timeline_df["gameTimeInSec"],
            timeline_df["gameTimeInSec"] - 10_000 + timeline_df["h1_duration"],
        ],
        default=timeline_df["gameTimeInSec"],
    )
    return timeline_df


def compute_match_windows(events_df: pd.DataFrame) -> pd.DataFrame:
    """Return each player's first-to-last continuous-clock action per match."""
    return events_df.groupby(["seasonId", "matchId", "playerId"], as_index=False).agg(
        first_action_sec=("continuous_sec", "min"),
        last_action_sec=("continuous_sec", "max"),
    )


def compute_player_minutes(
    events_df: pd.DataFrame,
    goalkeeper_ids: np.ndarray,
    min_minutes: int,
) -> pd.DataFrame:
    """Calculate eligible outfield player-season minutes using continuous event clocks."""
    match_minutes_df = compute_match_windows(events_df)
    match_minutes_df["match_minutes"] = (
        match_minutes_df["last_action_sec"] - match_minutes_df["first_action_sec"]
    ) / 60
    player_minutes_df = (
        match_minutes_df.groupby(PLAYER_KEYS, as_index=False)["match_minutes"]
        .sum()
        .rename(columns={"match_minutes": "total_minutes"})
    )
    return player_minutes_df.loc[
        (player_minutes_df["total_minutes"] > min_minutes)
        & (~player_minutes_df["playerId"].isin(goalkeeper_ids))
    ].copy()


def build_possession_shares(events_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate each player-season's on-pitch possession share.

    For every match window (a player's first-to-last continuous-clock
    action), possession is the player's own club's share of active
    ball-control events (passes, dribbles, shots, and won duels) among both
    clubs during that window. Event counts are summed across matches before
    dividing, so matches with more recorded possession events carry
    proportionally more weight in the season-level share. `events_df` must
    already carry a `continuous_sec` column from `build_continuous_timeline`.
    """
    match_windows_df = compute_match_windows(events_df)
    player_clubs_df = (
        events_df.dropna(subset=["clubId"])
        .groupby(["matchId", "playerId"], as_index=False)["clubId"]
        .first()
    )
    match_windows_df = match_windows_df.merge(
        player_clubs_df, on=["matchId", "playerId"], how="inner"
    )

    possession_event_mask = (
        events_df["actionType"].isin(POSSESSION_ACTION_TYPES)
        | events_df["WON_GROUND_DUELS"].fillna(0).gt(0)
        | events_df["WON_AERIAL_DUELS"].fillna(0).gt(0)
    )
    possession_events_df = (
        events_df.loc[possession_event_mask, ["matchId", "continuous_sec", "attackingClubId"]]
        .dropna(subset=["attackingClubId"])
        .sort_values(["matchId", "continuous_sec"])
    )

    player_id_batches: list[np.ndarray] = []
    season_id_batches: list[np.ndarray] = []
    own_event_batches: list[np.ndarray] = []
    total_event_batches: list[np.ndarray] = []

    for match_id, match_events in possession_events_df.groupby("matchId", sort=False):
        match_windows = match_windows_df.loc[match_windows_df["matchId"].eq(match_id)]
        if match_windows.empty:
            continue

        event_times = match_events["continuous_sec"].to_numpy()
        club_values = match_events["attackingClubId"].to_numpy()
        club_cumulative_counts = {
            club: np.cumsum(club_values == club) for club in np.unique(club_values)
        }

        left_indices = np.searchsorted(
            event_times, match_windows["first_action_sec"].to_numpy(), side="left"
        )
        right_indices = np.searchsorted(
            event_times, match_windows["last_action_sec"].to_numpy(), side="right"
        )
        total_events = (right_indices - left_indices).astype(float)

        own_events = np.zeros(len(match_windows), dtype=float)
        own_clubs = match_windows["clubId"].to_numpy()
        for row_index, own_club in enumerate(own_clubs):
            cumulative_counts = club_cumulative_counts.get(own_club)
            if cumulative_counts is None:
                continue
            right_index = right_indices[row_index]
            left_index = left_indices[row_index]
            total_at_right = cumulative_counts[right_index - 1] if right_index > 0 else 0
            total_at_left = cumulative_counts[left_index - 1] if left_index > 0 else 0
            own_events[row_index] = total_at_right - total_at_left

        season_id_batches.append(match_windows["seasonId"].to_numpy())
        player_id_batches.append(match_windows["playerId"].to_numpy())
        own_event_batches.append(own_events)
        total_event_batches.append(total_events)

    if not player_id_batches:
        return pd.DataFrame(columns=[*PLAYER_KEYS, "possession_share"])

    possession_df = pd.DataFrame(
        {
            "seasonId": np.concatenate(season_id_batches),
            "playerId": np.concatenate(player_id_batches),
            "own_possession_events": np.concatenate(own_event_batches),
            "total_possession_events": np.concatenate(total_event_batches),
        }
    )
    season_possession_df = possession_df.groupby(PLAYER_KEYS, as_index=False)[
        ["own_possession_events", "total_possession_events"]
    ].sum()
    season_possession_df["possession_share"] = np.divide(
        season_possession_df["own_possession_events"],
        season_possession_df["total_possession_events"],
        out=np.full(len(season_possession_df), np.nan),
        where=season_possession_df["total_possession_events"].gt(0),
    )
    return season_possession_df.loc[:, [*PLAYER_KEYS, "possession_share"]]


def possession_multiplier(possession_pct: pd.Series) -> pd.Series:
    """Return `NEUTRAL_POSSESSION_PCT / possession_pct`, or 1.0 where the percentage is not positive.

    Multiplying a raw metric by this factor re-expresses it at a hypothetical
    50% possession baseline. Pass the team's possession % for attacking
    metrics and the opponent's (100 - team %) for defensive metrics.
    """
    percentages = possession_pct.to_numpy(dtype=float)
    multiplier = np.divide(
        NEUTRAL_POSSESSION_PCT,
        percentages,
        out=np.ones_like(percentages),
        where=percentages > 0,
    )
    return pd.Series(multiplier, index=possession_pct.index)


def _is_penalty_area(coordinate_x: pd.Series, coordinate_y: pd.Series) -> pd.Series:
    """Identify the attacking penalty area in normalized $105 x 68$ coordinates."""
    return coordinate_x.ge(PENALTY_AREA_X) & coordinate_y.abs().le(
        PENALTY_AREA_HALF_WIDTH
    )


def _sum_by_player(
    events_df: pd.DataFrame,
    event_mask: pd.Series,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Aggregate a numeric event value per player-season for rows satisfying a mask."""
    return (
        events_df.loc[event_mask]
        .groupby(PLAYER_KEYS, as_index=False)[value_column]
        .sum()
        .rename(columns={value_column: output_column})
    )


def _count_by_player(
    events_df: pd.DataFrame,
    event_mask: pd.Series,
    output_column: str,
) -> pd.DataFrame:
    """Count qualifying events per player-season."""
    return (
        events_df.loc[event_mask]
        .groupby(PLAYER_KEYS, as_index=False)
        .size()
        .rename(columns={"size": output_column})
    )


def build_key_passes(events_df: pd.DataFrame) -> pd.DataFrame:
    """Count successful passes whose receiver shoots within five seconds.

    A completed pass is credited once when its named receiver produces a shot
    for the same attacking team, in the same match period, within $(0, 5]$
    seconds. This captures post-reception carries and dribbles before a shot.
    """
    successful_passes_df = events_df.loc[
        events_df["actionType"].eq("PASS")
        & events_df["SUCCESSFUL_PASSES"].fillna(0).eq(1),
        [
            "seasonId",
            "matchId",
            "periodId",
            "eventNumber",
            "gameTimeInSec",
            "attackingClubId",
            "playerId",
            "passReceiverPlayerId",
        ],
    ].rename(
        columns={
            "eventNumber": "pass_event_number",
            "gameTimeInSec": "pass_time_sec",
        }
    )
    shots_df = events_df.loc[
        events_df["actionType"].eq("SHOT"),
        [
            "matchId",
            "periodId",
            "eventNumber",
            "gameTimeInSec",
            "attackingClubId",
            "playerId",
        ],
    ].rename(
        columns={
            "eventNumber": "shot_event_number",
            "gameTimeInSec": "shot_time_sec",
            "playerId": "shooter_id",
        }
    )
    key_passes_df = successful_passes_df.merge(
        shots_df,
        left_on=[
            "matchId",
            "periodId",
            "attackingClubId",
            "passReceiverPlayerId",
        ],
        right_on=[
            "matchId",
            "periodId",
            "attackingClubId",
            "shooter_id",
        ],
        how="inner",
    )
    key_passes_df["elapsed_seconds"] = (
        key_passes_df["shot_time_sec"] - key_passes_df["pass_time_sec"]
    )
    key_passes_df = key_passes_df.loc[
        key_passes_df["elapsed_seconds"].gt(0)
        & key_passes_df["elapsed_seconds"].le(5)
    ].drop_duplicates(subset=["matchId", "periodId", "pass_event_number"])
    return _count_by_player(
        key_passes_df,
        pd.Series(True, index=key_passes_df.index),
        "key_passes",
    )


def build_enriched_flag_counts(flag_events_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Count successful full-back actions per player-season from the enriched flags.

    Progressive passes/carries, crosses, dangerous crosses and wide-lane box
    entries require `result == SUCCESS` so the metrics reward execution quality
    rather than attempt volume. Ball recoveries and won ground duels are positive
    outcomes by construction; lost ground duels feed the duel win rate.
    """
    is_success = flag_events_df["result"].eq("SUCCESS")
    flag_metrics = {
        "progressive_passes": flag_events_df["is_progressive_pass"] & is_success,
        "progressive_carries": flag_events_df["is_progressive_carry"] & is_success,
        "crosses_attempted": flag_events_df["is_cross_into_box"],
        "crosses_into_penalty_area": flag_events_df["is_successful_cross"],
        "dangerous_crosses": flag_events_df["is_dangerous_cross"] & is_success,
        "wide_lane_box_entries": flag_events_df["is_wide_lane_box_entry"] & is_success,
        "ball_recoveries": flag_events_df["is_ball_recovery"],
        "ground_duels_won": flag_events_df["is_ground_duel_won"],
    }
    count_frames = [
        _count_by_player(flag_events_df, event_mask, output_column)
        for output_column, event_mask in flag_metrics.items()
    ]
    # Ground-duel losses live on the opponents' won-duel rows (LOST_GROUND_DUELS is never populated).
    ground_duel_losses_df = (
        flag_events_df.loc[flag_events_df["ground_duel_loser_id"].notna(), ["seasonId", "ground_duel_loser_id"]]
        .astype({"ground_duel_loser_id": "int64"})
        .groupby(PLAYER_KEYS[:1] + ["ground_duel_loser_id"], as_index=False)
        .size()
        .rename(columns={"ground_duel_loser_id": "playerId", "size": "ground_duels_lost"})
    )
    return [*count_frames, ground_duel_losses_df]


def engineer_player_metrics(
    events_df: pd.DataFrame,
    player_minutes_df: pd.DataFrame,
    flag_events_df: pd.DataFrame,
    defensive_exponent: float = DEFENSIVE_PADJ_EXPONENT,
) -> pd.DataFrame:
    """Engineer requested evaluation metrics and normalize volumetric values per 90.

    `defensive_exponent` controls the possession adjustment of defensive metrics:
    0 leaves them unadjusted, 1 divides by the opponent share (1 - share), and
    values in between apply a partial adjustment.
    """
    # Possession adjustment needs a continuous clock to bound player windows.
    events_df = build_continuous_timeline(events_df)
    action_type = events_df["actionType"]

    # Using atol=1e-4 provides safe floating-point matching for 0.77941
    is_penalty_shot = action_type.eq("SHOT") & np.isclose(
        events_df["SHOT_XG"].fillna(0), 0.77941, atol=1e-4
    )

    successful_pass = action_type.eq("PASS") & events_df[
        "SUCCESSFUL_PASSES"
    ].fillna(0).eq(1)
    shot_mask = action_type.eq("SHOT") & ~is_penalty_shot
    dribble_mask = action_type.eq("DRIBBLE")
    start_in_penalty_area = _is_penalty_area(
        events_df["startCoordinatesX"], events_df["startCoordinatesY"]
    )
    end_in_penalty_area = _is_penalty_area(
        events_df["endCoordinatesX"], events_df["endCoordinatesY"]
    )

    shots_inside_box = shot_mask & start_in_penalty_area
    headers = events_df["bodyPart"].eq("HEAD") & action_type.isin(
        PRIMARY_HEADER_ACTIONS
    )
    progressive_carries = (
        dribble_mask
        & events_df["dribbleType"].eq("PROGRESSIVE_CARRY")
        & events_df["result"].eq("SUCCESS")
    )
    penalty_area_entries = (
        dribble_mask
        & events_df["result"].eq("SUCCESS")
        & ~start_in_penalty_area
        & end_in_penalty_area
    )
    final_third_entries = (
        events_df["startCoordinatesX"].lt(FINAL_THIRD_X)
        & events_df["endCoordinatesX"].ge(FINAL_THIRD_X)
    )
    passes_into_final_third_or_box = successful_pass & (
        final_third_entries | (~start_in_penalty_area & end_in_penalty_area)
    )
    take_ons = dribble_mask & events_df["dribbleType"].eq("ONE_VS_ONE")

    # Double-gated: Must have the OPPONENT_BOX tag AND be in the positive attacking pitch half
    touches_in_opponent_box_mask = (
        events_df["startPitchPosition"].eq("OPPONENT_BOX") & start_in_penalty_area
    )

    metric_frames = [
        _sum_by_player(events_df, shot_mask, "SHOT_XG", "npxg"),
        _count_by_player(events_df, shots_inside_box, "shots_inside_box"),
        _count_by_player(events_df, touches_in_opponent_box_mask, "touches_inside_box"),
        _count_by_player(events_df, headers, "headers"),
        _count_by_player(
            events_df,
            action_type.eq("DRIBBLE")
            & events_df["dribbleType"].eq("ONE_VS_ONE")
            & events_df["dribbleResult"].eq("WINNER"),
            "successful_take_ons",
        ),
        _count_by_player(events_df, take_ons, "total_take_ons"),
        _sum_by_player(
            events_df,
            progressive_carries,
            "dribbleDistance",
            "progressive_carrying_distance",
        ),
        _count_by_player(
            events_df,
            penalty_area_entries,
            "dribbles_carries_into_penalty_area",
        ),
        *build_enriched_flag_counts(flag_events_df),
        _sum_by_player(events_df, shot_mask, "SHOT_AT_GOAL_NUMBER", "shots"),
        _count_by_player(
            events_df,
            passes_into_final_third_or_box,
            "passes_to_final_third_or_penalty_box",
        ),
        build_key_passes(events_df),
    ]

    aerial_wins_df = _sum_by_player(
        events_df,
        pd.Series(True, index=events_df.index),
        "WON_AERIAL_DUELS",
        "won_aerial_duels",
    )
    aerial_losses_df = (
        events_df.loc[
            events_df["duelPlayerId"].notna()
            & events_df["WON_AERIAL_DUELS"].fillna(0).gt(0),
            ["seasonId", "duelPlayerId", "WON_AERIAL_DUELS"],
        ]
        .groupby(["seasonId", "duelPlayerId"], as_index=False)["WON_AERIAL_DUELS"]
        .sum()
        .rename(
            columns={
                "duelPlayerId": "playerId",
                "WON_AERIAL_DUELS": "lost_aerial_duels",
            }
        )
    )
    metric_frames.extend([aerial_wins_df, aerial_losses_df])

    metrics_df = player_minutes_df.copy()
    for metric_frame in metric_frames:
        metrics_df = metrics_df.merge(metric_frame, on=PLAYER_KEYS, how="left")

    raw_metric_columns = [
        "npxg",
        "shots_inside_box",
        "touches_inside_box",
        "headers",
        "successful_take_ons",
        "total_take_ons",
        "progressive_carrying_distance",
        "dribbles_carries_into_penalty_area",
        "crosses_attempted",
        "crosses_into_penalty_area",
        "dangerous_crosses",
        "shots",
        "passes_to_final_third_or_penalty_box",
        "key_passes",
        "progressive_carries",
        "progressive_passes",
        "ball_recoveries",
        "ground_duels_won",
        "ground_duels_lost",
        "wide_lane_box_entries",
        "won_aerial_duels",
        "lost_aerial_duels",
    ]
    metrics_df[raw_metric_columns] = metrics_df[raw_metric_columns].fillna(0)
    per90_multiplier = 90 / metrics_df["total_minutes"]
    per90_columns = {
        "npxg": "npxg_p90",
        "shots_inside_box": "shots_inside_box_p90",
        "touches_inside_box" : "touches_inside_box_p90",
        "headers": "headers_p90",
        "progressive_carrying_distance": "progressive_carrying_distance_p90",
        "total_take_ons": "take_ons_p90",
        "dribbles_carries_into_penalty_area": (
            "dribbles_carries_into_penalty_area_p90"
        ),
        "crosses_into_penalty_area": "crosses_into_penalty_area_p90",
        "dangerous_crosses": "dangerous_crosses_p90",
        "shots": "shots_p90",
        "passes_to_final_third_or_penalty_box": (
            "passes_to_final_third_or_penalty_box_p90"
        ),
        "key_passes": "key_passes_p90",
        "progressive_carries": "progressive_carries_p90",
        "progressive_passes": "progressive_passes_p90",
        "ball_recoveries": "ball_recoveries_p90",
        "ground_duels_won": "ground_duels_won_p90",
        "wide_lane_box_entries": "wide_lane_box_entries_p90",
    }
    for raw_column, per90_column in per90_columns.items():
        metrics_df[per90_column] = metrics_df[raw_column] * per90_multiplier

    # Possession adjustment: re-express volumetric per-90 metrics at a 50%
    # possession baseline, PAdj = raw * (50 / team possession %), so low-possession
    # teams are not penalized relative to dominant-possession teams. A missing
    # share defaults to the 50% baseline (multiplier 1).
    possession_shares_df = build_possession_shares(events_df)
    metrics_df = metrics_df.merge(possession_shares_df, on=PLAYER_KEYS, how="left")
    metrics_df["possession_share"] = metrics_df["possession_share"].fillna(
        NEUTRAL_POSSESSION_PCT / 100
    )
    metrics_df["possession_share_pct"] = metrics_df["possession_share"] * 100

    # Defensive actions scale with the opponent's possession, so they use
    # (50 / opponent possession %) ** exponent instead of the team's own share.
    attacking_multiplier = possession_multiplier(metrics_df["possession_share_pct"])
    defensive_multiplier = np.power(
        possession_multiplier(100.0 - metrics_df["possession_share_pct"]),
        defensive_exponent,
    )
    for per90_column in per90_columns.values():
        padj_column = per90_column.replace("_p90", "_padj_p90")
        multiplier = (
            defensive_multiplier
            if per90_column in DEFENSIVE_PADJ_P90_COLUMNS
            else attacking_multiplier
        )
        metrics_df[padj_column] = metrics_df[per90_column] * multiplier

    total_aerial_duels = (
        metrics_df["won_aerial_duels"] + metrics_df["lost_aerial_duels"]
    )
    metrics_df["aerial_duel_success_rate_pct"] = np.divide(
        metrics_df["won_aerial_duels"] * 100,
        total_aerial_duels,
        out=np.zeros(len(metrics_df), dtype=float),
        where=total_aerial_duels.gt(0),
    )
    # Rate metric: possession-independent, so it is left unadjusted like the aerial rate.
    total_ground_duels = metrics_df["ground_duels_won"] + metrics_df["ground_duels_lost"]
    metrics_df["ground_duel_win_rate_pct"] = np.divide(
        metrics_df["ground_duels_won"] * 100,
        total_ground_duels,
        out=np.zeros(len(metrics_df), dtype=float),
        where=total_ground_duels.gt(0),
    )
    metrics_df["ground_duels_total"] = total_ground_duels
    # Rate metric: completed crosses into the box over all attempted ones (possession-independent).
    metrics_df["cross_accuracy_pct"] = np.divide(
        metrics_df["crosses_into_penalty_area"] * 100,
        metrics_df["crosses_attempted"],
        out=np.zeros(len(metrics_df), dtype=float),
        where=metrics_df["crosses_attempted"].gt(0),
    )
    metrics_df["successful_take_on_rate_pct"] = np.divide(
        metrics_df["successful_take_ons"] * 100,
        metrics_df["total_take_ons"],
        out=np.zeros(len(metrics_df), dtype=float),
        where=metrics_df["total_take_ons"].gt(0),
    )
    raw_p90_columns = list(per90_columns.values())
    return metrics_df.loc[
        :,
        [
            *PLAYER_KEYS,
            "total_minutes",
            "possession_share_pct",
            *METRIC_COLUMNS,
            "ground_duels_total",
            *raw_p90_columns,
        ],
    ].copy()


def _zscore(values: pd.Series) -> pd.Series:
    """Return population Z-scores, using zero where a comparison group is constant."""
    standard_deviation = values.std(ddof=0)
    if standard_deviation == 0 or pd.isna(standard_deviation):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / standard_deviation


def add_cluster_zscores(player_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add metric Z-scores evaluated only against each player-season's tactical peers."""
    standardized_df = player_metrics_df.copy()
    for metric_column in METRIC_COLUMNS:
        zscore_column = f"{metric_column}_zscore"
        standardized_df[zscore_column] = standardized_df.groupby(
            "tactical_cluster"
        )[metric_column].transform(_zscore)
    return standardized_df


def calculate_performance_index(
    player_metrics_df: pd.DataFrame,
    target_cluster: int,
    zscore_columns: Sequence[str],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Calculate a normalized weighted index for one tactical-cluster population.

    Weights are normalized to sum to one, so equal weights produce the mean of
    the selected Z-scores and future weighting revisions preserve index scale.
    """
    missing_columns = set(zscore_columns).difference(player_metrics_df.columns)
    if missing_columns:
        raise KeyError(f"Missing Z-score columns: {sorted(missing_columns)}")
    if set(zscore_columns) != set(weights):
        raise ValueError("Weights must contain exactly one entry per Z-score column.")

    raw_weights = pd.Series(weights, dtype=float).reindex(zscore_columns)
    if raw_weights.isna().any() or raw_weights.sum() == 0:
        raise ValueError("Weights must be numeric and sum to a non-zero value.")

    cluster_metrics_df = player_metrics_df.loc[
        player_metrics_df["tactical_cluster"].eq(target_cluster)
    ].copy()
    
    # 1. Calculate the raw weighted index
    cluster_metrics_df["performance_index_raw"] = cluster_metrics_df.loc[
        :, zscore_columns
    ].mul(raw_weights / raw_weights.sum(), axis=1).sum(axis=1)

    # 2. Standardize the raw index so it has a mean of 0 and std of 1.0
    raw_mean = cluster_metrics_df["performance_index_raw"].mean()
    raw_std = cluster_metrics_df["performance_index_raw"].std(ddof=0)
    
    if raw_std == 0 or pd.isna(raw_std):
        z_index = pd.Series(0.0, index=cluster_metrics_df.index)
    else:
        z_index = (cluster_metrics_df["performance_index_raw"] - raw_mean) / raw_std

    # 3. Apply the FIFA-style linear transformation
    target_mean = 72.0
    target_std = 8.0
    
    cluster_metrics_df["performance_index"] = (z_index * target_std) + target_mean

    # 4. Clip the ratings to strictly enforce realistic boundaries (e.g., min 40, max 99)
    cluster_metrics_df["performance_index"] = cluster_metrics_df[
        "performance_index"
    ].clip(lower=40.0, upper=99.0)

    # 5. Rank the players
    cluster_metrics_df["performance_rank"] = cluster_metrics_df[
        "performance_index"
    ].rank(method="dense", ascending=False).astype(int)
    
    return cluster_metrics_df


def add_role_performance_indices(
    player_metrics_df: pd.DataFrame,
    role_clusters: Mapping[str, int],
) -> pd.DataFrame:
    """Apply weighted role-specific indices to every evaluated tactical cluster."""
    indexed_df = player_metrics_df.copy()
    indexed_df["evaluation_role"] = indexed_df["tactical_cluster"].map(
        {cluster: role for role, cluster in role_clusters.items()}
    )
    indexed_df["performance_index"] = np.nan
    indexed_df["performance_rank"] = pd.Series(pd.NA, index=indexed_df.index, dtype="Int64")

    for role, zscore_columns in ROLE_ZSCORE_COLUMNS.items():
        role_index_df = calculate_performance_index(
            indexed_df,
            role_clusters[role],
            zscore_columns,
            ROLE_WEIGHTS[role],
        )
        indexed_df.loc[role_index_df.index, "performance_index"] = role_index_df[
            "performance_index"
        ]
        indexed_df.loc[role_index_df.index, "performance_rank"] = role_index_df[
            "performance_rank"
        ].astype("Int64")
    return indexed_df


def build_evaluation_output(
    cluster_assignments_df: pd.DataFrame,
    player_metrics_df: pd.DataFrame,
    role_clusters: Mapping[str, int],
) -> pd.DataFrame:
    """Join cluster labels to metrics, then add peer Z-scores and role indices."""
    identity_columns = [
        *PLAYER_KEYS,
        "seasonName",
        "playerName",
        "clubName",
        "tactical_cluster",
    ]
    output_df = cluster_assignments_df.loc[:, identity_columns].merge(
        player_metrics_df,
        on=PLAYER_KEYS,
        how="inner",
        validate="one_to_one",
    )
    output_df = add_cluster_zscores(output_df)
    return add_role_performance_indices(output_df, role_clusters)


def compare_defensive_adjustments(
    evaluation_events_df: pd.DataFrame,
    player_minutes_df: pd.DataFrame,
    flag_events_df: pd.DataFrame,
    cluster_assignments_df: pd.DataFrame,
    role_clusters: Mapping[str, int],
) -> pd.DataFrame:
    """Compare full-back possession correlations and index ranks across adjustment variants.

    Reports, per variant, the Pearson correlation of each defensive metric with
    possession share and the Spearman rank correlation of the performance
    index against the unadjusted variant.
    """
    variant_frames: dict[str, pd.DataFrame] = {}
    for name, exponent in DEFENSIVE_EXPONENT_VARIANTS.items():
        metrics_df = engineer_player_metrics(
            evaluation_events_df, player_minutes_df, flag_events_df, exponent
        )
        output_df = build_evaluation_output(cluster_assignments_df, metrics_df, role_clusters)
        variant_frames[name] = output_df.loc[
            output_df["tactical_cluster"].eq(role_clusters[FULL_BACK_ROLE])
        ].set_index(
            PLAYER_KEYS
        )

    baseline_index = variant_frames["unadjusted"]["performance_index"]
    rows = []
    for name, frame in variant_frames.items():
        rows.append(
            {
                "variant": name,
                "exponent": DEFENSIVE_EXPONENT_VARIANTS[name],
                "corr_possession_recoveries": frame["possession_share_pct"].corr(
                    frame["ball_recoveries_padj_p90"]
                ),
                "corr_possession_ground_duels": frame["possession_share_pct"].corr(
                    frame["ground_duels_won_padj_p90"]
                ),
                "index_spearman_vs_unadjusted": baseline_index.corr(
                    frame["performance_index"].reindex(baseline_index.index),
                    method="spearman",
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run metric engineering, cluster-peer standardization, and CSV export."""
    minutes_events_df = load_minutes_events(TARGET_SEASON_IDS)
    goalkeeper_ids = identify_goalkeepers(minutes_events_df)
    player_minutes_df = compute_player_minutes(
        build_continuous_timeline(minutes_events_df),
        goalkeeper_ids,
        MIN_MINUTES_THRESHOLD,
    )
    evaluation_events_df = load_evaluation_events(TARGET_SEASON_IDS)
    flag_events_df = load_enriched_flag_events(TARGET_SEASON_IDS)
    cluster_assignments_df = pd.read_csv(CLUSTERS_CSV)
    role_clusters = ROLE_CLUSTERS
    print(f"Evaluated role clusters: {role_clusters}")
    if "--compare-defensive" in sys.argv:
        comparison_df = compare_defensive_adjustments(
            evaluation_events_df,
            player_minutes_df,
            flag_events_df,
            cluster_assignments_df,
            role_clusters,
        )
        print(comparison_df.to_string(index=False))
        comparison_df.to_csv(DEFENSIVE_COMPARISON_CSV, index=False)
        print(f"Exported defensive exponent comparison to {DEFENSIVE_COMPARISON_CSV}")
        return
    player_metrics_df = engineer_player_metrics(
        evaluation_events_df,
        player_minutes_df,
        flag_events_df,
    )
    output_df = build_evaluation_output(cluster_assignments_df, player_metrics_df, role_clusters)
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(output_df):,} player-seasons to {OUTPUT_CSV}")

    # --- DIAGNOSTIC: Top Players Output ---
    print("\n--- DIAGNOSTICS: Top Players for Touches in Box p90 ---")
    diagnostic_columns = ["playerName", "seasonName", "evaluation_role", "touches_inside_box_p90"]
    top_touches = output_df.sort_values(by="touches_inside_box_p90", ascending=False).head(5)
    print(top_touches[diagnostic_columns])
    print("-------------------------------------------------------\n")

    # --- DIAGNOSTIC: Top Attacking Full Backs ---
    print(f"\n--- DIAGNOSTICS: Top {FULL_BACK_ROLE} by Performance Index ---")
    full_back_columns = [
        "playerName",
        "seasonName",
        "performance_index",
        *[column.removesuffix("_zscore") for column in DEFENSIVE_ROLE_ZSCORE_COLUMNS[FULL_BACK_ROLE]],
    ]
    full_backs = output_df.loc[output_df["tactical_cluster"].eq(role_clusters[FULL_BACK_ROLE])]
    print(full_backs.sort_values(by="performance_index", ascending=False).head(10)[full_back_columns])
    print("-------------------------------------------------------\n")

    # --- DIAGNOSTIC: Possession Adjustment (Raw vs. PAdj) for Evaluated Roles ---
    print("\n--- DIAGNOSTICS: Possession Adjustment for Evaluated Roles ---")
    evaluated_rows = output_df.loc[output_df["evaluation_role"].notna()].copy()
    possession_columns = [
        "playerName",
        "seasonName",
        "evaluation_role",
        "possession_share_pct",
        "npxg_p90",
        "npxg_padj_p90",
        "key_passes_p90",
        "key_passes_padj_p90",
    ]
    print("Highest on-pitch possession share (e.g. dominant-possession clubs):")
    print(
        evaluated_rows.sort_values(by="possession_share_pct", ascending=False)
        .head(5)[possession_columns]
    )
    print("\nLowest on-pitch possession share (e.g. lower-possession clubs):")
    print(
        evaluated_rows.sort_values(by="possession_share_pct", ascending=True)
        .head(5)[possession_columns]
    )
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    main()