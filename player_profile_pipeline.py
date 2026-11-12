# %%
"""Player Profiling Pipeline.

Tactical archetype discovery for outfield players from Wyscout event data
(2023/24 & 2024/25 seasons). Every (seasonId, playerId) pair is an independent
entity: minutes, features, and cluster assignment are computed per player-season.

Phase 1 - Playstyle Fingerprint: continuous match-timeline reconstruction,
per-90 volumetric feature engineering, and spatial mapping.
Phase 2 - Tactical Classification: PCA dimensionality reduction and K-Means
clustering into tactical archetypes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

from player_evaluation_pipeline import build_possession_shares, possession_multiplier

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

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
ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"
_load_local_env_file(ENV_FILE)

DATA_DIR: Final[Path] = Path(os.getenv("GOALUNIT_DATA_DIR", str(PROJECT_ROOT / "data")))
EVENTS_PARQUET: Final[Path] = Path(
    os.getenv("GOALUNIT_EVENTS_PARQUET", str(DATA_DIR / "events.parquet"))
)

SEASON_ID_23_24: Final[int] = 862  # 2023/2024
SEASON_ID_24_25: Final[int] = 1773  # 2024/2025
TARGET_SEASON_IDS: Final[list[int]] = [SEASON_ID_23_24, SEASON_ID_24_25]

MIN_MINUTES_THRESHOLD: Final[int] = 900

# Attack-normalized (Adj) pitch geometry, centred on (0, 0); goal line at +PITCH_HALF_LENGTH.
# Mirrors enrich_events_pipeline.py so progressive/cross definitions stay identical across the project.
PITCH_HALF_LENGTH: Final[float] = 52.5
PENALTY_AREA_DEPTH: Final[float] = 16.5
PENALTY_AREA_HALF_WIDTH: Final[float] = 20.16
PENALTY_AREA_X_MIN: Final[float] = PITCH_HALF_LENGTH - PENALTY_AREA_DEPTH
WIDE_AREA_MIN_ABS_Y: Final[float] = PENALTY_AREA_HALF_WIDTH
# Minimum distance gained towards the opponent's goal centre, by start/end half.
PROGRESSIVE_MIN_GAIN_OWN_TO_OWN: Final[float] = 30.0
PROGRESSIVE_MIN_GAIN_OWN_TO_OPP: Final[float] = 15.0
PROGRESSIVE_MIN_GAIN_OPP_TO_OPP: Final[float] = 10.0

PRIMARY_HEADER_ACTIONS: Final[list[str]] = ["PASS", "SHOT", "BLOCK", "DRIBBLE", "CLEARANCE"]

VARIANCE_TARGET: Final[float] = 0.85
KMEANS_K_RANGE: Final[range] = range(4, 16)
OPTIMAL_K: Final[int] = 9
RANDOM_STATE: Final[int] = 42
# Defensive volumes are scaled by (50 / opponent possession %) ** exponent; attacking
# volumes by 50 / team possession %. Both re-express the metric at a 50% possession baseline.
PROFILE_DEFENSIVE_PADJ_EXPONENT: Final[float] = 1.0
# K-Means ids are arbitrary: update after each re-run (matches ROLE_CLUSTERS in player_evaluation_pipeline.py).
PROGRESSIVE_FULLBACK_CLUSTER: Final[int] = 3

# Exposed artifacts for VS Code Data Viewer after full pipeline execution.
TRAINING_FEATURE_COLUMNS: list[str] = []
FINAL_TRAINING_DF: pd.DataFrame | None = None
FINAL_CLUSTERED_PLAYERS_DF: pd.DataFrame | None = None
FINAL_CLUSTER_PROFILES_DF: pd.DataFrame | None = None
RAW_PLAYER_FEATURES_DF: pd.DataFrame | None = None

MINUTES_COLUMNS: Final[list[str]] = [
    "seasonId", "seasonName", "matchId", "periodId",
    "playerId", "playerName", "clubName", "gameTimeInSec", "actionType",
    # Needed by build_possession_shares (club attribution and possession-event definition).
    "clubId", "attackingClubId", "WON_GROUND_DUELS", "WON_AERIAL_DUELS",
]

FEATURE_COLUMNS: Final[list[str]] = [
    "seasonId", "playerId", "actionType",
    "SUCCESSFUL_PASSES", "NEUTRAL_PASSES", "UNSUCCESSFUL_PASSES",
    "passDistance",
    "SHOT_AT_GOAL_NUMBER", "shotDistance", "shotAngle",
    "DISTANCE_TO_GOAL_COVERED_DRIBBLE",
    "startAdjCoordinatesX", "startAdjCoordinatesY", "endAdjCoordinatesX", "endAdjCoordinatesY",
    "bodyPart",
    "BALL_WIN_NUMBER", "duelPlayerId",
    "WON_GROUND_DUELS", "LOST_GROUND_DUELS", "WON_AERIAL_DUELS", "LOST_AERIAL_DUELS",
    "startPitchPosition", "startLane",
]


def _validate_data_paths() -> None:
    """Validate the resolved parquet path before attempting pipeline execution."""
    if not EVENTS_PARQUET.exists():
        raise FileNotFoundError(
            "events.parquet was not found. Configure GOALUNIT_DATA_DIR or "
            "GOALUNIT_EVENTS_PARQUET in .env or your shell environment. "
            f"Resolved path: {EVENTS_PARQUET}"
        )

# %%
# --------------------------------------------------------------------------- #
# Data Loading
# --------------------------------------------------------------------------- #

def load_minutes_events(season_ids: list[int]) -> pd.DataFrame:
    """Load the minimal event columns required for playing-time reconstruction."""
    _validate_data_paths()
    events_df = pd.read_parquet(EVENTS_PARQUET, columns=MINUTES_COLUMNS)
    return events_df.loc[events_df["seasonId"].isin(season_ids)].copy()


def load_feature_events(season_ids: list[int]) -> pd.DataFrame:
    """Load the event columns required for playstyle feature engineering."""
    _validate_data_paths()
    events_df = pd.read_parquet(EVENTS_PARQUET, columns=FEATURE_COLUMNS)
    return events_df.loc[events_df["seasonId"].isin(season_ids)].copy()


# %%
# --------------------------------------------------------------------------- #
# Playing-Time Reconstruction
# --------------------------------------------------------------------------- #

def identify_goalkeepers(events_df: pd.DataFrame) -> np.ndarray:
    """Return player IDs who ever performed a goalkeeper-specific ('GK_') action."""
    return events_df.loc[
        events_df["actionType"].str.startswith("GK_", na=False), "playerId"
    ].dropna().unique()


def build_continuous_timeline(events_df: pd.DataFrame) -> pd.DataFrame:
    """Stitch both match halves onto a single continuous in-game clock.

    Wyscout offsets second-half timestamps by 10,000 seconds; this replaces
    that offset with the true elapsed duration of each match's first half.
    """
    h1_end_times = (
        events_df.loc[events_df["periodId"] == 1]
        .groupby("matchId", as_index=False)["gameTimeInSec"]
        .max()
        .rename(columns={"gameTimeInSec": "h1_duration"})
    )
    events_df = events_df.merge(h1_end_times, on="matchId", how="left")

    conditions = [events_df["periodId"] == 1, events_df["periodId"] == 2]
    choices = [
        events_df["gameTimeInSec"],
        (events_df["gameTimeInSec"] - 10_000) + events_df["h1_duration"],
    ]
    events_df["continuous_sec"] = np.select(conditions, choices, default=events_df["gameTimeInSec"])
    return events_df


def compute_player_minutes(
    events_df: pd.DataFrame, gk_ids: np.ndarray, min_minutes: int
) -> pd.DataFrame:
    """Aggregate playing time per outfield player-season.

    Per-match minutes are derived from the span between each player's first
    and last continuous-clock action, summed within each season, and filtered
    to outfield player-seasons exceeding `min_minutes`.
    """
    match_minutes_df = events_df.groupby(["seasonId", "matchId", "playerId"], as_index=False).agg(
        first_action_sec=("continuous_sec", "min"),
        last_action_sec=("continuous_sec", "max"),
    )
    match_minutes_df["match_minutes"] = (
        match_minutes_df["last_action_sec"] - match_minutes_df["first_action_sec"]
    ) / 60

    player_minutes_df = (
        match_minutes_df.groupby(["seasonId", "playerId"], as_index=False)["match_minutes"]
        .sum()
        .rename(columns={"match_minutes": "total_minutes"})
    )

    player_names_df = (
        events_df.loc[:, ["playerId", "playerName"]]
        .dropna(subset=["playerName"])
        .drop_duplicates(subset=["playerId"])
    )
    season_names_df = (
        events_df.loc[:, ["seasonId", "seasonName"]]
        .dropna(subset=["seasonName"])
        .drop_duplicates(subset=["seasonId"])
    )
    # Mid-season transfers yield every club the player featured for within that season.
    player_clubs_df = (
        events_df.loc[:, ["seasonId", "playerId", "clubName"]]
        .dropna(subset=["clubName"])
        .groupby(["seasonId", "playerId"], as_index=False)["clubName"]
        .agg(lambda clubs: ", ".join(sorted(pd.unique(clubs))))
    )

    player_minutes_df = (
        player_minutes_df.merge(player_names_df, on="playerId", how="left")
        .merge(season_names_df, on="seasonId", how="left")
        .merge(player_clubs_df, on=["seasonId", "playerId"], how="left")
    )

    return player_minutes_df.loc[
        (player_minutes_df["total_minutes"] > min_minutes) & (~player_minutes_df["playerId"].isin(gk_ids)),
        ["seasonId", "seasonName", "playerId", "playerName", "clubName", "total_minutes"],
    ]


# %%
# --------------------------------------------------------------------------- #
# Feature Engineering: Family-Level Aggregations
# --------------------------------------------------------------------------- #

def _is_penalty_area(coordinate_x: pd.Series, coordinate_y: pd.Series) -> pd.Series:
    """Attacking penalty area in attack-normalized Adj coordinates."""
    return coordinate_x.ge(PENALTY_AREA_X_MIN) & coordinate_y.abs().le(PENALTY_AREA_HALF_WIDTH)


def flag_progressive_actions(events_df: pd.DataFrame) -> pd.Series:
    """Flag actions that move the ball decisively towards the opponent's goal.

    Tiered distance-gain thresholds on the Euclidean distance to the goal
    centre (+PITCH_HALF_LENGTH, 0): 30m own-half -> own-half, 15m own-half ->
    opposition half, 10m opposition half -> opposition half. Identical to the
    definition in `enrich_events_pipeline.py`; outcome-agnostic by design.
    """
    start_x, start_y = events_df["startAdjCoordinatesX"], events_df["startAdjCoordinatesY"]
    end_x, end_y = events_df["endAdjCoordinatesX"], events_df["endAdjCoordinatesY"]

    dist_start_to_goal = np.sqrt((PITCH_HALF_LENGTH - start_x) ** 2 + start_y**2)
    dist_end_to_goal = np.sqrt((PITCH_HALF_LENGTH - end_x) ** 2 + end_y**2)
    distance_closer_to_goal = dist_start_to_goal - dist_end_to_goal

    start_in_own_half = start_x.lt(0)
    end_in_own_half = end_x.lt(0)
    return (
        (start_in_own_half & end_in_own_half & distance_closer_to_goal.ge(PROGRESSIVE_MIN_GAIN_OWN_TO_OWN))
        | (start_in_own_half & ~end_in_own_half & distance_closer_to_goal.ge(PROGRESSIVE_MIN_GAIN_OWN_TO_OPP))
        | (~start_in_own_half & ~end_in_own_half & distance_closer_to_goal.ge(PROGRESSIVE_MIN_GAIN_OPP_TO_OPP))
    )


def build_passing_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passing volume and style metrics per player.

    The progressive-pass share is a possession-neutral behavioural ratio
    (progressive attempts / all attempts), so players on dominant teams are
    not separated merely by volume of the ball. Both the numerator and the
    denominator count attempted passes regardless of outcome, keeping
    completion quality out of the fingerprint. `crosses_into_box` counts
    attempted passes from wide channels that end inside the penalty area.
    """
    passes_df = events_df.loc[
        events_df["actionType"] == "PASS",
        [
            "seasonId", "playerId", "SUCCESSFUL_PASSES", "passDistance",
            "startAdjCoordinatesX", "startAdjCoordinatesY", "endAdjCoordinatesX", "endAdjCoordinatesY",
            "startLane", "startPitchPosition",
        ],
    ].copy()
    passes_df["progressive_pass"] = flag_progressive_actions(passes_df)
    passes_df["cross_into_box"] = passes_df["startAdjCoordinatesY"].abs().ge(WIDE_AREA_MIN_ABS_Y) & _is_penalty_area(
        passes_df["endAdjCoordinatesX"], passes_df["endAdjCoordinatesY"]
    )
    # Complement of a cross: starts inside the central corridor, ends in the penalty area.
    passes_df["central_pass_into_box"] = passes_df["startAdjCoordinatesY"].abs().lt(
        WIDE_AREA_MIN_ABS_Y
    ) & _is_penalty_area(passes_df["endAdjCoordinatesX"], passes_df["endAdjCoordinatesY"]) & ~_is_penalty_area(
        passes_df["startAdjCoordinatesX"], passes_df["startAdjCoordinatesY"]
    )

    passing_features_df = passes_df.groupby(["seasonId", "playerId"], as_index=False).agg(
        total_passes=("SUCCESSFUL_PASSES", "sum"),
        pass_attempts=("progressive_pass", "size"),
        avg_pass_distance=("passDistance", "mean"),
        progressive_passes=("progressive_pass", "sum"),
        crosses_into_box=("cross_into_box", "sum"),
        central_passes_into_box=("central_pass_into_box", "sum"),
    )
    passing_features_df["progressive_pass_ratio"] = (
        passing_features_df["progressive_passes"] / passing_features_df["pass_attempts"]
    )
    return passing_features_df.drop(columns=["pass_attempts", "progressive_passes"])


def build_shooting_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shot volume, average range, and shot-angle centrality per player."""
    shots_df = events_df.loc[
        events_df["actionType"] == "SHOT",
        ["seasonId", "playerId", "SHOT_AT_GOAL_NUMBER", "shotDistance", "shotAngle"],
    ].copy()
    return shots_df.groupby(["seasonId", "playerId"], as_index=False).agg(
        shot_volume=("SHOT_AT_GOAL_NUMBER", "sum"),
        shot_range=("shotDistance", "mean"),
        # Normalize the 0-360 angle to -180/180 before taking the absolute mean.
        shot_centrality=("shotAngle", lambda values: ((values + 180) % 360 - 180).abs().mean()),
    )


def build_dribbling_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate carry volume/threat metrics and the progressive-carry share per player.

    `dribbleType` is populated only in 2024/25, so every DRIBBLE event is
    treated as a carry and progression is derived from geometry
    (`flag_progressive_actions`), which is consistent across both seasons.
    """
    carries_df = events_df.loc[
        events_df["actionType"] == "DRIBBLE",
        [
            "seasonId", "playerId", "DISTANCE_TO_GOAL_COVERED_DRIBBLE",
            "startAdjCoordinatesX", "startAdjCoordinatesY", "endAdjCoordinatesX", "endAdjCoordinatesY",
        ],
    ].copy()
    carries_df["progressive_carry"] = flag_progressive_actions(carries_df)

    dribbling_features_df = carries_df.groupby(["seasonId", "playerId"], as_index=False).agg(
        total_to_goal_covered_dribble=("DISTANCE_TO_GOAL_COVERED_DRIBBLE", "sum"),
        total_carries=("progressive_carry", "size"),
        progressive_carries=("progressive_carry", "sum"),
    )
    dribbling_features_df["progressive_carry_ratio"] = (
        dribbling_features_df["progressive_carries"] / dribbling_features_df["total_carries"]
    )
    return dribbling_features_df.drop(columns=["total_carries", "progressive_carries"])


def build_header_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Count primary actions executed with the head.

    Restricted to `PRIMARY_HEADER_ACTIONS` to avoid double-counting Wyscout's
    sequential event chains.
    """
    return (
        events_df.loc[
            (events_df["bodyPart"] == "HEAD") & (events_df["actionType"].isin(PRIMARY_HEADER_ACTIONS))
        ]
        .groupby(["seasonId", "playerId"], as_index=False)
        .agg(total_headers=("bodyPart", "count"))
    )


def build_spatial_features(events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot each player's action start zones and lanes into wide frequency counts.
    
    The pitch is folded symmetrically to remove left/right positional bias 
    while preserving the tactical concept of width (Wing vs. Center).
    """
    spatial_df = events_df.loc[:, ["seasonId", "playerId", "startPitchPosition", "startLane"]].copy()

    # Fold the pitch horizontally to merge mirrored tactical roles
    lane_mapping = {
        "LEFT_WING": "WING",
        "RIGHT_WING": "WING",
        "LEFT_HALF_SPACE": "HALF_SPACE",
        "RIGHT_HALF_SPACE": "HALF_SPACE",
        "CENTER": "CENTER"
    }
    spatial_df["startLane"] = spatial_df["startLane"].replace(lane_mapping)

    zone_features_df = (
        spatial_df.pivot_table(index=["seasonId", "playerId"], columns="startPitchPosition", aggfunc="size", fill_value=0)
        .add_prefix("zone_")
        .reset_index()
    )
    lane_features_df = (
        spatial_df.pivot_table(index=["seasonId", "playerId"], columns="startLane", aggfunc="size", fill_value=0)
        .add_prefix("lane_")
        .reset_index()
    )
    return zone_features_df, lane_features_df


def build_defensive_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ball-winning, duel, and ball-win-share metrics from both duel perspectives.

    Duels are recorded from the primary actor's perspective only, so the
    opponent's losses/wins are reconstructed via `duelPlayerId` to derive
    symmetric win rates.
    """
    defensive_df = events_df.loc[
        :,
        [
            "seasonId", "playerId", "duelPlayerId", "actionType", "BALL_WIN_NUMBER",
            "WON_GROUND_DUELS", "LOST_GROUND_DUELS", "WON_AERIAL_DUELS", "LOST_AERIAL_DUELS",
            "startAdjCoordinatesX",
        ],
    ].copy()

    # Primary perspective: the player is the main actor, WON_ columns are duels this player won.
    # Attack-normalized X so ball-win height is comparable across halves (raw X flips at half time).
    primary_df = defensive_df.groupby(["seasonId", "playerId"], as_index=False).agg(
        won_aerial_duels=("WON_AERIAL_DUELS", "sum"),
        won_ground_duels=("WON_GROUND_DUELS", "sum"),
        total_ball_wins=("BALL_WIN_NUMBER", "sum"),
        total_events=("actionType", "size"),
        average_defensive_height=(
            "startAdjCoordinatesX",
            lambda values: values[defensive_df.loc[values.index, "BALL_WIN_NUMBER"].gt(0)].mean(),
        ),
    )

    # Secondary perspective: the player is the opponent (duelPlayerId); a primary win is a secondary loss.
    secondary_df = defensive_df.dropna(subset=["duelPlayerId"]).groupby(
        ["seasonId", "duelPlayerId"], as_index=False
    ).agg(
        lost_aerial_duels=("WON_AERIAL_DUELS", "sum"),
        lost_ground_duels=("WON_GROUND_DUELS", "sum"),
    )

    defensive_features_df = primary_df.merge(
        secondary_df,
        left_on=["seasonId", "playerId"],
        right_on=["seasonId", "duelPlayerId"],
        how="left",
    ).fillna(0)

    defensive_features_df["total_aerial_duels"] = (
        defensive_features_df["won_aerial_duels"] + defensive_features_df["lost_aerial_duels"]
    )
    defensive_features_df["total_ground_duels"] = (
        defensive_features_df["won_ground_duels"] + defensive_features_df["lost_ground_duels"]
    )
    defensive_features_df["total_duels"] = (
        defensive_features_df["total_aerial_duels"] + defensive_features_df["total_ground_duels"]
    )
    defensive_features_df["aerial_win_rate"] = (
        defensive_features_df["won_aerial_duels"] / defensive_features_df["total_aerial_duels"]
    ).fillna(0)
    defensive_features_df["ground_duel_win_rate"] = (
        defensive_features_df["won_ground_duels"] / defensive_features_df["total_ground_duels"]
    ).fillna(0)
    # Behavioural (outcome-agnostic) split of duel involvement: ground vs. aerial.
    defensive_features_df["ground_duel_share"] = (
        defensive_features_df["total_ground_duels"] / defensive_features_df["total_duels"]
    ).fillna(0)
    # Ball-winning as a share of all the player's events: separates ball-winning midfielders
    # from deep-lying playmakers of similar volume, independent of team possession.
    defensive_features_df["ball_win_share"] = (
        defensive_features_df["total_ball_wins"] / defensive_features_df["total_events"]
    )

    return defensive_features_df.drop(
        columns=[
            "duelPlayerId", "won_aerial_duels", "lost_aerial_duels", "won_ground_duels", "lost_ground_duels",
            "total_events",
        ],
        errors="ignore",
    )


def build_aerial_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Share of a player's actions that happen in the air rather than on the ground.

    Aerial actions are headed primary actions (`PRIMARY_HEADER_ACTIONS`) and
    won aerial duels on the player's own rows, plus aerial duels the player
    lost (reconstructed from opponents' won-duel rows via `duelPlayerId`).
    The denominator is every action involving the player, so the ground share
    is the complement (1 - aerial_action_share).
    """
    won_aerial = events_df["WON_AERIAL_DUELS"].fillna(0).gt(0)
    is_aerial = (
        events_df["bodyPart"].eq("HEAD") & events_df["actionType"].isin(PRIMARY_HEADER_ACTIONS)
    ) | won_aerial
    own_df = (
        events_df.assign(is_aerial=is_aerial)
        .groupby(["seasonId", "playerId"], as_index=False)
        .agg(aerial_actions=("is_aerial", "sum"), total_actions=("is_aerial", "size"))
    )
    lost_df = (
        events_df.loc[events_df["duelPlayerId"].notna() & won_aerial, ["seasonId", "duelPlayerId"]]
        .groupby(["seasonId", "duelPlayerId"], as_index=False)
        .size()
        .rename(columns={"duelPlayerId": "playerId", "size": "lost_aerial_actions"})
    )
    aerial_df = own_df.merge(lost_df, on=["seasonId", "playerId"], how="left")
    lost_actions = aerial_df["lost_aerial_actions"].fillna(0)
    aerial_df["aerial_action_share"] = (aerial_df["aerial_actions"] + lost_actions) / (
        aerial_df["total_actions"] + lost_actions
    )
    return aerial_df[["seasonId", "playerId", "aerial_action_share"]]


def build_discipline_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fouls-committed counts per player.

    Yellow cards are excluded: they are only recorded in one of the two seasons,
    so they would act as a season flag rather than a behavioural signal.
    """
    return (
        events_df.loc[events_df["actionType"] == "FOUL"]
        .groupby(["seasonId", "playerId"], as_index=False)
        .agg(total_fouls=("actionType", "count"))
    )



# %%
# --------------------------------------------------------------------------- #
# Feature Engineering: Assembly & Per-90 Normalization
# --------------------------------------------------------------------------- #

def engineer_player_features(
    events_df: pd.DataFrame, player_minutes_df: pd.DataFrame, possession_shares_df: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the per-player-season playstyle fingerprint, normalized to per-90 values.

    Per-90 volumes are possession-adjusted to a 50% baseline: attacking volumes
    are multiplied by 50 / team possession %, defensive volumes by
    (50 / opponent possession %) ** `PROFILE_DEFENSIVE_PADJ_EXPONENT`.

    Each (seasonId, playerId) pair is an independent entity. Feature families
    are merged with outer joins so player-seasons missing an
    entire action type (e.g. zero shots) are not dropped. The subsequent
    inner join against `player_minutes_df` then acts as the strict
    minutes-eligibility filter for the final population.
    """
    passing_features_df = build_passing_features(events_df)
    shooting_features_df = build_shooting_features(events_df)
    dribble_features_df = build_dribbling_features(events_df)
    header_features_df = build_header_features(events_df)
    zone_features_df, lane_features_df = build_spatial_features(events_df)
    defensive_features_df = build_defensive_features(events_df)
    fouls_df = build_discipline_features(events_df)
    aerial_features_df = build_aerial_features(events_df)

    player_keys = ["seasonId", "playerId"]
    player_features_df = passing_features_df.merge(shooting_features_df, on=player_keys, how="outer")
    player_features_df = player_features_df.merge(dribble_features_df, on=player_keys, how="outer")
    player_features_df = player_features_df.merge(header_features_df, on=player_keys, how="outer")
    player_features_df = player_features_df.merge(
        player_minutes_df.loc[:, ["seasonId", "seasonName", "playerId", "playerName", "clubName", "total_minutes"]],
        on=player_keys,
        how="inner",
    )

    # No attempts of an action type means a zero progressive share, not a missing value.
    ratio_cols = ["progressive_pass_ratio", "progressive_carry_ratio"]
    player_features_df[ratio_cols] = player_features_df[ratio_cols].fillna(0)

    volume_cols = [
        "total_passes", "crosses_into_box", "central_passes_into_box", "shot_volume",
        "total_to_goal_covered_dribble", "total_headers",
    ]

    # A missing possession share falls back to the 50% baseline (multiplier 1).
    player_features_df = player_features_df.merge(possession_shares_df, on=player_keys, how="left")
    team_possession_pct = (player_features_df["possession_share"] * 100).fillna(50.0)
    attacking_multiplier = possession_multiplier(team_possession_pct)
    defensive_multiplier = possession_multiplier(100.0 - team_possession_pct) ** PROFILE_DEFENSIVE_PADJ_EXPONENT

    per90_multiplier = 90 / player_features_df["total_minutes"]
    for column in volume_cols:
        player_features_df[f"{column}_p90"] = (
            player_features_df[column].fillna(0) * per90_multiplier * attacking_multiplier
        )

    player_features_df = player_features_df[
        [
            "seasonId", "seasonName", "playerId", "playerName", "clubName", "total_minutes",
            "avg_pass_distance", *ratio_cols, "shot_range", "shot_centrality",
            *[f"{column}_p90" for column in volume_cols],
        ]
    ]

    player_features_df = player_features_df.merge(zone_features_df, on=player_keys, how="left")
    player_features_df = player_features_df.merge(lane_features_df, on=player_keys, how="left")
    spatial_volume_cols = [
        column for column in player_features_df.columns if column.startswith("zone_") or column.startswith("lane_")
    ]
    player_features_df[spatial_volume_cols] = player_features_df[spatial_volume_cols].fillna(0)
    # Raw-count ratio (scale-free, so computed before per-90 conversion); +1 guards zero box touches.
    player_features_df["drop_deep_ratio"] = np.log1p(player_features_df["zone_MIDDLE_THIRD"] / (
        player_features_df["zone_OPPONENT_BOX"] + 1
    ))
    # Zones and lanes each partition the same event total, so per-90 counts duplicate overall activity.
    # Express them as shares of the player's events and drop one reference category per family
    # (the shares sum to 1, so it is fully determined by the others).
    for prefix, reference in (("zone_", "zone_FIRST_THIRD"), ("lane_", "lane_CENTER")):
        family_cols = [column for column in spatial_volume_cols if column.startswith(prefix)]
        family_totals = player_features_df[family_cols].sum(axis=1)
        for column in family_cols:
            if column != reference:
                player_features_df[f"{column}_event_share"] = player_features_df[column] / family_totals.where(
                    family_totals.gt(0)
                )
    share_cols = [column for column in player_features_df.columns if column.endswith("_event_share")]
    player_features_df[share_cols] = player_features_df[share_cols].fillna(0)

    player_features_df = player_features_df.merge(defensive_features_df, on=player_keys, how="left")
    player_features_df = player_features_df.merge(fouls_df, on=player_keys, how="left")
    player_features_df = player_features_df.merge(aerial_features_df, on=player_keys, how="left")
    # No recorded actions at all is treated as a purely ground profile.
    player_features_df["aerial_action_share"] = player_features_df["aerial_action_share"].fillna(0)

    # `total_defensive_touches` is dropped: it is ~perfectly collinear with ball wins (r > 0.999).
    defensive_volume_cols = [
        "total_ball_wins", "total_aerial_duels", "total_ground_duels", "total_fouls",
    ]
    for column in defensive_volume_cols:
        player_features_df[f"{column}_p90"] = (
            player_features_df[column].fillna(0) * per90_multiplier * defensive_multiplier
        )

    # Fill NaNs created by taking the mean of 0 actions
    mean_cols = ["avg_pass_distance", "shot_range", "shot_centrality", "average_defensive_height"]
    player_features_df[mean_cols] = player_features_df[mean_cols].fillna(0)

    return player_features_df[
        [
            "seasonId", "seasonName", "playerId", "playerName", "clubName", "total_minutes",
            "avg_pass_distance", *ratio_cols, "shot_range", "shot_centrality",
            "aerial_win_rate", "ground_duel_win_rate", "ground_duel_share", "ball_win_share",
            "aerial_action_share", "average_defensive_height",
            "drop_deep_ratio",
            *[column for column in player_features_df.columns if column.endswith(("_p90", "_event_share"))],
        ]
    ]


# %%
# --------------------------------------------------------------------------- #
# Dimensionality Reduction (PCA)
# --------------------------------------------------------------------------- #

def scale_features(player_features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Log-transform per-90 volumes, then z-score numeric playstyle metrics.

    Per-90 counts are right-skewed (e.g. aerial duels, crosses, shots), so a few
    outliers would dominate Euclidean distances and PCA variance. `log1p` is
    applied only to `_p90` volume columns, inside the clustering input; the
    exported feature frame keeps interpretable raw per-90 units. Ratios, shares
    and averages are already bounded/scale-free and are left untouched.
    Identifiers, minutes and qualitative rates are excluded.
    """
    # Explicitly define columns to exclude from the clustering phase
    # For example we exclude duels win rate since they represent qualitative assessments rather than raw counts
    exclude_cols = ["seasonId", "playerId", "total_minutes", "aerial_win_rate", "ground_duel_win_rate"]

    feature_columns = player_features_df.select_dtypes(include=["number"]).columns.drop(exclude_cols)
    model_input_df = player_features_df[feature_columns].copy()
    volume_columns = [column for column in feature_columns if column.endswith("_p90")]
    model_input_df[volume_columns] = np.log1p(model_input_df[volume_columns])

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(model_input_df)
    scaled_features_df = pd.DataFrame(scaled_array, columns=feature_columns, index=player_features_df.index)
    return scaled_features_df, feature_columns


def explore_pca_variance(scaled_features_df: pd.DataFrame, random_state: int = RANDOM_STATE) -> PCA:
    """Fit an unrestricted PCA to inspect the full explained-variance spectrum."""
    pca = PCA(random_state=random_state)
    pca.fit(scaled_features_df)
    return pca


def plot_explained_variance(pca: PCA, variance_target: float = VARIANCE_TARGET) -> None:
    """Plot cumulative explained variance against the number of retained components."""
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="--")
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.axhline(y=variance_target, color="r", linestyle="-")
    plt.grid(True)
    plt.show()


def summarize_explained_variance(pca: PCA) -> pd.DataFrame:
    """Tabulate per-component and cumulative explained variance as formatted percentages."""
    individual_variance = pca.explained_variance_ratio_
    cumulative_variance = individual_variance.cumsum()
    summary_df = pd.DataFrame(
        {
            "Number_of_Components": range(1, len(individual_variance) + 1),
            "Variance_Added": individual_variance,
            "Total_Cumulative_Variance": cumulative_variance,
        }
    )
    summary_df["Variance_Added"] = (summary_df["Variance_Added"] * 100).round(2).astype(str) + "%"
    summary_df["Total_Cumulative_Variance"] = (
        (summary_df["Total_Cumulative_Variance"] * 100).round(2).astype(str) + "%"
    )
    return summary_df


def fit_final_pca(
    scaled_features_df: pd.DataFrame, variance_target: float = VARIANCE_TARGET, random_state: int = RANDOM_STATE
) -> tuple[PCA, np.ndarray, int]:
    """Fit PCA with the selected component count and project the scaled features."""
    # Passing a float (0.0 to 1.0) tells sklearn to select the number of components automatically
    pca = PCA(n_components=variance_target, random_state=random_state)
    pca_features_array = pca.fit_transform(scaled_features_df)

    # Extract the dynamically chosen number of components
    n_components_chosen = pca.n_components_
    
    return pca, pca_features_array, n_components_chosen


def build_pca_dataframe(
    pca_features_array: np.ndarray, player_features_df: pd.DataFrame, n_components_chosen: int
) -> tuple[pd.DataFrame, list[str]]:
    """Attach player identity columns to the dynamically projected principal component scores."""
    pca_cols = [f"PC{i + 1}" for i in range(n_components_chosen)]
    pca_df = pd.DataFrame(pca_features_array, columns=pca_cols, index=player_features_df.index)
    final_pca_df = pd.concat(
        [player_features_df[["seasonId", "seasonName", "playerId", "playerName", "clubName", "total_minutes"]], pca_df],
        axis=1,
    )
    return final_pca_df, pca_cols


def compute_component_loadings(pca: PCA, feature_columns: pd.Index, pca_cols: list[str]) -> pd.DataFrame:
    """Return the PCA loading matrix (features x components)."""
    return pd.DataFrame(pca.components_.T, columns=pca_cols, index=feature_columns)


def print_component_drivers(loadings_df: pd.DataFrame, pca_cols: list[str], top_n: int = 5) -> None:
    """Print the top positive and negative feature loadings for each principal component."""
    for pc in pca_cols:
        print(f"========== {pc} ==========")
        sorted_loadings = loadings_df[pc].sort_values(ascending=False)
        top_positive = sorted_loadings.head(top_n)
        top_negative = sorted_loadings.tail(top_n)

        print("Top Positive Drivers (+):")
        for feature, weight in top_positive.items():
            print(f"  {feature}: {weight:.3f}")

        print("\nTop Negative Drivers (-):")
        for feature, weight in top_negative.sort_values(ascending=True).items():
            print(f"  {feature}: {weight:.3f}")
        print("\n")


# %%
# --------------------------------------------------------------------------- #
# Clustering (K-Means)
# --------------------------------------------------------------------------- #

def evaluate_kmeans_k(
    clustering_data: pd.DataFrame, k_range: range, random_state: int = RANDOM_STATE
) -> tuple[list[float], list[float]]:
    """Fit K-Means across a range of k and collect inertia and silhouette diagnostics."""
    inertia_values: list[float] = []
    silhouette_scores: list[float] = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(clustering_data)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(clustering_data, cluster_labels))
    return inertia_values, silhouette_scores


def plot_kmeans_diagnostics(k_range: range, inertia_values: list[float], silhouette_scores: list[float]) -> None:
    """Plot the elbow (inertia) and silhouette score curves side by side."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(k_range, inertia_values, marker="o", linestyle="--")
    ax1.set_title("Elbow Method (Inertia)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Sum of Squared Distances)")
    ax1.set_xticks(k_range)
    ax1.grid(True)

    ax2.plot(k_range, silhouette_scores, marker="o", linestyle="--", color="green")
    ax2.set_title("Silhouette Score Validation")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Coefficient")
    ax2.set_xticks(k_range)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def fit_final_kmeans(
    final_pca_df: pd.DataFrame, pca_cols: list[str], k: int, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, KMeans]:
    """Assign tactical cluster labels and each player's distance to its assigned centroid."""
    clustering_data = final_pca_df[pca_cols]
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    final_pca_df = final_pca_df.copy()
    final_pca_df["tactical_cluster"] = kmeans.fit_predict(clustering_data)

    distances = pairwise_distances(clustering_data, kmeans.cluster_centers_)
    final_pca_df["distance_to_center"] = [
        distances[i, label] for i, label in enumerate(final_pca_df["tactical_cluster"])
    ]
    return final_pca_df, kmeans


def summarize_cluster_profiles(
    final_pca_df: pd.DataFrame, pca_cols: list[str], n_examples: int = 3
) -> pd.DataFrame:
    """Compute mean PC scores per cluster and attach the closest-to-centroid example players.

    These "purest" examples represent the clearest execution of each
    discovered tactical archetype.
    """
    cluster_profiles = final_pca_df.groupby("tactical_cluster")[pca_cols].mean()

    top_examples = (
        final_pca_df.assign(example_label=final_pca_df["playerName"] + " (" + final_pca_df["seasonName"] + ")")
        .sort_values(["tactical_cluster", "distance_to_center"])
        .groupby("tactical_cluster")
        .head(n_examples)
        .groupby("tactical_cluster")["example_label"]
        .apply(lambda names: ", ".join(names))
    )
    cluster_profiles["purest_player_examples"] = top_examples
    return cluster_profiles


def print_cluster_feature_drivers(
    scaled_features_df: pd.DataFrame, cluster_labels: pd.Series, cluster_id: int, top_n: int = 5
) -> None:
    """Print the original features that most strongly define one cluster.

    Each feature is z-scored over the whole population (mean 0, SD 1), so a
    cluster's mean scaled value reads as "standard deviations above/below the
    average player". The largest positive and negative means are the cluster's
    dominant over- and under-indexed features.
    """
    cluster_means = scaled_features_df.loc[cluster_labels.eq(cluster_id)].mean().sort_values(ascending=False)
    print(f"========== Cluster {cluster_id} feature drivers (mean z-score, n={int(cluster_labels.eq(cluster_id).sum())}) ==========")
    print(f"Top {top_n} Positive Drivers (+):")
    for feature, value in cluster_means.head(top_n).items():
        print(f"  {feature}: {value:+.2f}")
    print(f"\nTop {top_n} Negative Drivers (-):")
    for feature, value in cluster_means.tail(top_n).sort_values().items():
        print(f"  {feature}: {value:+.2f}")
    print("\n")


# %%
# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the full playstyle-fingerprint and tactical-classification pipeline."""
    global TRAINING_FEATURE_COLUMNS, FINAL_TRAINING_DF, FINAL_CLUSTERED_PLAYERS_DF, FINAL_CLUSTER_PROFILES_DF, RAW_PLAYER_FEATURES_DF

    # Phase 1: Playstyle Fingerprint
    minutes_events_df = load_minutes_events(TARGET_SEASON_IDS)
    gk_ids = identify_goalkeepers(minutes_events_df)
    minutes_events_df = build_continuous_timeline(minutes_events_df)
    player_minutes_df = compute_player_minutes(minutes_events_df, gk_ids, MIN_MINUTES_THRESHOLD)

    possession_shares_df = build_possession_shares(minutes_events_df)

    feature_events_df = load_feature_events(TARGET_SEASON_IDS)
    player_features_df = engineer_player_features(feature_events_df, player_minutes_df, possession_shares_df)

    RAW_PLAYER_FEATURES_DF = player_features_df.copy()

    # Phase 2: Tactical Classification
    scaled_features_df, feature_columns = scale_features(player_features_df)
    TRAINING_FEATURE_COLUMNS = feature_columns.tolist()
    FINAL_TRAINING_DF = pd.concat(
        [
            player_features_df.loc[
                :, ["seasonId", "seasonName", "playerId", "playerName", "clubName", "total_minutes"]
            ],
            scaled_features_df,
        ],
        axis=1,
    )

    exploratory_pca = explore_pca_variance(scaled_features_df)
    plot_explained_variance(exploratory_pca)
    print(summarize_explained_variance(exploratory_pca).head(10))

    # Fit PCA dynamically
    final_pca, pca_features_array, n_components_chosen = fit_final_pca(scaled_features_df, VARIANCE_TARGET)
    
    # Build dataframe using the chosen components
    final_pca_df, pca_cols = build_pca_dataframe(pca_features_array, player_features_df, n_components_chosen)

    loadings_df = compute_component_loadings(final_pca, feature_columns, pca_cols)
    print_component_drivers(loadings_df, pca_cols)

    inertia_values, silhouette_scores = evaluate_kmeans_k(final_pca_df[pca_cols], KMEANS_K_RANGE)
    plot_kmeans_diagnostics(KMEANS_K_RANGE, inertia_values, silhouette_scores)

    final_pca_df, final_kmeans = fit_final_kmeans(final_pca_df, pca_cols, OPTIMAL_K)
    cluster_profiles = summarize_cluster_profiles(final_pca_df, pca_cols)

    FINAL_CLUSTERED_PLAYERS_DF = final_pca_df.copy()
    FINAL_CLUSTER_PROFILES_DF = cluster_profiles.copy()

    print(cluster_profiles)
    print_cluster_feature_drivers(
        scaled_features_df, final_pca_df["tactical_cluster"], PROGRESSIVE_FULLBACK_CLUSTER
    )

    # CSV has no sheet support, so loadings and per-cluster PC profiles go into one workbook
    with pd.ExcelWriter(PROJECT_ROOT / "pca_loadings.xlsx") as writer:
        loadings_df.rename_axis("feature").to_excel(writer, sheet_name="PCA_Loadings")
        cluster_profiles.to_excel(writer, sheet_name="Cluster_PCA_Profiles")

    # Save the final dataframe to a CSV for external visualization scripts
    FINAL_CLUSTERED_PLAYERS_DF.to_csv(PROJECT_ROOT / "final_tactical_clusters.csv", index=False)

    # Merge the original per-90 features with the new cluster labels
    radar_export_df = player_features_df.copy()
    radar_export_df["tactical_cluster"] = final_pca_df["tactical_cluster"]
    radar_export_df.to_csv(PROJECT_ROOT / "final_tactical_clusters_radar.csv", index=False)

if __name__ == "__main__":
    main()
