"""Value-for-money pipeline for player evaluation.

This standalone script loads the evaluation metrics export and the players
market-data export, joins them on (seasonId, playerId), computes a VFM ratio
against the Goalunit Fair Price (GFP), standardizes the ratio within tactical
clusters, exports a clean ranking table, and writes an interactive Plotly
scatter (performance index vs. inverted fair price on a log axis, coloured by
age) with a tactical-cluster dropdown.

VFM ranking: fair prices span ~2 orders of magnitude while the performance index
spans roughly +/-1 sd, so a raw `index / price` ratio is almost a pure inverse
price ranking (Spearman ~ -0.99 with price). The ranking metric is therefore the
scale-free `vfm_score = z(performance index) - PRICE_WEIGHT * z(ln GFP)`, each
standardised within the tactical cluster over priced players. `vfm_ratio` is kept
as a legacy column for comparison only.

Cluster/role labels come from the evaluation export (`evaluation_role`), which
`player_evaluation_pipeline.py` assigns from its hardcoded `ROLE_CLUSTERS`
mapping. Before ranking, the script warns if the evaluation export is older than
the latest clustering, so an export left over from an earlier profiling run is
not used silently.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go


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

EVALUATION_CSV: Final[Path] = Path(
    os.getenv(
        "GOALUNIT_EVALUATION_CSV",
        str(PROJECT_ROOT / "player_evaluation_metrics.csv"),
    )
)
PLAYERS_CSV: Final[Path] = Path(
    os.getenv("GOALUNIT_PLAYERS_CSV", str(DATA_DIR / "players.csv"))
)
OUTPUT_CSV: Final[Path] = Path(
    os.getenv("GOALUNIT_VFM_OUTPUT_CSV", str(PROJECT_ROOT / "vfm_results.csv"))
)
CLUSTERS_CSV: Final[Path] = Path(
    os.getenv("GOALUNIT_CLUSTERS_CSV", str(PROJECT_ROOT / "final_tactical_clusters.csv"))
)
SCATTER_HTML: Final[Path] = Path(
    os.getenv("GOALUNIT_VFM_SCATTER_HTML", str(PROJECT_ROOT / "vfm_scatter.html"))
)

PLAYER_SEASON_KEYS: Final[list[str]] = ["seasonId", "playerId"]
GFP_COLUMN: Final[str] = "Goalunit Fair Price (GFP)"
DATE_FORMAT: Final[str] = "%d/%m/%Y"
DAYS_PER_YEAR: Final[float] = 365.25
# Weight of the standardised log price in the VFM score (1.0 = price and performance count equally).
PRICE_WEIGHT: Final[float] = 1.0
# Price-axis presentation (fair price in millions, logarithmic axis).
LOG_PRICE_AXIS_PAD: Final[float] = 0.06  # padding in log10 units
LOG_PRICE_TICKS_M: Final[tuple[float, ...]] = (1, 2, 5, 10, 20, 50, 100, 200)


def _validate_input_paths() -> None:
    """Validate that the required input files exist."""
    missing_paths = [path for path in (EVALUATION_CSV, PLAYERS_CSV) if not path.exists()]
    if missing_paths:
        formatted_paths = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required input files were not found: {formatted_paths}")


def validate_cluster_consistency(evaluation_df: pd.DataFrame) -> None:
    """Warn if the evaluation export predates the latest clustering.

    K-Means indices are re-assigned on every profiling run and the role-to-cluster
    mapping is hardcoded, so an evaluation export older than the clusters file
    may carry roles attached to the wrong clusters and must be re-run.
    """
    if CLUSTERS_CSV.exists() and EVALUATION_CSV.stat().st_mtime < CLUSTERS_CSV.stat().st_mtime:
        warnings.warn(
            f"{EVALUATION_CSV.name} is older than {CLUSTERS_CSV.name}; the evaluation metrics may predate "
            "the latest profiling run.",
            stacklevel=2,
        )


def _zscore(values: pd.Series) -> pd.Series:
    """Return population Z-scores, using zero when a cluster has no variance."""
    std = values.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def build_price_table(players_df: pd.DataFrame) -> pd.DataFrame:
    """Return one GFP and age-at-valuation row per player-season.

    The latest valuation snapshot within each season is kept, and age is
    measured at that snapshot date so price and age refer to the same moment.
    """
    price_df = players_df[
        ["seasonId", "playerId", "dateOfBirth", "snapshot_date", "fairPrice"]
    ].copy()
    snapshot_date = pd.to_datetime(price_df["snapshot_date"], format=DATE_FORMAT, errors="coerce")
    birth_date = pd.to_datetime(price_df["dateOfBirth"], format=DATE_FORMAT, errors="coerce")
    price_df["snapshot_date"] = snapshot_date
    price_df["age"] = (snapshot_date - birth_date).dt.days / DAYS_PER_YEAR
    price_df = price_df.sort_values([*PLAYER_SEASON_KEYS, "snapshot_date"], na_position="first")
    price_df = price_df.drop_duplicates(subset=PLAYER_SEASON_KEYS, keep="last")
    return price_df.rename(columns={"fairPrice": GFP_COLUMN}).drop(
        columns=["dateOfBirth", "snapshot_date"]
    )


def add_vfm_score(merged_df: pd.DataFrame, price_weight: float = PRICE_WEIGHT) -> pd.DataFrame:
    """Add the cluster-standardised VFM score ``z(performance) - w * z(ln GFP)``.

    Both terms are z-scored within each tactical cluster over the priced players
    only (GFP > 0), so the score is unit-free and the two components are on the
    same scale. Unpriced (free-transfer) players receive NaN.
    """
    scored_df = merged_df.copy()
    gfp = pd.to_numeric(scored_df[GFP_COLUMN], errors="coerce")
    scored_df["log_gfp"] = np.log(gfp.where(gfp.gt(0)))
    priced = scored_df["log_gfp"].notna()

    for source, target in (("performance_index", "performance_z"), ("log_gfp", "log_price_z")):
        scored_df[target] = np.nan
        scored_df.loc[priced, target] = (
            scored_df.loc[priced].groupby("tactical_cluster")[source].transform(_zscore)
        )
    scored_df["vfm_score"] = scored_df["performance_z"] - price_weight * scored_df["log_price_z"]
    return scored_df


def prepare_plot_data(vfm_df: pd.DataFrame) -> pd.DataFrame:
    """Return the evaluated player-seasons that can be plotted, with price in millions."""
    plot_df = vfm_df.loc[
        vfm_df["evaluation_role"].notna()
        & vfm_df[GFP_COLUMN].gt(0)  # <-- Changed to .ge(0) to include 0-value players
        & vfm_df["age"].notna()
    ].copy()
    plot_df["price_m"] = plot_df[GFP_COLUMN] / 1_000_000.0
    return plot_df


def build_vfm_figure(vfm_df: pd.DataFrame, default_cluster: int | None = None) -> go.Figure:
    """Scatter of performance index (x) vs. inverted fair price (y), coloured by age.

    The y-axis is reversed so the top-right quadrant holds high-performing,
    low-priced players. A dropdown switches between tactical clusters; dashed
    lines mark each cluster's median index and median price to frame the quadrants.
    """
    plot_df = prepare_plot_data(vfm_df)
    cluster_roles = (
        plot_df.drop_duplicates("tactical_cluster")
        .set_index("tactical_cluster")["evaluation_role"]
        .sort_index()
    )
    clusters = list(cluster_roles.index)
    default_cluster = clusters[0] if default_cluster is None else default_cluster
    age_min, age_max = float(plot_df["age"].min()), float(plot_df["age"].max())

    figure = go.Figure()
    updates: list[dict] = []
    for position, cluster in enumerate(clusters):
        group = plot_df.loc[plot_df["tactical_cluster"].eq(cluster)]
        
        # <-- Added logic to gracefully handle the NaN VFM ratios of free agents
        hover = []
        for row in group.itertuples():
            vfm_str = f"{row.vfm_score:+.2f} sd (rank {int(row.vfm_rank)})" if pd.notna(row.vfm_score) else "N/A (Free Transfer)"
            hover.append(
                f"<b>{row.playerName}</b> ({row.seasonName})<br>{row.clubName}<br>"
                f"Age: {row.age:.1f}<br>Performance index: {row.performance_index:.1f}<br>"
                f"Fair price: {row.price_m:.1f}M<br>VFM score: {vfm_str}"
            )
            
        figure.add_trace(
            go.Scatter(
                x=group["performance_index"], y=group["price_m"], mode="markers",
                name=cluster_roles[cluster], text=hover, hoverinfo="text",
                visible=cluster == default_cluster, showlegend=False,
                marker=dict(
                    size=11, opacity=0.9, color=group["age"],
                    colorscale="Viridis", reversescale=True, cmin=age_min, cmax=age_max,
                    colorbar=dict(title="Age", thickness=14),
                    line=dict(width=0.8, color="rgba(255,255,255,0.7)"),
                ),
            )
        )

        median_index = float(group["performance_index"].median())
        median_price = float(group["price_m"].median())
        x_pad = 0.05 * float(group["performance_index"].max() - group["performance_index"].min())
        visibility = [index == position for index in range(len(clusters))]
        line_style = dict(color="rgba(0,0,0,0.4)", width=1.2, dash="dash")
        updates.append(
            {
                "visible": visibility,
                "xaxis.range": [
                    float(group["performance_index"].min() - x_pad),
                    float(group["performance_index"].max() + x_pad),
                ],
                # Log axis (range in log10 units), reversed: the larger bound comes first so cheap players sit on top.
                "yaxis.range": [
                    float(np.log10(group["price_m"].max()) + LOG_PRICE_AXIS_PAD),
                    float(np.log10(group["price_m"].min()) - LOG_PRICE_AXIS_PAD),
                ],
                "shapes": [
                    dict(type="line", xref="x", yref="paper", x0=median_index, x1=median_index,
                         y0=0, y1=1, line=line_style),
                    dict(type="line", xref="paper", yref="y", x0=0, x1=1,
                         y0=median_price, y1=median_price, line=line_style),
                ],
            }
        )

    buttons = [
        dict(
            label=f"{cluster_roles[cluster]} (cluster {cluster})", method="update",
            args=[
                {"visible": update["visible"]},
                {key: value for key, value in update.items() if key != "visible"},
            ],
        )
        for cluster, update in zip(clusters, updates)
    ]

    initial = updates[clusters.index(default_cluster)]
    axis_style = dict(gridcolor="rgba(0,0,0,0.1)", color="#000000", tickfont=dict(size=12))
    figure.update_layout(
        title=dict(
            text=(
                "<b>Performance index vs. Goalunit Fair Price</b><br>"
                "<sup>Top-right = high performance at a low price · colour = age at valuation · "
                "dashed lines = cluster medians</sup>"
            ),
            x=0.02,
        ),
        xaxis=dict(title="Performance index", range=initial["xaxis.range"], **axis_style),
        yaxis=dict(title="Goalunit Fair Price (M, log scale) - inverted", type="log",
                   range=initial["yaxis.range"], tickvals=list(LOG_PRICE_TICKS_M),
                   ticktext=[f"{tick:g}M" for tick in LOG_PRICE_TICKS_M], **axis_style),
        shapes=initial["shapes"],
        annotations=[
            dict(text="<b>Underpriced Zone</b><br>high performance · low price", xref="paper",
                 yref="paper", x=0.99, y=0.99, xanchor="right", yanchor="top", showarrow=False,
                 font=dict(color="#2FCA21", size=16), align="right"),
            dict(text="<b>Overpriced Zone</b><br>low performance · high price", xref="paper",
                 yref="paper", x=0.02, y=0.12, xanchor="left", yanchor="bottom", showarrow=False,
                 font=dict(color="#d62728", size=16), align="left"),
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#000000", family="Arial"),
        updatemenus=[
            dict(
                buttons=buttons, direction="down", x=1.0, xanchor="right", y=1.12,
                yanchor="top", active=clusters.index(default_cluster),
                bgcolor="#FFFFFF", bordercolor="#CCCCCC", font=dict(color="#000000"),
            )
        ],
        hoverlabel=dict(bgcolor="#1C2230", font_color="#FFFFFF"),
        height=760, margin=dict(l=70, r=40, t=100, b=70),
    )
    return figure


def main() -> None:
    """Load, merge, score, rank, and export the VFM results."""
    _validate_input_paths()

    evaluation_df = pd.read_csv(EVALUATION_CSV)
    players_df = pd.read_csv(PLAYERS_CSV)

    for key in PLAYER_SEASON_KEYS:
        if key not in evaluation_df.columns or key not in players_df.columns:
            raise KeyError(f"Both input files must contain a {key} column.")

    validate_cluster_consistency(evaluation_df)
    evaluation_df = evaluation_df.copy()
    price_df = build_price_table(players_df)

    merged_df = evaluation_df.merge(
        price_df,
        on=PLAYER_SEASON_KEYS,
        how="inner",
        validate="one_to_one",
    )

    matched_count = len(merged_df)
    total_evaluation_players = len(evaluation_df)
    print(
        f"Matched {matched_count} of {total_evaluation_players} evaluation player-seasons "
        f"to market data."
    )

    if matched_count == 0:
        raise ValueError("No players were matched between the evaluation and player market files.")

    if "performance_index" not in merged_df.columns:
        raise KeyError("performance_index column was not found in the evaluation metrics export.")

    # Filter strictly for players who have a valid performance index
    merged_df = merged_df.dropna(subset=["performance_index"]).copy()

    gfp_series = pd.to_numeric(merged_df["Goalunit Fair Price (GFP)"], errors="coerce")
    merged_df["vfm_ratio"] = pd.Series(np.nan, index=merged_df.index)
    valid_gfp_mask = gfp_series.notna() & gfp_series.gt(0)
    gfp_in_million = gfp_series.loc[valid_gfp_mask] / 1_000_000.0
    merged_df.loc[valid_gfp_mask, "vfm_ratio"] = (
        merged_df.loc[valid_gfp_mask, "performance_index"] / gfp_in_million
    )

    # merged_df["vfm_ratio"] = pd.to_numeric(merged_df["vfm_ratio"], errors="coerce")

    merged_df = add_vfm_score(merged_df)

    # Ranking uses the scale-free VFM score; `vfm_ratio` remains only as a legacy comparison column.
    merged_df["vfm_rank"] = (
        merged_df.groupby("tactical_cluster")["vfm_score"]
        .rank(method="dense", ascending=False, na_option="keep")
        .astype("Int64")
    )

    export_columns = [
        "playerId",
        "seasonId",
        "seasonName",
        "playerName",
        "clubName",
        "age",
        "tactical_cluster",
        "evaluation_role",
        "performance_index",
        "Goalunit Fair Price (GFP)",
        "performance_z",
        "log_price_z",
        "vfm_score",
        "vfm_ratio",
        "vfm_rank",
    ]

    export_df = merged_df.loc[:, export_columns].copy()
    export_df = export_df.sort_values(["tactical_cluster", "vfm_rank"]).reset_index(drop=True)
    export_df.to_csv(OUTPUT_CSV, index=False)

    # --- INSERT CORRELATION CHECK HERE ---
    print("\n--- Spearman Correlation: Performance vs. Fair Price ---")
    cluster_correlations = (
        export_df.groupby("evaluation_role")[["performance_index", "Goalunit Fair Price (GFP)"]]
        .corr(method="spearman")
        .iloc[0::2, 1]
    )
    # The .rename removes the messy "Goalunit Fair Price (GFP)" column header from the print output
    print(cluster_correlations.rename(index={"performance_index": ""}).to_string(name=False))
    # -------------------------------------

    print(f"\n--- Top Value-for-Money Targets by Role ---")
    top_vfm = export_df[export_df["vfm_rank"] <= 3]
    for role, group in top_vfm.groupby("evaluation_role"):
        print(f"\nRole: {role}")
        print(group[["playerName", "seasonName", "age", "performance_index", "Goalunit Fair Price (GFP)", "vfm_score", "vfm_rank"]].to_string(index=False))
    print("-------------------------------------------\n")

    print(f"Exported {len(export_df):,} VFM rows to {OUTPUT_CSV}")

    figure = build_vfm_figure(export_df)
    figure.write_html(SCATTER_HTML, include_plotlyjs="cdn")
    print(f"Wrote interactive VFM scatter to {SCATTER_HTML}")


if __name__ == "__main__":
    main()
