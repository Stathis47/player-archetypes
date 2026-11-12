"""Standalone t-SNE tactical-archetype topology map exporter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
INPUT_FILE: Path = REPO_ROOT / "final_tactical_clusters.csv"
OUTPUT_FILE: Path = REPO_ROOT / "tactical_tsne_map_2d.html"
TSNE_COORDINATES_FILE: Path = REPO_ROOT / "tactical_tsne_coordinates.csv"
REQUIRED_COLUMNS: set[str] = {"tactical_cluster", "playerName"}
RANDOM_STATE: int = 42

# Hardcoded football role names per K-Means cluster id. K-Means ids are arbitrary,
# so update this mapping by hand after every re-run of `player_profile_pipeline.py`.
CLUSTER_NAMES: dict[int, str] = {
    0: "Ball-Playing Centre-Backs",
    1: "Mobile Strikers",
    2: "Creative Midfielders (No. 10s)",
    3: "Progressive Fullbacks",
    4: "Deep-Lying Playmakers",
    5: "Wingers",
    6: "Aerial Stopper Centre-Backs",
    7: "Defensive Utility Players",
    8: "Target Men",
}


def _get_pca_columns(df: pd.DataFrame) -> list[str]:
    """Return PCA columns ordered numerically by component number."""
    pca_columns = [column for column in df.columns if column.startswith("PC")]
    return sorted(pca_columns, key=lambda column: int(column.removeprefix("PC")))


def export_tsne_map(df: pd.DataFrame) -> None:
    """Project PCA features with t-SNE and export an interactive topology map.

    Args:
        df: Player-level dataframe containing PCA components, tactical_cluster,
            and playerName.

    Raises:
        ValueError: If required columns are absent, no PCA columns exist, or
            the dataframe does not contain enough rows for t-SNE.
    """
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    pca_columns = _get_pca_columns(df)
    if not pca_columns:
        raise ValueError("Input DataFrame must contain at least one column beginning with 'PC'.")
    if len(df) < 3:
        raise ValueError("t-SNE requires at least three player records.")

    perplexity = min(30.0, float(len(df) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    )

    plot_df = df.copy()
    projection = tsne.fit_transform(plot_df[pca_columns])
    plot_df["tSNE_1"] = projection[:, 0]
    plot_df["tSNE_2"] = projection[:, 1]

    identifier_columns = [
        column
        for column in ("seasonId", "seasonName", "playerId", "playerName", "clubName", "tactical_cluster")
        if column in plot_df.columns
    ]
    plot_df[identifier_columns + ["tSNE_1", "tSNE_2"]].to_csv(TSNE_COORDINATES_FILE, index=False)

    cluster_ids = sorted(plot_df["tactical_cluster"].dropna().astype(int).unique())
    football_labels = {cid: f"{cid} - {name}" for cid, name in CLUSTER_NAMES.items()}
    unlabelled = [cid for cid in cluster_ids if cid not in football_labels]
    if unlabelled:
        print(f"Warning: no name in CLUSTER_NAMES for clusters {unlabelled}; add them to the mapping.")

    # Top 3 player-seasons per cluster, ranked by Euclidean distance to the cluster
    # centroid in PCA space (rank 0 = closest). Rows are tracked by index, so a player
    # appearing in both seasons is never confused with their other season.
    top_rows_by_cluster: dict[int, list] = {}
    for cluster_id in cluster_ids:
        cluster_data = df[df["tactical_cluster"] == cluster_id]
        centroid = cluster_data[pca_columns].mean().values
        distances = np.linalg.norm(cluster_data[pca_columns].values - centroid, axis=1)
        top_rows_by_cluster[cluster_id] = cluster_data.index[np.argsort(distances)[:3]].tolist()

    def _player_label(row_index) -> str:
        name = plot_df.at[row_index, "playerName"]
        season = plot_df.at[row_index, "seasonName"] if "seasonName" in plot_df.columns else None
        return f"{name} ({season})" if pd.notna(season) else str(name)

    # Legend entries list the football role name followed by its three most representative players.
    cluster_names = {
        cluster_id: f"{football_labels.get(cluster_id, f'Cluster {cluster_id}')}<br>"
        + "<br>".join(
            f"  {rank + 1}. {_player_label(row_index)}"
            for rank, row_index in enumerate(top_rows_by_cluster[cluster_id])
        )
        for cluster_id in cluster_ids
    }
    plot_df["tactical_cluster"] = plot_df["tactical_cluster"].astype("Int64").map(cluster_names)
    ordered_clusters = [cluster_names[cluster_id] for cluster_id in cluster_ids]

    fig = px.scatter(
        plot_df,
        x="tSNE_1",
        y="tSNE_2",
        color="tactical_cluster",
        hover_name="playerName",
        hover_data={"tactical_cluster": True, "tSNE_1": False, "tSNE_2": False},
        title="Tactical Archetype Topology: 2D t-SNE Projection",
        labels={
            "tSNE_1": "t-SNE Dimension 1",
            "tSNE_2": "t-SNE Dimension 2",
            "tactical_cluster": "Tactical Cluster",
        },
        category_orders={"tactical_cluster": ordered_clusters},
    )

    # Annotate the same top-3 player-seasons computed above for the legend.
    marquee_players_with_rank = [
        (row_index, rank)
        for row_indices in top_rows_by_cluster.values()
        for rank, row_index in enumerate(row_indices)
    ]

    # Define directional offsets for the annotations (ax = x-offset, ay = y-offset in pixels)
    # Rank 0: Top, Rank 1: Bottom-Right, Rank 2: Bottom-Left
    offsets = {
        0: {"ax": 0, "ay": -45},
        1: {"ax": 45, "ay": 35},
        2: {"ax": -45, "ay": 35}
    }

    for row_index, rank in marquee_players_with_rank:
        fig.add_annotation(
            x=plot_df.at[row_index, "tSNE_1"],
            y=plot_df.at[row_index, "tSNE_2"],
            text=plot_df.at[row_index, "playerName"],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="gray",
            ax=offsets[rank]["ax"],  # Dynamically assign X offset
            ay=offsets[rank]["ay"],  # Dynamically assign Y offset
            font=dict(size=9, color="black"),
            bgcolor="white",
            opacity=0.9,
            bordercolor="black",
            borderwidth=1
        )

    fig.update_traces(marker=dict(size=9, opacity=0.82))
    fig.update_xaxes(showticklabels=False, zeroline=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, zeroline=False, showgrid=False)
    fig.update_layout(legend_title_text="Tactical Cluster")
    fig.write_html(OUTPUT_FILE, include_plotlyjs="cdn", full_html=True)


if __name__ == "__main__":
    try:
        input_df = pd.read_csv(Path(INPUT_FILE))
        export_tsne_map(input_df)
        print(f"Exported: {Path(OUTPUT_FILE).resolve()}")
    except FileNotFoundError:
        print(f"Input file not found: {Path(INPUT_FILE).resolve()}")
    except ValueError as e:
        print(f"Error: {e}")
