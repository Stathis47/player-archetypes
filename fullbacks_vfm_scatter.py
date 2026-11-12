"""Presentation-ready static plots built from the VFM pipeline outputs.

Currently renders the Attacking Full-Back / Wing-Back value-for-money scatter
(performance index vs. inverted Goalunit Fair Price, coloured by age) as a
high-resolution PNG for slides. The price axis shows the fair price in millions
on a logarithmic scale; the VFM ranking behind the chart uses the standardised
log price (see `vfm_pipeline.add_vfm_score`). The figure is the same one produced by
`vfm_pipeline.build_vfm_figure`; this script restricts it to one tactical role
and adds player-name labels for the standout value options only.

Labelled players:
    * the best value-for-money player-seasons (lowest rank = highest VFM score);
    * young (<= `YOUNG_MAX_AGE`) player-seasons performing above the role median,
      ranked by VFM ratio - the "future profit" candidates.

Label positions are chosen greedily so that no label overlaps a marker, another
label, the value-zone caption, or the plot edge.

Requires `kaleido` for PNG export (`pip install kaleido`).

Usage:
    python presentation_plots.py
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from player_evaluation_pipeline import FULL_BACK_ROLE
from vfm_pipeline import GFP_COLUMN, build_vfm_figure, prepare_plot_data

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
VFM_RESULTS_CSV: Final[Path] = Path(
    os.getenv("GOALUNIT_VFM_OUTPUT_CSV", str(PROJECT_ROOT / "vfm_results.csv"))
)
OUTPUT_PNG: Final[Path] = Path(
    os.getenv(
        "GOALUNIT_PRESENTATION_FULLBACKS_PNG",
        str(PROJECT_ROOT / "fullbacks_vfm_scatter.png"),
    )
)

# Single source of truth: the wordalized cluster name the evaluation pipeline scores this role under.
ROLE_NAME: Final[str] = FULL_BACK_ROLE
TOP_VFM_LABELS: Final[int] = 5
MIN_AGE: Final[float] = 18.0
YOUNG_MAX_AGE: Final[float] = 25.0
YOUNG_LABELS: Final[int] = 5

# Layout in CSS pixels; the PNG is exported at `EXPORT_SCALE` x this size.
FIGURE_WIDTH: Final[int] = 1600
FIGURE_HEIGHT: Final[int] = 1000
EXPORT_SCALE: Final[int] = 2
MARGIN: Final[dict[str, int]] = dict(l=100, r=150, t=150, b=100)
MARKER_SIZE: Final[int] = 16
LABEL_FONT_SIZE: Final[int] = 16
AXIS_FONT_SIZE: Final[int] = 16
LABEL_PADDING_PX: Final[float] = 6.0
# Rough average glyph width of bold Arial as a fraction of the font size.
CHAR_WIDTH_RATIO: Final[float] = 0.6
# Footprint of the "Value zone" caption anchored at the plot's top-right corner.
VALUE_ZONE_SIZE_PX: Final[tuple[float, float]] = (300.0, 60.0)
RING_RADII_PX: Final[range] = range(20, 241, 10)
RING_ANGLES_DEG: Final[range] = range(0, 360, 10)


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in plot-pixel coordinates (origin top-left)."""

    left: float
    top: float
    right: float
    bottom: float

    def overlaps(self, other: Box) -> bool:
        """Return True when the two rectangles intersect."""
        return not (
            self.right <= other.left
            or other.right <= self.left
            or self.bottom <= other.top
            or other.bottom <= self.top
        )

    def overlaps_circle(self, center_x: float, center_y: float, radius: float) -> bool:
        """Return True when a circle intersects the rectangle."""
        nearest_x = min(max(center_x, self.left), self.right)
        nearest_y = min(max(center_y, self.top), self.bottom)
        return math.hypot(nearest_x - center_x, nearest_y - center_y) < radius


# def select_label_players(role_df: pd.DataFrame) -> pd.DataFrame:
#     """Pick the best-VFM players plus the best young, above-median performers."""
#     ranked_df = role_df.dropna(subset=["vfm_ratio"])
#     top_vfm = ranked_df.nsmallest(TOP_VFM_LABELS, "vfm_rank")

#     median_index = role_df["performance_index"].median()
#     young_pool = ranked_df.loc[
#         ranked_df["age"].le(YOUNG_MAX_AGE) & ranked_df["performance_index"].ge(median_index)
#     ]
#     young_value = young_pool.nlargest(YOUNG_LABELS, "vfm_ratio")

#     # Priority order: top VFM first so they claim the best label slots.
#     return pd.concat([top_vfm, young_value]).drop_duplicates(subset=["playerId", "seasonId"])

def select_label_players(role_df: pd.DataFrame) -> pd.DataFrame:
    """Label every plotted player-season."""
    return role_df.drop_duplicates(subset=["playerId", "seasonId"])


def select_highlighted_players(role_df: pd.DataFrame) -> pd.DataFrame:
    """Select a specific list of players to label on the plot (unused when labelling all)."""
    # Define the exact players you want to highlight
    target_players = [
        "Milos Kerkez",
        #"Bilal El Khannouss",
        "Issa Kaboré", 
        "Lewis Hall",
        "Josko Gvardiol",
        #"Malo Gusto",
        #"Neco Williams",
        #"Kieran Trippier",
        "Mads Roerslev",
        "Tariq Lamptey",
        "Vitinho",
        "Luke Thomas"
    ]
    
    # Filter the dataframe to only label these specific players
    return role_df[role_df["playerName"].isin(target_players)].drop_duplicates(subset=["playerId", "seasonId"])


def build_label_texts(label_df: pd.DataFrame) -> list[str]:
    """Player names, with a short season suffix only where a name appears twice."""
    is_repeated = label_df["playerName"].duplicated(keep=False)
    return [
        f"{row.playerName} ({str(row.seasonName)[2:4]}/{str(row.seasonName)[7:9]})"
        if repeated
        else str(row.playerName)
        for row, repeated in zip(label_df.itertuples(), is_repeated)
    ]


def _to_pixels(
    x_value: float, y_value: float, x_range: list[float], y_range: list[float]
) -> tuple[float, float]:
    """Map axis coordinates to plot-area pixels (origin top-left, reversed y-axis).

    The price axis is logarithmic, so `y_value` and `y_range` are in log10 units.
    Plotly lists a y-range bottom-to-top, so `y_range[1]` is the top edge.
    """
    plot_width = FIGURE_WIDTH - MARGIN["l"] - MARGIN["r"]
    plot_height = FIGURE_HEIGHT - MARGIN["t"] - MARGIN["b"]
    pixel_x = (x_value - x_range[0]) / (x_range[1] - x_range[0]) * plot_width
    pixel_y = (y_value - y_range[1]) / (y_range[0] - y_range[1]) * plot_height
    return pixel_x, pixel_y


def _segment_hits_circle(
    start: tuple[float, float], end: tuple[float, float], center: tuple[float, float], radius: float
) -> bool:
    """Return True when the segment start->end passes within `radius` of `center`."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return False
    t = max(0.0, min(1.0, ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / length_sq))
    return math.hypot(start[0] + t * dx - center[0], start[1] + t * dy - center[1]) < radius


def place_labels(
    marker_pixels: list[tuple[float, float]],
    label_pixels: list[tuple[float, float]],
    label_texts: list[str],
) -> list[tuple[float, float]]:
    """Greedy, collision-free label placement; returns each label's pixel offset.

    For every label (in priority order) the closest ring position whose text box
    clears all markers, previously placed labels, the value-zone caption and the
    plot edges is chosen, preferring one whose leader line avoids other markers.
    Raises if no collision-free position exists, rather than drawing a messy label.
    """
    plot_width = FIGURE_WIDTH - MARGIN["l"] - MARGIN["r"]
    plot_height = FIGURE_HEIGHT - MARGIN["t"] - MARGIN["b"]
    plot_area = Box(0.0, 0.0, plot_width, plot_height)
    value_zone = Box(
        plot_width - VALUE_ZONE_SIZE_PX[0], 0.0, plot_width, VALUE_ZONE_SIZE_PX[1]
    )

    overpriced_zone = Box(
            0.0, plot_height - VALUE_ZONE_SIZE_PX[1], VALUE_ZONE_SIZE_PX[0], plot_height
        )
    
    marker_radius = MARKER_SIZE / 2 + 3.0

    placed_boxes: list[Box] = [value_zone, overpriced_zone]

    offsets: list[tuple[float, float]] = []

    for (label_x, label_y), text in zip(label_pixels, label_texts):
        half_width = len(text) * LABEL_FONT_SIZE * CHAR_WIDTH_RATIO / 2 + LABEL_PADDING_PX
        half_height = LABEL_FONT_SIZE * 1.3 / 2 + LABEL_PADDING_PX

        others = [pixel for pixel in marker_pixels if pixel != (label_x, label_y)]
        chosen: tuple[float, float] | None = None
        # Pass 1 also keeps the leader line clear of neighbouring markers; pass 2 relaxes that.
        for require_clear_line in (True, False):
            for radius in RING_RADII_PX:
                for angle in RING_ANGLES_DEG:
                    angle_rad = math.radians(angle)
                    # Push the box centre out so its near edge, not its centre, sits at `radius`.
                    center_x = label_x + (radius + half_width) * math.cos(angle_rad)
                    center_y = label_y - (radius + half_height) * math.sin(angle_rad)
                    box = Box(
                        center_x - half_width, center_y - half_height,
                        center_x + half_width, center_y + half_height,
                    )
                    if not (
                        box.left >= plot_area.left and box.right <= plot_area.right
                        and box.top >= plot_area.top and box.bottom <= plot_area.bottom
                    ):
                        continue
                    if any(box.overlaps(placed) for placed in placed_boxes):
                        continue
                    if any(box.overlaps_circle(mx, my, marker_radius) for mx, my in others):
                        continue
                    if require_clear_line and any(
                        _segment_hits_circle(
                            (label_x, label_y), (center_x, center_y), pixel, marker_radius
                        )
                        for pixel in others
                    ):
                        continue
                    chosen = (center_x, center_y)
                    break
                if chosen is not None:
                    break
            if chosen is not None:
                break

        if chosen is None:
            raise RuntimeError(f"No free position found for the label '{text}'.")
        placed_boxes.append(
            Box(chosen[0] - half_width, chosen[1] - half_height,
                chosen[0] + half_width, chosen[1] + half_height)
        )
        offsets.append((chosen[0] - label_x, chosen[1] - label_y))
    return offsets


def build_fullback_figure(vfm_df: pd.DataFrame) -> go.Figure:
    """Attacking full-back VFM scatter with readable labels on the standout players."""
    vfm_df = vfm_df.loc[vfm_df["age"].between(MIN_AGE, YOUNG_MAX_AGE)]
    role_df = prepare_plot_data(vfm_df).loc[lambda df: df["evaluation_role"].eq(ROLE_NAME)]
    if role_df.empty:
        raise ValueError(
            f"No plottable players aged {MIN_AGE:g}-{YOUNG_MAX_AGE:g} found for role '{ROLE_NAME}'."
        )
    cluster = int(role_df["tactical_cluster"].iloc[0])

    figure = build_vfm_figure(vfm_df, default_cluster=cluster)
    figure.data = tuple(trace for trace in figure.data if trace.visible is True)
    figure.data[0].marker.size = MARKER_SIZE
    figure.data[0].marker.colorbar = dict(
        title=dict(text="Age", font=dict(size=AXIS_FONT_SIZE)),
        thickness=20, len=0.9, tickfont=dict(size=AXIS_FONT_SIZE - 2),
    )

    x_range = list(figure.layout.xaxis.range)
    y_range = list(figure.layout.yaxis.range)
    marker_pixels = [
        _to_pixels(x, float(np.log10(y)), x_range, y_range)
        for x, y in zip(role_df["performance_index"], role_df["price_m"])
    ]
    pixel_by_key = dict(zip(zip(role_df["seasonId"], role_df["playerId"]), marker_pixels))

    label_df = select_label_players(role_df)
    label_pixels = [pixel_by_key[key] for key in zip(label_df["seasonId"], label_df["playerId"])]
    label_texts = build_label_texts(label_df)
    offsets = place_labels(marker_pixels, label_pixels, label_texts)

    annotations = list(figure.layout.annotations)
    for row, text, (offset_x, offset_y) in zip(label_df.itertuples(), label_texts, offsets):
        annotations.append(
            dict(
                # Annotation coordinates on a log axis are expressed in log10 units.
                x=row.performance_index, y=float(np.log10(row.price_m)), xref="x", yref="y",
                ax=offset_x, ay=offset_y, axref="pixel", ayref="pixel",
                text=f"<b>{text}</b>", showarrow=True,
                xanchor="center", yanchor="middle",
                arrowhead=0, arrowwidth=1.2, arrowcolor="rgba(0,0,0,0.55)", standoff=8,
                font=dict(size=LABEL_FONT_SIZE, color="#000000", family="Arial"),
                bgcolor="rgba(255,255,255,0.85)", borderpad=LABEL_PADDING_PX / 2,
            )
        )
    for annotation in annotations[: len(figure.layout.annotations)]:
        if annotation.text and "Value Zone" in annotation.text:
            annotation.font.size = AXIS_FONT_SIZE + 2

    figure.layout.updatemenus = []
    figure.update_layout(
        annotations=annotations,
        title=dict(
            text=(
                f"<b>{ROLE_NAME}: Performance vs. Fair Price</b><br>"
                f"<sup>Top-right = high performance at a low price · colour = age at valuation · "
                f"players aged {MIN_AGE:g}-{YOUNG_MAX_AGE:g} only · dashed lines = medians<br>"
                f"Price axis: fair price in millions, log scale · VFM score = z(performance index) "
                f"− z(ln fair price), standardised within the role</sup>"
            ),
            x=0.02, font=dict(size=26),
        ),
        xaxis=dict(title=dict(text="Performance index", font=dict(size=AXIS_FONT_SIZE + 2)),
                   tickfont=dict(size=AXIS_FONT_SIZE)),
        yaxis=dict(title=dict(text=f"{GFP_COLUMN} (M, log scale) - inverted",
                              font=dict(size=AXIS_FONT_SIZE + 2)),
                   tickfont=dict(size=AXIS_FONT_SIZE)),
        width=FIGURE_WIDTH, height=FIGURE_HEIGHT, margin=MARGIN,
    )
    return figure


def main() -> None:
    """Render the attacking full-back VFM scatter to a PNG."""
    if not VFM_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{VFM_RESULTS_CSV} was not found. Run vfm_pipeline.py first."
        )
    vfm_df = pd.read_csv(VFM_RESULTS_CSV)
    figure = build_fullback_figure(vfm_df)
    figure.write_image(OUTPUT_PNG, width=FIGURE_WIDTH, height=FIGURE_HEIGHT, scale=EXPORT_SCALE)
    print(f"Wrote presentation plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
