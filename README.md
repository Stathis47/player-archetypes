# Player Archetypes: Role-Based Player Evaluation and Value for Money

Which players give a club the most performance for their price, *within a given role*?

Positional labels such as "right-back" hide very different jobs: one fullback overlaps and crosses, another mostly defends and wins the ball back. Comparing them on one scale is unfair to both and hides what each one is actually worth. This project groups players by how they actually play, judges each only against others doing the same job, and then sets that role-specific performance against a market price.

## Approach

The pipeline has three steps, run on Premier League player-seasons from 2023/24 and 2024/25 (outfield players with more than 900 minutes, 614 player-seasons).

1. **Tactical profiling.** Each player-season gets a behavioural fingerprint from Wyscout event data: passing, carrying, shooting, headers, pitch zones and lanes, defensive actions and discipline. PCA reduces it, and K-Means groups players into nine tactical archetypes (for example Progressive Fullbacks, Wingers and Target Men). Profiling ignores outcomes on purpose, so it describes style, not quality.
2. **Role-specific evaluation.** Within an archetype, metrics are possession-adjusted per 90, standardised against the archetype's own players, weighted by what matters for the role, and rescaled to a 0-100 Performance Index.
3. **Value for money.** The Performance Index is set against Goalunit Fair Price (GFP) within each archetype, so a fullback is only ever compared with fullbacks. The ranking uses the difference between standardised performance and standardised log price.

The worked example is the **Progressive Fullbacks** archetype.

The optional t-SNE scripts in `t-sne visuals/` produce a 2D and a 3D map of the archetypes. They are used for visualisation only; the clustering is done on the PCA scores.

## Repository structure

| File | Purpose |
|---|---|
| `convert_events_csv_to_parquet.py` | One-off, lossless conversion of the raw `events.csv` into `events.parquet` |
| `enrich_events_pipeline.py` | Reads the raw events and precomputes tactical flags (progressive actions, crosses, duels, recoveries) into `events_enriched.parquet` |
| `player_profile_pipeline.py` | Step 1: behavioural features, PCA, K-Means, archetype assignment |
| `player_evaluation_pipeline.py` | Step 2: possession-adjusted metrics and the role-specific Performance Index |
| `vfm_pipeline.py` | Step 3: value-for-money score and ranking |
| `fullbacks_vfm_scatter.py` | Performance versus price scatter for the Progressive Fullbacks |
| `t-sne visuals/` | Optional 2D and 3D t-SNE maps of the archetypes |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for the data paths |

## Data

The data was provided by Goalunit and is **not included**. Put these files in one folder:

- **`events.csv`:** Wyscout Premier League events for 2023/24 (`seasonId` 862) and 2024/25 (`seasonId` 1773). The first script converts it to `events.parquet`, which every later step reads.
- **`players.csv`:** Goalunit Fair Price (GFP), date of birth and valuation snapshot date. Only `vfm_pipeline.py` reads it.

Copy `.env.example` to `.env` and set `GOALUNIT_DATA_DIR` to that folder. If it is unset, the scripts look in a `data/` folder inside the repository. Individual files can be overridden with the other `GOALUNIT_*` variables listed in `.env.example` and at the top of each script. Generated files (parquet, CSV, HTML, PNG, XLSX) are git-ignored.

## How to run

Run from the repository root.

```bash
pip install -r requirements.txt
cp .env.example .env     # then edit the paths

python convert_events_csv_to_parquet.py   # once: events.csv -> events.parquet
python enrich_events_pipeline.py
python player_profile_pipeline.py
python player_evaluation_pipeline.py
python vfm_pipeline.py
python fullbacks_vfm_scatter.py

# optional t-SNE maps
python "t-sne visuals/plot_tactical_tsne_2d.py"
python "t-sne visuals/plot_tactical_tsne_3d.py"
```

The conversion script will not overwrite an existing `events.parquet` unless you pass `--overwrite`.

**Important:** clustering is seeded (`random_state=42`), so the same data and library versions give the same cluster numbers. If you change the input data, the feature set, `k` or the seed, the numbering can change. In that case update the hardcoded `ROLE_CLUSTERS` mapping in `player_evaluation_pipeline.py` so each role points to the right cluster, then re-run the later steps.

## Limitations

- The profiling uses event data only, which limits how finely archetypes can be separated. For example, without off-ball information, fullbacks who move inside during possession can only be seen indirectly through lane shares, and one archetype (Ball-Winning Midfielders / Defensive Fullbacks) mixes midfielders and fullbacks. Richer inputs, such as tracking data or more detailed event tags, should allow more distinct roles.
- The number of archetypes (k = 9) was chosen by judgement alongside stability and silhouette diagnostics.
- The metric weights in the Performance Index are set by hand.
- Scores are relative to the archetype, not absolute: a high score means "strong among players doing this job", not "strong for the league".
- Goalunit Fair Price is a proprietary estimate and cannot be reproduced from this repository. Players on free transfers have no price and no value-for-money score.

## Acknowledgements

The event and player data, including Goalunit Fair Price, were provided by Goalunit.

## Author

Efstathios Papadopoulos, Stockholm.