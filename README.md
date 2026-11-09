# Tennis Match Outcome Model (ATP, XGBoost)

An XGBoost classifier that predicts the winner of an ATP match from the difference between the two players' attributes and match statistics. It uses ATP match files from 1968–2024. Everything runs in one Colab notebook, `tennis_prediction_model.ipynb`.

The model scores 92.9% test accuracy and ROC-AUC 0.9823. Most of the features are serve statistics from the match being predicted. The model therefore explains a result after the match is over and cannot forecast one before it starts. See [Limitations](#limitations-read-this-before-using-the-numbers).

## Data

`data/atp_matches_YYYY.csv` holds one file per year from 1968 to 2024 (57 files). The notebook concatenates all of them, giving 194,996 matches × 49 columns. It then drops rows that are missing serve statistics, rank, rank points, age or height, which leaves **95,375 matches**.

## Features

Every feature is a winner-minus-loser difference.

| Group | Features | Available before the match? |
|---|---|---|
| Player profile | `RANK_DIFF`, `AGE_DIFF`, `HT_DIFF` | Yes |
| Head-to-head | `H2H_WIN`, `H2H_WIN_SURFACE` (running win counts per pair, overall and per surface) | Meant to be, but see below |
| In-match serve stats | `ACE_DIFF`, `DF_DIFF`, `FIRST_SERVE_PCT_DIFF`, `FIRST_SERVE_WIN_DIFF`, `SECOND_SERVE_WIN_DIFF`, `BP_SAVE_PCT_DIFF` | **No**, these come from the match itself |

`POINT_DIFF`, `BEST_OF` and `DRAW_SIZE` are computed but not used. Surface and tournament level are only used to key the surface H2H counter.

**Symmetric rows.** The raw data is always ordered winner/loser, so the notebook duplicates each match. One copy has the winner as player 1 (`TARGET=1`). The other has the loser as player 1, with every difference feature negated (`TARGET=0`). This produces 190,750 rows with balanced classes and stops the model from learning column position.

```
95,375 matches ──► diff features ──► mirror rows (190,750, 50/50 labels)
                                           │
                     random split 60 / 20 / 20 (train / test / validation)
                                           │
                               XGBClassifier (default params)
```

## Results

All numbers come from the notebook outputs.

| Split | Accuracy |
|---|---|
| Train | 94.91% |
| Test (38,150 rows) | 92.91% |
| Validation (named "Cross-Validation" in the notebook; it is a single held-out split) | 92.93% |

Test ROC-AUC is 0.9823. Precision, recall and F1 are 0.93 for both classes.

In the notebook's feature-importance plot, `FIRST_SERVE_WIN_DIFF` (about 0.56) and `SECOND_SERVE_WIN_DIFF` (about 0.26) dominate. Rank, age, height and H2H each contribute very little.

## Limitations (read this before using the numbers)

I checked the notebook for leakage. It has three problems, and they explain why the scores are this high.

1. **Serve and ace features come from the match being predicted.** `ACE_DIFF`, `DF_DIFF`, the first- and second-serve percentages and break-point save % are built from `w_ace`, `l_ace`, `w_1stWon` and the other columns of the same row. Those values only exist once the match is finished. Winning a higher share of serve points than your opponent almost defines winning the match, which is why these two features account for over 80% of importance. The model answers "given the box score, who won?" It does not answer "who will win?"
2. **Head-to-head counts include future matches.** The H2H loop updates its counters as it walks through the rows, which is correct in principle. But the files are loaded from 2024 down to 1968, and within a file the final comes first. So each match's "prior" H2H actually counts the pair's *later* meetings.
3. **The mirrored rows leak across the split.** Both copies of a match go into one pool before the random split. For many test rows, the exact negated twin is in the training set.

To turn this into a real pre-match predictor:
- Replace the in-match stats with each player's rolling serve averages over their previous N matches, computed strictly before `tourney_date`.
- Sort by `tourney_date` and `match_num` before computing H2H.
- Split by time (e.g. train ≤ 2021, test 2023–2024), keeping both mirrors of a match on the same side.
- Add pre-match context that is already in the data: rank points, surface, tournament level, best-of.

A model built that way will score well below 92.9%. That lower number would be the honest one.

## How to run

The notebook was written for Google Colab and reads from `/content/data/`.

1. Open `tennis_prediction_model.ipynb` in Colab (there is a badge at the top of the notebook).
2. Upload the `data/` folder to `/content/data/`.
3. Run all cells.

To run locally, change the paths in cell 2 to `data/atp_matches_<year>.csv` and install the packages that are actually used:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tqdm
```

`requirement.txt` is a full `pip list` dump of the Colab environment (more than 600 packages). Use it as a version reference only, and don't install from it.

## License

MIT
