# nbs/

Curated exploration and the paper's aggregation notebook. Earlier exploratory notebooks (cancer-type classification, TCGA tutorials, transfer-learning experiments, etc.) have been moved under `archive/nbs/` — see `archive/README.md` for what was archived and why.

## Paper artifact

| Notebook | Purpose |
|---|---|
| [`post-processing.ipynb`](post-processing.ipynb) | **The paper's aggregation pipeline.** Cells 41/46/48 produce the published results table from the per-split `test_scores.csv` files under `projects/bin_rsp_drug_pairs_all_samples/runs_*/` plus the external LGBM scores. See `PAPER.md` for the full mapping. `rerun_paper_aggregation.py` at the repo root replicates this aggregation as a standalone script. |

## Data preparation history

These predate the script-based pipeline in `scripts/` but document how the merged dataset and TFRecords were built. Useful reference; not load-bearing for the paper.

| Notebook | Purpose |
|---|---|
| [`02_merge_meta_files.ipynb`](02_merge_meta_files.ipynb) | Merges three metadata files into `data/meta/meta_merged.csv`. |
| [`03_build_df.ipynb`](03_build_df.ipynb) | Builds the unified dataframe saved as `data/data_merged.csv`. |
| [`pdx_meta_files.ipynb`](pdx_meta_files.ipynb) | Exploration of the cross-reference (Cref) and PDX metadata files. |
| [`pdx_to_tfrecords.ipynb`](pdx_to_tfrecords.ipynb) | Generating TFRecords from PDX slide tiles. |
| [`eda_splits.ipynb`](eda_splits.ipynb) | EDA on the train/val/test splits. |

## Visualization and reference

| Notebook | Purpose |
|---|---|
| [`PDX_FIXED_RSP.ipynb`](PDX_FIXED_RSP.ipynb) | Visualizes tile images decoded from TFRecords. |
| [`AUC_example.ipynb`](AUC_example.ipynb) | AUC interpretation reference (balanced vs imbalanced cases). |
