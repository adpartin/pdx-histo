# Repository Audit — Open Items

Open items remaining after polish/archive passes. Resolved items have been
removed from this doc — see `git log` for what was actioned.

## Reclassified — keep, do not archive

These files initially looked like dead weight but are load-bearing for the paper. Documented here so future cleanup passes don't reflag them.

- `src/post_processing.py`, `src/eda.py` — imported from `nbs/post-processing.ipynb` (paper aggregation).
- `scripts/tile_dd1_dd2.bash` — paper UMH-Net baseline.
- `scripts/ge_dd1_dd2_drop_aug.bash`, `scripts/ge_dd1_dd2_only_pairs.bash` — paper ablations (UME-Net_org, UME-Net_pairs).
- `projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd/`, `runs_ge_dd/`, `runs_tile_dd/` — paper sweeps consumed by `rerun_paper_aggregation.py`.

---

## 1. Hardcoded local paths (live tree)

| File | Line | Path | Recommendation |
|---|---|---|---|
| `conda_create.sh` | 16 | `source ~/miniconda3/etc/profile.d/conda.sh` | OK for a shell installer; consider noting that miniconda must live at `~/miniconda3` or document an override. |
| `src/config.py` | 19, 30–41 | A dozen commented-out `cfg.SF_TFR_DIR*` definitions | Delete the commented block — git history preserves them. |

---

## 2. Documentation polish

| Issue | Where | Recommendation |
|---|---|---|
| No CI configuration | `.github/workflows/` | Optional, but a one-step `pytest` smoke run would signal maintained code. |

---

## Live-tree paper artifact (do not remove)

- **Bash:** `scripts/tile_ge_dd1_dd2.bash` (MM-Net), `ge_dd1_dd2.bash` (UME-Net), `tile_dd1_dd2.bash` (UMH-Net), `ge_dd1_dd2_drop_aug.bash` (UME-Net_org), `ge_dd1_dd2_only_pairs.bash` (UME-Net_pairs), `build_tidy_drug_pairs_all_samples.py`, `update_tfrecords.py`.
- **Python:** `src/trn_multimodal.py`, `src/models.py`, `src/sf_utils.py`, `src/tfrecords.py`, `src/load_data.py`, `src/datasets/tidy.py`, `src/config.py`, `src/post_processing.py`, `src/eda.py`, `src/tf_utils.py`, `src/ml/{evals,keras_utils,scale,data}.py`, `src/utils/{utils,classlogger,plots}.py`.
- **Data:** `default_params/` templates, `projects/bin_rsp_drug_pairs_all_samples/runs_*/` (paper sweeps), external `data/PDX_Transfer_Learning_Classification/.../1.0_True_False_100_31/` (LGBM scores).
- **Notebooks:** `nbs/post-processing.ipynb` (paper aggregation).
- **Reproducibility:** `rerun_paper_aggregation.py`, `PAPER.md`, `CLAUDE.md`, `archive/README.md`.
