# archive/

Files in this directory are **not part of the paper pipeline** but are preserved here (rather than deleted) so they remain in `git log` and are one `git mv` away from being restored.

See `../PAPER.md` for what *is* the paper pipeline.

## How to restore something

```bash
git mv archive/path/to/file path/to/file
```

The original directory layout is mirrored under `archive/`, so the restore target is always the same path with `archive/` stripped off.

## What's here and why

### Top-level

| File | Why archived |
|---|---|
| `notes` | Pre-Makefile setup instructions, opens with "I haven't setup a working Makefile yet" — but `Makefile` now exists. |
| `notes_new` | 3-line scratchpad listing three filenames with typos. |
| `conda_create_on_lamina.sh` | Host-specific installer for the lamina cluster; `conda_create.sh` is the canonical one. |
| `conda_create_on_rbdgx.sh` | Host-specific installer for rbdgx; same reason as above. |

### scripts/

Non-paper experiment drivers and superseded data-prep scripts. The paper-pipeline bash scripts (`tile_ge_dd1_dd2.bash`, `ge_dd1_dd2.bash`, `tile_dd1_dd2.bash`, `ge_dd1_dd2_drop_aug.bash`, `ge_dd1_dd2_only_pairs.bash`) and the active data-prep (`build_tidy_drug_pairs_all_samples.py`, `update_tfrecords.py`) remain in `scripts/`.

| File | Why archived |
|---|---|
| `split_tmp.py` | Filename says "tmp"; had hardcoded `/vol/ml/apartin/...` path and 4 module-level ipdb hooks. |
| `grp_build_tidy_df_partially_balanced.py` | Near-duplicate of `build_tidy_df_partially_balanced.py`; no caller. |
| `build_tidy_df_all_samples.py` | Earlier dataset variant superseded by `build_tidy_drug_pairs_all_samples.py`. |
| `build_tidy_df_partially_balanced.py` | Same. |
| `build_tidy_single_drug_all_samples.py` | Same. |
| `baseline.bash` | Calls `src/trn_baseline.py` (also archived); paper UME-Net runs through `ge_dd1_dd2.bash` → `trn_multimodal.py`. |
| `ge.bash`, `tile_ge.bash`, `tape.bash` | Modality-ablation scripts not in the paper. |
| `split.bash`, `split_new.bash` | Reference `apps/` and `/vol/ml/...` paths; modern split workflow uses pre-computed `--split_id` via `trn_multimodal.py`. |
| `type_classifier.bash`, `type_classifier_gen.bash` | Cancer-type classifier experiment (not the drug-response paper). |
| `infer_serial.sh` | File header literally says `# TODO: didn't finish`. |

### src/

| Item | Why archived |
|---|---|
| `old/` (7 files) | Pre-refactor pipeline; explicitly labelled "old". |
| `proj/` (4 files) | `build_bin_balance_*.py` superseded by `scripts/build_tidy_*.py`. |
| `ml-data-splits/` | Vendored copy of an external splitter package; diverged from `src/datasplit/`. |
| `datasplit/` | Top-level splitter copy; `splitter.py` has a broken `from utils.plots import plot_hist` import. Only consumer was `cv_splits.py` (also archived). Paper splits are pre-baked CSVs under `data/processed/`. |
| `cv_splits.py` | Only importer of `src/datasplit/`; itself unimported anywhere. |
| `batch_train.py`, `batch_infer.py` | Docstrings describe batch wrappers around `trn_multimodal.py`, but no caller imports them; the actual sweep loop is in `scripts/tile_ge_dd1_dd2.bash`. |
| `trn_tape.py` | TAPE protein-language-model variant; not in paper. |
| `trn_baseline.py` | Standalone (non-TFRecord) baseline trainer; paper UME-Net uses `trn_multimodal.py` with `--use_ge --use_dd1 --use_dd2`. |
| `trn_ctype_cls.py`, `trn_ctype_cls_gen.py` | Cancer-type classifier; separate experiment from the drug-response paper. |
| `test_model_save.py` | Scratch validation script; misnamed (lives in `src/` but named like a test). |
| `ml/ml_models.py` | 30 KB of generic LightGBM/Keras model factories; unused. The paper's LGBM baseline is consumed as pre-computed scores from `data/PDX_Transfer_Learning_Classification/...`, not trained from this file. |
| `deephistopath/` (~2.6 MB) | Vendored copy of CMU's `deep-histopath` WSI processing library (tile extraction, slide I/O, filtering). Original tile-extraction stack from 2021; replaced by `src/sf_utils.py` (slideflow-derived) once the tiles were extracted into `data/PDX_FIXED/`. No live `.py` file imports it. Includes a hardcoded `SRC_TRAIN_DIR` in `wsi/slide.py:44` flagged in the original audit. |

### nbs/

| Item | Why archived |
|---|---|
| `Dockerfile`, `notes` | 7-line Ubuntu container that just `COPY notes` and runs bash; `notes` is a docker-CLI cheatsheet. |
| `triple-stratified-kfold-with-tfrecords.ipynb` | 0 of 17 cells executed — never ran. |
| `04-ctype_cls.ipynb`, `04-ctype_cls_img_sf.ipynb`, `04-ctype_cls_img_rna_sf.ipynb` | Cancer-type classifier exploration (not the drug-response paper); each has 1–2 stored exception tracebacks and 16–24 empty cells. |
| `01_get_meta_from_slides.ipynb`, `03-build_tfrec.ipynb`, `04-trn_nn.ipynb`, `tf_data.ipynb` | Depend on the archived `src/deephistopath/` package — pre-slideflow exploration that's no longer runnable without restoring deephistopath. |
| `update_tfrecords.ipynb` (Feb 2021) + `update_tfrecords_new.ipynb` (Apr 2021) | Pre-script exploration of adding RNA-seq metadata to TFRecords. `_new` is a successor of the original (less Colab boilerplate, helpers promoted into `src/tf_utils.py`, adds RNA-seq sanity-check cells). Both superseded by `scripts/update_tfrecords.py` which is what produced the paper's TFRecords. |
| `03-prep-annotations-for-slideflow.ipynb` | 0/17 cells executed — never ran. |
| `transfer_learning.ipynb` | 0/37 cells executed — never ran. |
| `tcga-img-eda.ipynb` | TCGA histolab tutorial; a different dataset (TCGA, not PDX) and tooling (histolab, not slideflow). 16 empty cells. |
| `pdx-img-eda.ipynb` | Histolab tutorial template applied to PDX slides. 15 empty cells. Tooling not used in the paper pipeline. |
| `04-ctype_rna.ipynb` | Cancer-type RNA classifier — a separate experiment from the drug-response paper. |
| `trns_lrn_pdx.ipynb` | Transfer-learning experiment on PDX; not part of the published paper. |
| `yitan_data.ipynb` | Investigation notes; the title opens as a personal note to a co-author. Pre-paper data-prep exploration. |
| `pdx_drug_response_meta.csv` (1.1 MB) | CSV checked into `nbs/`; should live under `data/`. |
| `tfr_from_csv/` (5 × ~5 MB tfrecords, 26 MB total) | TFRecord experiments; should live under `data/`. |

### tests/

The whole `tests/` directory was archived because none of the three files were actual tests — zero `def test_*` functions, no `pytest`/`unittest` imports.

| Item | Why archived |
|---|---|
| `test_seed.py` | 17 KB integration script with `argparse` that loads TFRecord data + builds the full multimodal model to investigate TF nondeterminism. Docstring opens with "TODO: problems reproducing the same data batches for every run!" — the property it tests is one the author couldn't achieve. |
| `seeds_in_tf.py` | Verbatim copy of TensorFlow documentation examples about `tf.random.set_seed`. Not a test of this project. |
| `test_callbacks.py` | Downloads the Kaggle credit-card-fraud dataset from Google Storage to investigate Keras callback behavior on the canonical TF tutorial example. Unrelated to drug response. |

## What was NOT archived (per `PAPER.md`)

The paper pipeline keeps:
- Bash: `scripts/build_tidy_drug_pairs_all_samples.py`, `scripts/update_tfrecords.py`, `scripts/tile_ge_dd1_dd2.bash`, `scripts/ge_dd1_dd2.bash`, `scripts/tile_dd1_dd2.bash`, `scripts/ge_dd1_dd2_drop_aug.bash`, `scripts/ge_dd1_dd2_only_pairs.bash`.
- Python: `src/trn_multimodal.py`, `src/models.py`, `src/sf_utils.py`, `src/tfrecords.py`, `src/load_data.py`, `src/datasets/tidy.py`, `src/config.py`, `src/post_processing.py`, `src/eda.py`, `src/tf_utils.py`, `src/ml/{evals,keras_utils,scale,data}.py`, `src/utils/{utils,classlogger,plots}.py`.
- Notebooks: `nbs/post-processing.ipynb` (paper aggregation), plus the other 22 notebooks left in `nbs/` (clearing outputs / moving to a `scratch/` subdir is a separate pass).
- Reproducibility: `rerun_paper_aggregation.py` at the repo root.
