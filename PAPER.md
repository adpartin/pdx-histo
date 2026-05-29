# Paper → Code Map

> Partin A, Brettin T, Zhu Y, Dolezal JM, Kochanny S, Pearson AT, Shukla M, Evrard YA, Doroshow JH, Stevens RL.
> **Data augmentation and multimodal learning for predicting drug response in patient-derived xenografts from gene expressions and histology images.**
> *Frontiers in Medicine* 10 (2023). doi:10.3389/fmed.2023.1058919

This document maps each row of the paper's results table to the bash script that launched it, the project directory where the 100 split outputs live, and the hyperparameter file that was actually used. Trained outputs are under `projects/` (gitignored — present on this workstation, not in the GitHub mirror).

---

## Headline results (paper table) → artifacts

| Paper model | MCC / AUPRC / AUROC | Bash entry point | Project subdir (100 splits) | Run dates | HPs |
|---|---|---|---|---|---|
| **MM-Net** (tile + GE + DD1 + DD2) | 0.3102 / 0.2974 / 0.7978 | `scripts/tile_ge_dd1_dd2.bash` | `projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd/split_<0..99>_tile_ge_dd1_dd2_2021-05-11_*` | **May 11, 2021** (paper sweep — verified by re-aggregation) | `projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd/split_*/params.json` (legacy HPs, not `params_tile_ge_dd1_dd2.json` at the project root) |
| **UME-Net** (GE + DD1 + DD2; no tiles) | 0.2958 / 0.2996 / 0.8047 | `scripts/ge_dd1_dd2.bash` | `projects/bin_rsp_drug_pairs_all_samples/runs_ge_dd/split_*_ge_dd1_dd2_2021-05-21_*` | May 21, 2021 | `projects/bin_rsp_drug_pairs_all_samples/params_ge_dd1_dd2.json` |
| **UMH-Net** (tile + DD1 + DD2; no GE) | 0.2124 / 0.2303 / 0.7977 | `scripts/tile_dd1_dd2.bash` | `projects/bin_rsp_drug_pairs_all_samples/runs_tile_dd/split_*_tile_dd1_dd2_2021-07-16/25_*` | Jul 16–25, 2021 | `projects/bin_rsp_drug_pairs_all_samples/params_tile_dd1_dd2.json` |
| **UME-Net_org** (no augmented drug-pairs) | 0.2391 / 0.2610 / 0.7766 | `scripts/ge_dd1_dd2_drop_aug.bash` (passes `--drop_drug_pair_aug`) | `projects/bin_rsp_drug_pairs_drop_aug/split_*_ge_dd1_dd2_drop_aug_*` | Sep 2021 onwards | `projects/bin_rsp_drug_pairs_drop_aug/params_ge_dd1_dd2.json` |
| **UME-Net_pairs** (drug-pairs only, no single-drug) | 0.2039 / 0.2355 / 0.7423 | `scripts/ge_dd1_dd2_only_pairs.bash` (passes `--drop_single_drug`; `prjname=bin_rsp_drug_pairs_only_pairs`) | **MISSING** — `projects/bin_rsp_drug_pairs_only_pairs/` is not present on this workstation | N/A | template at `default_params/default_params_ge_dd1_dd2.json` |
| **LGBM** (GE + DD1 + DD2 baseline) | 0.2594 / 0.2784 / 0.8065 | **Not trained in this repo.** LGBM predictions were produced externally and dropped in as a pre-computed results bundle at `data/PDX_Transfer_Learning_Classification/Results_MultiModal_Learning/1.0_True_False_100_31/cv_<0..99>/` (102 dirs + `AllData.pkl`). Each `cv_<i>/` contains `te_scores.csv`, `test_preds.csv`, plus a `Model/` checkpoint. The repo only **consumes** these scores in the aggregation notebook (see below). | N/A — externally trained | N/A — folder name encodes the upstream HPs (`1.0_True_False_100_31`) |

The earlier MM-Net sweep at `projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd/` (May 11, 2021) was **superseded** by the Oct 7–8 final sweep after commit `41e157c 2021-10-05 "random tiles with global min"`. Don't use it for paper numbers.

---

## Confirmed paper run configuration

From `projects/bin_rsp_drug_pairs_all_samples/split_81_tile_ge_dd1_dd2_2021-10-07_h19-m19/logger.log` (Oct 2021 sweep — config is the same dataset/split scheme as the May 2021 paper sweep; verify against an individual May run's logger.log for the exact paper HPs):

```
dataname            = tidy_drug_pairs_all_samples
prjname             = bin_rsp_drug_pairs_all_samples
target              = Response          (binary)
split_on            = Group             (leakage-safe split unit)
n_samples           = -1                (no subsampling)
tfr_dir_name        = PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles    (training)
pred_tfr_dir_name   = PDX_FIXED_RSP_DRUG_PAIR                 (inference)
use_tile/ge/dd1/dd2 = True / True / True / True
drop_single_drug    = False
drop_drug_pair_aug  = False
final dataframe     = (6962, 4954)      ← matches paper's 6,962 samples
```

Subtle: **UMH-Net used `PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles` at both training and inference** (per `runs_tile_dd/split_0/logger.log`), while MM-Net used the full `PDX_FIXED_RSP_DRUG_PAIR` at inference. Worth knowing if reproducing baseline numbers exactly.

---

## Hyperparameters actually used (vs. defaults)

The repo ships templates at `default_params/`; on first run for a given `prjname` + feature combo, `trn_multimodal.py:158-161` copies the template into `projects/<prjname>/params_<feacombo>.json` and **the project-level copy is the source of truth from then on**. The paper's numbers were produced from per-run `params.json` files inside each `runs_*/split_*/` directory — NOT the project-level `params_*.json` at the project root (those were overwritten by the later Oct 2021 re-run and reflect post-paper HPs).

Per-split params.json from `runs_tile_ge_dd/split_0_tile_ge_dd1_dd2_2021-05-11_h16-m02/params.json` (paper MM-Net):

| HP | **MM-Net PAPER** (May 2021) | MM-Net post-paper (Oct 2021, do not use) | `default_params/...tile_ge_dd1_dd2.json` template |
|---|---:|---:|---:|
| `epochs` | **100** | 70 | 50 |
| `patience` | **10** | 11 | 10 |
| `batch_patience` | **50** | 5 | 50 |
| `validate_on_batch` | **null** | 250 | null |
| `from_logits` | (absent → False) | True | True |
| `class_weights_method` | BY_TILE | BY_TILE | BY_TILE |
| `base_image_model` | Xception | Xception | Xception |
| `learning_rate` | 5e-4 | 5e-4 | 5e-4 |
| `label_smoothing` | 0.05 | 0.05 | 0.05 |

**Important correction:** the paper MM-Net used `validate_on_batch=null`, which means **vanilla Keras `model.fit`** — *not* the custom mid-epoch training loop in `src/models.py:Multimodal.train_step`. The `Multimodal` class was introduced in commit `99865dd 2021-08-06 "custom multimodal class"`, three months *after* the May paper sweep, and is post-paper infrastructure. The May runs used a simpler `build_model_rsp` + standard Keras fit path. Earlier draft of this document overstated the role of the custom loop.

The same caveat applies to other baselines — their per-run `params.json` (not the project-root copy) reflects what was actually used:
- UME-Net: `epochs=100, patience=30, validate_on_batch=null` — vanilla Keras.
- UMH-Net: per-run params at `runs_tile_dd/split_*/params.json` — verify against that, not the project-level `params_tile_dd1_dd2.json`.

---

## Dataset construction (one-time prep, before any training)

```
scripts/build_tidy_drug_pairs_all_samples.py        # merges rsp + rna + dd + cref + pdx_meta
        ↓
data/processed/tidy_drug_pairs_all_samples/
    annotations.csv
    annotations_slideflow.csv                       # adds submitter_id / slide cols for sf_utils
        ↓
scripts/update_tfrecords.py --frac_tiles 0.1        # uses src/tfrecords.py:FEA_SPEC_RSP_DRUG_PAIR
        ↓
data/PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles/299px_302um/*.tfrec
```

The dataset shape `(6962, 4954)` reported in the paper is set by `build_tidy_drug_pairs_all_samples.py:37`'s call to `load_data.load_tidy_dataset_rsp(single_drug=False, add_type_labels=True)`. Augmentation logic (drug-pair position swap, single-drug → pseudo-pair homogenization) lives inside `src/load_data.py:load_tidy_dataset_rsp`.

---

## Model code (load-bearing for the paper)

| Concern | File / symbol |
|---|---|
| Argument parsing, training/eval orchestration | `src/trn_multimodal.py:parse_args`, `src/trn_multimodal.py:run` (CAVEAT: current `trn_multimodal.py` is post-paper; the May 2021 paper run predates the `Multimodal` class. Use `git log -- src/trn_multimodal.py` and check the state at commit `45385a0 2021-05-27` or earlier to see the paper-era trainer.) |
| Four-tower architecture builder | `src/models.py:build_model_rsp` |
| ~~Custom mid-batch training loop & early stop~~ | `src/models.py:Multimodal.train_step`, `Multimodal.fit` — **POST-PAPER**, added in commit `99865dd 2021-08-06`. Paper used vanilla Keras `model.fit`. |
| Weighted binary cross-entropy from logits | `src/models.py:MySparseBCE_From_Logits` (verify when this was added vs paper sweep date) |
| TFRecord parsing | `src/sf_utils.py:parse_tfrec_fn_rsp`, `src/tfrecords.py:FEA_SPEC_RSP_DRUG_PAIR` |
| tf.data pipeline assembly | `src/sf_utils.py:create_tf_data`, `create_manifest`, `calc_class_weights` |
| Image preprocessing (per CNN backbone) | `src/sf_utils.py:preprocess_img_input` |
| Modality preprocessing & split-aware scaling | `src/datasets/tidy.py:TidyData`, `split_data_and_extract_fea` |
| Hyperparameter loader | `src/utils/utils.py:Params` |
| Output directory naming convention `split_<id>_<feacombo>_<ts>/` | `src/utils/utils.py:create_outdir_2` |
| Path/constant single source of truth | `src/config.py:cfg` |

Everything else under `src/` (trn_baseline, trn_ctype_cls*, trn_tape, batch_*, eda, post_processing, cv_splits, proj/, old/) is **not** part of the MM-Net paper pipeline — see `AUDIT.md` for details.

---

## Aggregation: per-split outputs → paper tables

Each of the 100 split directories contains (sample, from `runs_tile_dd/split_0_*/`):

```
best_model.ckpt              # best-weights checkpoint
final_model.ckpt             # weights at end of training
params.json                  # actual HPs (== project-level params copy)
logger.log                   # full stdout including HPs, dataset shape, per-epoch metrics
training.log                 # Keras CSVLogger output
test_scores.csv              # the row that goes into the paper aggregation
test_tile_preds.csv          # per-tile predictions
test_smp_preds.csv           # per-sample (slide × treatment) predictions
test_grp_preds.csv           # per-group predictions
test_{tile,smp,grp}_confusion.png
{tr,vl,te}_meta.csv          # split membership
```

The aggregation step that produces the paper's MCC/AUPRC/AUROC table from these 100 `test_scores.csv` files runs in `nbs/post-processing.ipynb`. The load-bearing cells:

- **Cell 41** loads all four models' per-split scores via `src.post_processing.agg_scores_from_splits(...)` — `mm-NN` from `runs_tile_ge_dd/`, `umh-NN` from `runs_tile_dd/`, `ume-NN` from `runs_ge_dd/` (filename `test_keras_scores.csv`, not `test_scores.csv` — UME-Net used vanilla Keras), and `ume-LGBM` from the external `data/PDX_Transfer_Learning_Classification/Results_MultiModal_Learning/1.0_True_False_100_31/cv_*/te_scores.csv`.
- **Cells 46–48** compute pairwise comparisons (`mm-NN` vs each baseline) and run paired `scipy.stats.ttest_rel` for significance.
- **Cells 50–53** produce per-split ROC-curve plots and calibration curves.
- **Cells 71–77** are the secondary aggregation used for paper figures (per-sample / per-tile / per-group predictions).

The aggregation library is `src/post_processing.py`, providing `agg_scores_from_splits`, `scores_boxplot`, `scores_barplot`, `t_test`, `t_test_all_metrics`. It is imported only from this notebook — that's why `grep` over `*.py` and `*.bash` missed it. Mark `src/post_processing.py` as paper-relevant (do not delete; see `AUDIT.md` correction).

`nbs/04-trn_nn.ipynb` is **not** related — it's an earlier exploratory notebook using a different tf.data + NN setup and references `../apps/` (gitignored, predates the current pipeline). No LGBM in it.

---

## Known gaps in the on-disk record

1. ~~Which MM-Net sweep did the paper actually use?~~ **RESOLVED — May 2021.** Re-running cells 41/46/48 (via `rerun_paper_aggregation.py`) against the Oct 2021 sweep produced MM-Net MCC=0.2226 / AUPRC=0.2482 / AUROC=0.7488 at Group level — substantially **worse** than the paper. Running against the May 2021 sweep (`runs_tile_ge_dd/`) reproduces the paper table to four decimal places at Group-level mean:

   | Model | Paper (MCC / AUPRC / AUROC) | May sweep, Group-mean | Match |
   |---|---|---|---|
   | MM-Net  | 0.3102 / 0.2974 / 0.7978 | 0.3102 / 0.2974 / 0.7978 | ✓ exact |
   | UME-Net | 0.2958 / 0.2996 / 0.8047 | 0.2958 / 0.2996 / 0.8047 | ✓ exact |
   | UMH-Net | 0.2124 / 0.2303 / 0.7977 | 0.2124 / 0.2303 / 0.7977 | ✓ exact |
   | LGBM    | 0.2594 / 0.2784 / 0.8065 | 0.2594 / 0.2784 / 0.8065 | ✓ exact |

   So the paper used:
   - **MM-Net runs**: `projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd/split_*_2021-05-11_*` (the May sweep, not the Oct one).
   - **Aggregation**: **Group-level mean** across 100 splits (not smp-level, not median).
   - **Paper's "46/100" claim**: matches `mm-NN > ume-NN on AUPRC: 46/100` at Group level exactly.

   The Oct 2021 top-level sweep is **post-paper exploration** — possibly a "random tiles with global min" follow-up that did not improve over the May result and was never used in the publication. The earlier confusion (one-split MCC comparison favoring Oct) was due to a single-split outlier; in aggregate, the May sweep wins by ~5 MCC points absolute. **Treat the top-level Oct sweep as non-paper.**

2. **`projects/bin_rsp_drug_pairs_only_pairs/` is absent**, but `scripts/ge_dd1_dd2_only_pairs.bash` sets that project name. The UME-Net_pairs row (MCC 0.2039) in the paper cannot be re-derived from this workstation; the runs were either deleted, archived elsewhere, or stored under a different prjname.

3. **`projects/bin_rsp_drug_pairs_drop_aug/` has only `split_0`**, but the bash script loops 0–99. Either an incomplete sweep is checked in here, or splits 1–99 were stored elsewhere. The UME-Net_org paper row (MCC 0.2391) depends on a complete 100-split sweep.

4. **Why `_0.1_of_tiles` and not `_rnd_glob_min`** for the final MM-Net sweep. Commit `41e157c 2021-10-05 "random tiles with global min"` introduced `--random --take_glob_min` flags to `update_tfrecords.py`, and the `runs_tile_ge_dd_rnd_glob_min/` placeholder dir exists but is empty. The Oct 7–8 sweep used `_0.1_of_tiles`, which suggests that directory was *regenerated* with the rnd-glob-min logic between Oct 5 and Oct 7 (rather than a new dir being created). Cannot verify without the TFRecord file timestamps.

---

## tl;dr

The minimum reading list for the paper artifact:

1. **The pipeline** — `scripts/build_tidy_drug_pairs_all_samples.py` → `scripts/update_tfrecords.py` → `scripts/tile_ge_dd1_dd2.bash` → `src/trn_multimodal.py` → `src/models.py:build_model_rsp`.
2. **The HPs** — `projects/bin_rsp_drug_pairs_all_samples/params_tile_ge_dd1_dd2.json` (not the `default_params/` template).
3. **The runs** — `projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd/split_<0..99>_tile_ge_dd1_dd2_2021-05-11_*/` (verified by re-aggregation). The top-level `split_*_2021-10-*` directories are a post-paper re-run that performed worse and were not used.
4. **The aggregation** — `nbs/post-processing.ipynb` cells 41/46/48, which call `src/post_processing.py:agg_scores_from_splits` over per-split `test_scores.csv` files. Use **Group-level mean** across 100 splits to match the paper table. Run `rerun_paper_aggregation.py` (untracked, repo root) to reproduce.
5. **The LGBM baseline** — externally produced, sitting in `data/PDX_Transfer_Learning_Classification/Results_MultiModal_Learning/1.0_True_False_100_31/cv_<0..99>/te_scores.csv`. Not trained from this repo.

Everything else is either supporting infrastructure (`load_data.py`, `sf_utils.py`, `tfrecords.py`, `datasets/tidy.py`, `utils/utils.py`, `config.py`) or non-paper code that should not factor into a code-review of the paper.
