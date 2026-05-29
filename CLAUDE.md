# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Multimodal neural network for binary drug-response prediction in PDX (patient-derived xenograft) tumors. Each sample combines three modalities: WSI histology tiles, RNA-seq (LINCS1000, 976 genes), and Mordred drug descriptors for one or two drugs (drug pairs). The pipeline goes raw data → tidy merged dataframe → per-sample TFRecords → multi-tower Keras model.

## Environment

The project runs under a conda env `pdx` (Python 3.7, cudatoolkit 10.1, cudnn 7.6, openslide 3.4.1) that wraps a virtualenv. Two parallel venvs exist for different TF versions:

```bash
./conda_create.sh           # create conda env once (lambda machines; use conda_create_on_lamina.sh on lamina)
conda activate pdx
make dev-venv_tf24          # or dev-venv_tf23 -- creates venv_tf24/ and installs reqs
source venv_tf24/bin/activate
```

`requirements_tf24.txt` pins `tensorflow-gpu==2.4.1`, `requirements_tf23.txt` pins 2.3. Code asserts `tf.__version__ >= "2.0"`.

## End-to-end pipeline

The README describes three sequential stages — each writes artifacts that the next reads.

**1. Build tidy dataframe** (merges all modalities into one CSV per "dataset name"):

```bash
python scripts/build_tidy_drug_pairs_all_samples.py     # drug-pairs (the active variant)
# also: build_tidy_df_all_samples.py, build_tidy_df_partially_balanced.py, build_tidy_single_drug_all_samples.py
```

Each script writes `data/processed/<DATASET_NAME>/annotations.csv` and `annotations_slideflow.csv`. Pick one DATASET_NAME and reuse it via `--dataname` downstream.

**2. Generate TFRecords from the original WSI TFRecords:**

```bash
python scripts/update_tfrecords.py --random --take_glob_min    # writes data/PDX_FIXED_RSP_DRUG_PAIR_rnd_glob_min
python scripts/update_tfrecords.py --frac_tiles 0.1             # writes data/PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles
```

Source TFRecords live in `data/PDX_FIXED/299px_302um/`. The output directory name encodes the subsampling choice (frac_tiles / take_glob_min / single_drug) and is what `--tfr_dir_name` selects at train time.

**3. Train the multimodal model** (one bash script per modality combination, looping over split ids):

```bash
./scripts/tile_ge_dd1_dd2.bash 0          # all four modalities; arg is CUDA device id
./scripts/ge_dd1_dd2.bash 0               # baseline, no tiles
./scripts/ge_dd1_dd2_drop_aug.bash 0      # baseline, drop drug-pair augmentation rows
./scripts/ge_dd1_dd2_only_pairs.bash 0    # baseline, drop single-drug rows
./scripts/baseline.bash 0                 # trn_baseline.py path (no TFRecords)
./scripts/type_classifier.bash 0 <split_id>   # tile → ctype classifier via trn_ctype_cls.py
```

All bash scripts ultimately call `src/trn_multimodal.py` (or `trn_ctype_cls.py` / `trn_baseline.py`) with `--target Response --split_on Group --split_id N --use_tile --use_ge --use_dd1 --use_dd2`. Edit the bash scripts' top-of-file variables (`dataname`, `prjname`, `tfr_dir_name`, `split_start`/`split_end`) rather than passing more CLI flags.

To re-run a single split manually:

```bash
CUDA_VISIBLE_DEVICES=0 python src/trn_multimodal.py \
    --train --eval --target Response --split_on Group --split_id 0 \
    --prjname bin_rsp_drug_pairs_all_samples \
    --dataname tidy_drug_pairs_all_samples \
    --tfr_dir_name PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles \
    --pred_tfr_dir_name PDX_FIXED_RSP_DRUG_PAIR \
    --use_tile --use_ge --use_dd1 --use_dd2
```

To re-evaluate a saved run, pass `--rundir projects/<prjname>/split_<id>_<fea>_<ts>` and omit `--train`.

## Hyperparameters

There is no CLI for HPs. `trn_multimodal.py` chooses a JSON file from `default_params/` based on which `--use_*` flags are active (e.g. all four modalities → `default_params_tile_ge_dd1_dd2.json`). On first run for a given `prjname`+feature-combo, that file is copied to `projects/<prjname>/params_<feacombo>.json` and reused for subsequent splits. To change HPs for a project, edit the per-project copy, not the default. `src/utils/utils.py:Params` is the loader.

## Outputs

Each training run creates `projects/<prjname>/split_<split_id>_<feacombo>[_drop_singles][_drop_aug]_<YYYY-MM-DD_hHH-mMM>/` (see `create_outdir_2` in `src/utils/utils.py`). `cfg.MAIN_PRJDIR` resolves to `<repo>/projects`. Both `projects/` and `data/` are gitignored.

## Architecture

**Single source of truth for paths and constants:** `src/config.py` builds a `types.SimpleNamespace` named `cfg` containing every data path, file name, tile/image/feature size, and seed (42). All modules `from src.config import cfg`. Update paths only there.

**Import convention:** every script does `sys.path.append(str(fdir/".."))` and then `from src.config import cfg` etc., so `src` is treated as a top-level package. When adding new entry points, follow the same pattern — do not import sibling modules as bare names.

**Data flow modules:**
- `src/load_data.py` — per-modality loaders (`load_rsp`, `load_rna`, `load_dd`, `load_crossref`, `load_pdx_meta2`) plus `load_tidy_dataset_rsp()` which performs the master merge. `PDX_SAMPLE_COLS = ['model', 'patient_id', 'specimen_id', 'sample_id']` is the join key parsed out of the `Sample` column (`patient~specimen~sample`).
- `src/datasets/tidy.py:TidyData` — given a merged dataframe + `{tr_id, vl_id, te_id}`, splits, fits scalers on train, transforms val/test. Column-prefix discovery: `ge_*`, `dd1_*`, `dd2_*`. `split_data_and_extract_fea()` is the lower-level functional equivalent.
- `src/tfrecords.py` — `FEA_SPEC_*` `tf.io.FixedLenFeature` dicts. `FEA_SPEC_RSP_DRUG_PAIR` is the current parser for training. RNA/DD payloads are serialized as `tf.string` blobs and reshaped during parsing.
- `src/sf_utils.py` — derived from slideflow: TFRecord parsers (`parse_tfrec_fn_rsp`, `parse_tfrec_fn_rna`), `create_tf_data`, `create_manifest`, `calc_class_weights`, image preprocessing keyed by `base_image_model` via `preprocess_img_input`. Also exports the color helpers (`green`, `red`, ...) used in logs.

**Model module:** `src/models.py`
- `ModelDict` maps `base_image_model` strings ("Xception", "ResNet50", "EfficientNetB1", ...) to their `tf.keras.applications` classes.
- `Multimodal` class owns the model and a manual `train_step` / `evaluate` loop that supports `validate_on_batch` (mid-epoch validation) and a `batch_patience` early-stop independent of epoch boundaries — distinct from the Keras `EarlyStopping` callback used elsewhere.
- `build_model_rsp(...)` (also exposed as a `Multimodal` method) assembles four optional towers: tile (frozen ImageNet backbone → Dense → BN), ge (Dense → BN), dd1 (Dense → BN), dd2 (Dense → BN). `Concatenate` → dense top → 1-logit head. The base image backbone is always `trainable=False` and runs with `training=False`.
- `MySparseBCE_From_Logits` implements weighted binary cross-entropy from logits for class imbalance.

**Splits:** `--split_on Group` (treatment group) is the hard-split unit; `--split_id` is an integer 0–99 indexing precomputed splits. `src/cv_splits.py` and `src/datasplit/splitter.py` generate them.

**Training entry points (mostly parallel evolutions, not generalizations):**
- `src/trn_multimodal.py` — primary; the only one all the active bash scripts call.
- `src/trn_baseline.py` — non-TFRecord baseline using only ge + dd dataframes.
- `src/trn_ctype_cls.py` / `trn_ctype_cls_gen.py` — tile → cancer-type classifier.
- `src/trn_tape.py` — TAPE variant.

These share helpers but duplicate the argparse/setup boilerplate. When changing CLI flags, check whether other entry points need the same change.

## Gotchas

- `image_id` and `slide` must be loaded as `str`. `trn_multimodal.py` passes `dtype = {"image_id": str, "slide": str}` to `pd.read_csv`; preserve that in new readers.
- `cfg.BAD_SLIDES` (poor quality / stain) is filtered out by the loaders — don't reintroduce them.
- Seeds are set in every entry point (`np.random.seed(cfg.seed); tf.random.set_seed(cfg.seed)`) but exact TF reproducibility is not guaranteed.
- No test runner is configured (`pytest` is not installed and there are no `def test_*` functions in the live tree). Earlier scratch reproducibility scripts have been moved to `archive/tests/`.
- `tox.ini` only configures flake8/pycodestyle ignores (`E402,E501,E712,W503`); there is no `tox` test environment.
- `data/`, `projects/`, `apps/`, and `nbs/histolab` are gitignored. Anything new under those paths won't be tracked.
