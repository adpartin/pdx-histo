"""
Prediction of drug response with TFRecords.
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path
from pprint import pprint
import json
import csv
import glob
import shutil
from time import time
from functools import partial
from typing import List, Optional, Union

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
assert tf.__version__ >= "2.0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow import keras
from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
# from tensorflow.keras import layers
# from tensorflow.keras import losses
# from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.utils import plot_model

from tf_utils import get_tfr_files
from sf_utils import (green, interleave_tfrecords,
                      parse_tfrec_fn_rsp, parse_tfrec_fn_rna,
                      create_manifest)
from models import build_model_rsp, build_model_rna, build_model_rsp_baseline
from ml.scale import get_scaler
from ml.evals import calc_scores, calc_preds, dump_preds
from utils.utils import Params, dump_dict
from datasets.tidy import split_data_and_extract_fea

fdir = Path(__file__).resolve().parent
from config import cfg

# Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


parser = argparse.ArgumentParser("Train NN.")
parser.add_argument('-t', '--target',
                    type=str,
                    nargs='+',
                    # default=['ctype'],
                    default=['Response'],
                    choices=['Response', 'ctype', 'csite'],
                    help='Name of target output.')
parser.add_argument('--id_name',
                    type=str,
                    # default='slide',
                    default='smp',
                    choices=['slide', 'smp'],
                    help='Column name of the ID.')
parser.add_argument('--split_on',
                    type=str,
                    default=None,
                    choices=['Sample', 'slide'],
                    help='Specify the hard split variable/column (default: None).')
parser.add_argument('--prjname',
                    type=str,
                    help='Project name (folder that contains the annotations.csv dataframe).')
parser.add_argument('--dataname',
                    type=str,
                    default='tidy_all',
                    help='Project name (folder that contains the annotations.csv dataframe).')

args, other_args = parser.parse_known_args()
pprint(args)

import ipdb; ipdb.set_trace()

# use_ge, use_dd = True, True
# use_tile = True

# Load dataframe (annotations)
prjdir = cfg.MAIN_PRJDIR/args.prjname
# annotations_file = prjdir/cfg.SF_ANNOTATIONS_FILENAME
annotations_file = cfg.DATA_PROCESSED_DIR/args.dataname/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

# Outdir
split_on = 'none' if args.split_on is (None or 'none') else args.split_on
outdir = prjdir/f"multimodal/split_on_{split_on}"
os.makedirs(outdir, exist_ok=True)

# Import parameters
# import ipdb; ipdb.set_trace()
prm_file_path = prjdir/"multimodal/params.json"
if prm_file_path.exists() is False:
    shutil.copy(fdir/"default_params.json", prm_file_path)
params = Params(prm_file_path)

if args.target[0] == 'Response':
    pprint(data.groupby(['ctype', 'Response']).agg({
        'index': 'nunique'}).reset_index().rename(columns={'index': 'count'}))
else:
    pprint(data[args.target[0]].value_counts())

ge_cols = [c for c in data.columns if c.startswith('ge_')]
dd_cols = [c for c in data.columns if c.startswith('dd_')]
data = data.astype({'image_id': str, 'slide': str})

# RNA scaler
if params.use_ge:
    ge_scaler = get_scaler(data[ge_cols])
else:
    ge_scaler = None

# Descriptors scaler
if params.use_dd:
    dd_scaler = get_scaler(data[dd_cols])
else:
    dd_scaler = None


BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'

# tile_px = 299
# tile_um = 302
# finetune_epochs = 1
toplayer_epochs = 0
# model = 'Xception'
# pooling = 'max'

loss = 'sparse_categorical_crossentropy'
# loss={'csite_label': tf.keras.losses.categorical_crossentropy,
#     'ctype_label': tf.keras.losses.categorical_crossentropy},
# loss = {'ctype': tf.keras.losses.SparseCategoricalCrossentropy()}

# learning_rate = 0.0001
# batch_size = 16
# hidden_layers = 1
# hidden_layer_width = 500
# optimizer = 'Adam'
early_stop = True 
# early_stop_patience = 10
early_stop_method = 'loss'  # uses only validation metrics
# balanced_training = 'BALANCE_BY_CATEGORY'
# balanced_validation = 'NO_BALANCE'
# trainable_layers = 0
# augment = True

# outcome_header = ['Response']
outcome_header = args.target
# aux_headers = ['ctype', 'csite']  # (ap)
# model_type = 'categorical'
# use_fp16 = True
# pretrain = 'imagenet'

label = f'{params.tile_px}px_{params.tile_um}um'


if args.target[0] == 'Response':
    tfr_dir = cfg.SF_TFR_DIR_RSP
    parse_fn = parse_tfrec_fn_rsp
elif args.target[0] == 'ctype':
    tfr_dir = cfg.SF_TFR_DIR_RNA_NEW
    parse_fn = parse_tfrec_fn_rna
tfr_dir = tfr_dir/label


# (ap) Create outcomes (for drug response)
# __init__ --> _trainer --> training_dataset.get_outcomes_from_annotations
outcomes = {}
unique_outcomes = list(set(data[args.target[0]].values))
unique_outcomes.sort()

for smp, o in zip(data[args.id_name], data[args.target[0]]):
    outcomes[smp] = {'outcome': unique_outcomes.index(o)}
    # outcomes[smp] = {'outcome': unique_outcomes.index(o), 'submitter_id': smp}

print("\n'outcomes':")
print(type(outcomes))
print(len(outcomes))
print(list(outcomes.keys())[:3])
print(outcomes[list(outcomes.keys())[3]])


# ---------------------
# Create outcome_labels - done
# ---------------------
# __init__ --> _trainer
outcome_labels = dict(zip(range(len(unique_outcomes)), unique_outcomes))
print(outcome_labels)

# ---------------
# Create manifest - done
# ---------------
manifest = create_manifest(directory=tfr_dir)
print('\nmanifest:')
print(type(manifest))
print(len(manifest))
print(list(manifest.keys())[:3])
print(manifest[list(manifest.keys())[3]])

# --------------------------------------------------
# Create training_tfrecords and validation_tfrecords - done
# --------------------------------------------------
# __init__ --> _trainer --> sfio.tfrecords.get_training_and_validation_tfrecords
# Note: I use my processing to split data

# import ipdb; ipdb.set_trace()

# from sklearn.model_selection import train_test_split
# # slides = list(outcomes.keys())
# # y = [outcomes[k]['outcome'] for k in outcomes.keys()]
# samples = list(outcomes.keys())
# y = [outcomes[k]['outcome'] for k in outcomes.keys()]
# s_tr, s_te, y_tr, y_te = train_test_split(samples, y, test_size=0.2,
#                                           stratify=y, random_state=seed)

# train_tfrecords = [str(directory/s)+'.tfrecords' for s in s_tr]
# val_tfrecords = [str(directory/s)+'.tfrecords' for s in s_te]

# data = data.astype({args.id_name: str})
# dtr = data[data[args.id_name].isin(s_tr)]
# dte = data[data[args.id_name].isin(s_te)]

# assert sorted(s_tr) == sorted(dtr[args.id_name].values.tolist()), "Sample names \
#     in the s_tr and dtr don't match."
# assert sorted(s_te) == sorted(dte[args.id_name].values.tolist()), "Sample names \
#     in the s_te and dte don't match."

# print(f'Training files {len(train_tfrecords)}')
# print(f'Validation files {len(val_tfrecords)}')



# -----------------------------------------------
# Data splits
# -----------------------------------------------

# import ipdb; ipdb.set_trace()

# T/V/E filenames
# splitdir = prjdir/f'annotations.splits/split_on_{split_on}'
splitdir = cfg.DATA_PROCESSED_DIR/args.dataname/f'annotations.splits/split_on_{split_on}'
split_id = 0

split_pattern = f'1fold_s{split_id}_*_id.txt'
single_split_files = glob.glob(str(splitdir/split_pattern))
# single_split_files = list(splitdir.glob(split_pattern))

def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines

def cast_list(ll, dtype=int):
    return [dtype(i) for i in ll]

# Get indices for the split
assert len(single_split_files) >= 2, f'Split {split_id} contains only one file.'
for id_file in single_split_files:
    if 'tr_id' in id_file:
        tr_id = cast_list(read_lines(id_file), int)
    elif 'vl_id' in id_file:
        vl_id = cast_list(read_lines(id_file), int)
    elif 'te_id' in id_file:
        te_id = cast_list(read_lines(id_file), int)

kwargs = {"ge_cols": ge_cols,
          "dd_cols": dd_cols,
          "ge_scaler": ge_scaler,
          "dd_scaler": dd_scaler,
          "ge_dtype": cfg.GE_DTYPE,
          "dd_dtype": cfg.DD_DTYPE
}
tr_ge, tr_dd = split_data_and_extract_fea(data, ids=tr_id, **kwargs)
vl_ge, vl_dd = split_data_and_extract_fea(data, ids=vl_id, **kwargs)
te_ge, te_dd = split_data_and_extract_fea(data, ids=te_id, **kwargs)

ge_shape = (te_ge.shape[1],)
dd_shape = (tr_dd.shape[1],)

# Variables (dict/dataframes/arrays) that are passed as features to the NN
xtr = {"ge_data": tr_ge.values, "dd_data": tr_dd.values}
xvl = {"ge_data": vl_ge.values, "dd_data": vl_dd.values}
xte = {"ge_data": te_ge.values, "dd_data": te_dd.values}

# Extarct meta for T/V/E
tr_meta = data.iloc[tr_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
vl_meta = data.iloc[vl_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
te_meta = data.iloc[te_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)

# List of sample names for T/V/E
tr_smp_names = list(tr_meta[args.id_name].values)
vl_smp_names = list(vl_meta[args.id_name].values)
te_smp_names = list(te_meta[args.id_name].values)

# TFRecords filenames
tr_tfr_files = get_tfr_files(tfr_dir, tr_smp_names)  # training_tfrecords
vl_tfr_files = get_tfr_files(tfr_dir, vl_smp_names)  # validation_tfrecords
te_tfr_files = get_tfr_files(tfr_dir, te_smp_names)
print('Total samples {}'.format(len(tr_tfr_files) + len(vl_tfr_files) + len(te_tfr_files)))

# # Dataframes of T/V/E samples
# tr_df = data.iloc[tr_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
# vl_df = data.iloc[vl_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
# te_df = data.iloc[te_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
# print('Total samples {}'.format(tr_df.shape[0] + vl_df.shape[0] + te_df.shape[0]))

# # List of sample names for T/V/E
# tr_smp_names = list(tr_df[args.id_name].values)
# vl_smp_names = list(vl_df[args.id_name].values)
# te_smp_names = list(te_df[args.id_name].values)

# # TFRecords filenames
# tr_tfr_files = get_tfr_files(tfr_dir, tr_smp_names)  # training_tfrecords
# vl_tfr_files = get_tfr_files(tfr_dir, vl_smp_names)  # validation_tfrecords
# te_tfr_files = get_tfr_files(tfr_dir, te_smp_names)
# print('Total samples {}'.format(len(tr_tfr_files) + len(vl_tfr_files) + len(te_tfr_files)))

# Missing tfrecords
print('\nThese samples miss a tfrecord ...\n')
print(data.loc[~data[args.id_name].isin(tr_smp_names + vl_smp_names + te_smp_names), ['smp', 'image_id']])

train_tfrecords = tr_tfr_files
val_tfrecords = vl_tfr_files
test_tfrecords = te_tfr_files

# import ipdb; ipdb.set_trace()

# tr_ge_df, tr_dd_df = tr_df[ge_cols], tr_df[dd_cols]
# vl_ge_df, vl_dd_df = vl_df[ge_cols], vl_df[dd_cols]
# te_ge_df, te_dd_df = te_df[ge_cols], te_df[dd_cols]

# tr_dd_df = pd.DataFrame(dd_scaler.transform(tr_dd_df), columns=dd_cols, dtype=cfg.DD_DTYPE)
# vl_dd_df = pd.DataFrame(dd_scaler.transform(vl_dd_df), columns=dd_cols, dtype=cfg.DD_DTYPE)
# te_dd_df = pd.DataFrame(dd_scaler.transform(te_dd_df), columns=dd_cols, dtype=cfg.DD_DTYPE)

# tr_ge_df = pd.DataFrame(ge_scaler.transform(tr_ge_df), columns=ge_cols, dtype=cfg.GE_DTYPE)
# vl_ge_df = pd.DataFrame(ge_scaler.transform(vl_ge_df), columns=ge_cols, dtype=cfg.GE_DTYPE)
# te_ge_df = pd.DataFrame(ge_scaler.transform(te_ge_df), columns=ge_cols, dtype=cfg.GE_DTYPE)

# tr_y_df = tr_df[args.target]
# vl_y_df = vl_df[args.target]
# te_y_df = te_df[args.target]

ydata = data[args.target[0]].values
ytr = ydata[tr_id]
yvl = ydata[vl_id]
yte = ydata[te_id]

assert sorted(tr_smp_names) == sorted(tr_meta[args.id_name].values.tolist()), "Sample names \
    in the tr_smp_names and tr_meta don't match."
assert sorted(vl_smp_names) == sorted(vl_meta[args.id_name].values.tolist()), "Sample names \
    in the vl_smp_names and vl_meta don't match."
assert sorted(te_smp_names) == sorted(te_meta[args.id_name].values.tolist()), "Sample names \
    in the te_smp_names and te_meta don't match."

# split_outdir = outdir/f'split_{split_id}'
# os.makedirs(split_outdir, exist_ok=True)
# tr_df.to_csv(split_outdir/'dtr.csv', index=False)
# vl_df.to_csv(split_outdir/'dvl.csv', index=False)
# te_df.to_csv(split_outdir/'dte.csv', index=False)

# num_slide_input = 0
# input_labels = None


# import ipdb; ipdb.set_trace()
print(len(outcomes))
print(len(manifest))
print(outcomes[list(outcomes.keys())[3]])
print(manifest[list(manifest.keys())[3]])


# -------------------------------
# create SlideflowModel model SFM
# -------------------------------
# import ipdb; ipdb.set_trace()

#DATA_DIR = '/vol/ml/apartin/projects/slideflow-proj/sf_pdx_bin_rsp2/project/models/ctype-Xception_v0-kfold1';
DATA_DIR = outdir
# os.makedirs(DATA_DIR, exist_ok=True)
MANIFEST = manifest
# IMAGE_SIZE = image_size = params.tile_px
DTYPE = 'float16' if params.use_fp16 else 'float32'
SLIDE_ANNOTATIONS = slide_annotations = outcomes
TRAIN_TFRECORDS = train_tfrecords
VALIDATION_TFRECORDS = val_tfrecords
# model_type = MODEL_TYPE = model_type
#SLIDES = list(slide_annotations.keys())
SAMPLES = list(slide_annotations.keys())
DATASETS = {}
# NUM_SLIDE_INPUT = num_slide_input

#outcomes_ = [slide_annotations[slide]['outcome'] for slide in SLIDES]
outcomes_ = [slide_annotations[smp]['outcome'] for smp in SAMPLES]

# normalizer = None

if params.model_type == 'categorical':
    NUM_CLASSES = len(list(set(outcomes_)))  # infer this from other variables

#ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SLIDES, outcomes_), -1)]
ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SAMPLES, outcomes_), -1)]


# pretrain_model_format = None
# resume_training = None
# checkpoint = None
# log_frequency = 100
# multi_image = False
# validate_on_batch = 16
val_batch_size = 32
validation_steps = 200
max_tiles_per_slide = 0
min_tiles_per_slide = 0
starting_epoch = 0  # what's that??
# ema_observations = 20
# ema_smoothing = 2
steps_per_epoch_override = None  # what's that??

# AUGMENT = True


# import ipdb; ipdb.set_trace()

if args.target[0] == 'Response':
    # Response
    parse_fn = parse_tfrec_fn_rsp
    parse_fn_kwargs = {
        'use_tile': params.use_tile,
        'use_ge': params.use_ge,
        'use_dd': params.use_dd,
        'ge_scaler': ge_scaler,
        'dd_scaler': dd_scaler,
        'id_name': args.id_name,
        'AUGMENT': params.augment,
        'ANNOTATIONS_TABLES': ANNOTATIONS_TABLES
    }
else:
    # Ctype
    parse_fn = parse_tfrec_fn_rna
    parse_fn_kwargs = {
        'use_tile': params.use_tile,
        'use_ge': params.use_ge,
        'ge_scaler': ge_scaler,
        'id_name': args.id_name,
        'MODEL_TYPE': params.model_type,
        'ANNOTATIONS_TABLES': ANNOTATIONS_TABLES,
        'AUGMENT': params.augment,
    }

print("\nTraining TFRecods")
train_data, _, num_tiles = interleave_tfrecords(
    tfrecords=TRAIN_TFRECORDS,
    batch_size=params.batch_size,
    balance=params.balanced_training,
    finite=False,
    max_tiles=max_tiles_per_slide,
    min_tiles=min_tiles_per_slide,
    include_smp_names=False,
    parse_fn=parse_fn,
    MANIFEST=MANIFEST,
    SLIDE_ANNOTATIONS=SLIDE_ANNOTATIONS,
    SAMPLES=SAMPLES,
    **parse_fn_kwargs
)

# Determine feature shapes from data
bb = next(train_data.__iter__())

# import ipdb; ipdb.set_trace()
if params.use_ge:
    ge_shape = bb[0]['ge_data'].numpy().shape[1:]
else:
    ge_shape = None

if params.use_dd:
    dd_shape = bb[0]['dd_data'].numpy().shape[1:]
else:
    dd_shape = None

for i, item in enumerate(bb):
    print(f"\nItem {i}")
    if isinstance(item, dict):
        for k in item.keys():
            print(f"\t{k}: {item[k].numpy().shape}")

for i, rec in enumerate(train_data.take(4)):
    tf.print(rec[1])

using_validation = True

# Test data
if using_validation:
    print("\nValidation TFRecods")
    val_data, val_data_with_smp_names, _ = interleave_tfrecords(
        tfrecords=VALIDATION_TFRECORDS,
        batch_size=val_batch_size,
        balance='NO_BALANCE',
        finite=True,
        max_tiles=max_tiles_per_slide,
        min_tiles=min_tiles_per_slide,
        include_smp_names=True,
        parse_fn=parse_fn,
        MANIFEST=MANIFEST,
        SLIDE_ANNOTATIONS=SLIDE_ANNOTATIONS,
        SAMPLES=SAMPLES,
        **parse_fn_kwargs
    )

    if validation_steps:
        validation_data_for_training = val_data.repeat()
        print(f"Using {validation_steps} batches ({validation_steps * params.batch_size} samples) each validation check")
    else:
        validation_data_for_training = val_data
        print(f"Using entire validation set each validation check")
else:
    log.info("Validation during training: None", 1)
    validation_data_for_training = None
    validation_steps = 0

# Test data
print("\nTest TFRecods")
test_data, test_data_with_smp_names, _ = interleave_tfrecords(
    tfrecords=test_tfrecords,
    batch_size=val_batch_size,
    balance='NO_BALANCE',
    finite=True,
    max_tiles=max_tiles_per_slide,
    min_tiles=min_tiles_per_slide,
    include_smp_names=True,
    parse_fn=parse_fn,
    MANIFEST=MANIFEST,
    SLIDE_ANNOTATIONS=SLIDE_ANNOTATIONS,
    SAMPLES=SAMPLES,
    **parse_fn_kwargs
)

# ----------------------
# Prep for training
# ----------------------
# import ipdb; ipdb.set_trace()

# Prepare results
results = {'epochs': {}}

# Calculate parameters
if max([params.finetune_epochs]) <= starting_epoch:
    print(f"Starting epoch ({starting_epoch}) cannot be greater than the \
          maximum target epoch ({max(params.finetune_epochs)})")
    
if early_stop and early_stop_method == 'accuracy' and params.model_type != 'categorical':
    print(f"Unable to use early stopping method 'accuracy' with a non-categorical \
          model type (type: '{params.model_type}')")
    
if starting_epoch != 0:
    print(f"Starting training at epoch {starting_epoch}")
    
total_epochs = toplayer_epochs + (max([params.finetune_epochs]) - starting_epoch)
# TODO: this might need to change??
steps_per_epoch = round(num_tiles/params.batch_size) if steps_per_epoch_override is None else steps_per_epoch_override
results_log = os.path.join(DATA_DIR, 'results_log.csv')
metrics = ['accuracy'] if params.model_type != 'linear' else [loss]


# Create callbacks for early stopping, checkpoint saving, summaries, and history
history_callback = tf.keras.callbacks.History()
checkpoint_path = os.path.join(DATA_DIR, "cp.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=1)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=DATA_DIR, histogram_freq=0, write_graph=False, update_freq=log_frequency)


class PredictionAndEvaluationCallback(tf.keras.callbacks.Callback):
    pass


#callbacks = [history_callback, PredictionAndEvaluationCallback(), cp_callback, tensorboard_callback]
callbacks = [history_callback, cp_callback]

# https://www.tensorflow.org/guide/mixed_precision
if DTYPE == 'float16':
    # TF 2.4
    # from tensorflow.keras import mixed_precision
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    # TF 2.3
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    print("Training with mixed precision")
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)


# -------------
# Train model
# -------------

# import ipdb; ipdb.set_trace()

if args.target[0] == 'Response':
    if params.use_tile is True:
        model = build_model_rsp(use_ge=params.use_ge,
                                use_dd=params.use_dd,
                                use_tile=params.use_tile,
                                ge_shape=ge_shape,
                                dd_shape=dd_shape,
                                model_type=params.model_type,
                                NUM_CLASSES=NUM_CLASSES)
    elif params.use_tile is False:
        model = build_model_rsp_baseline(use_ge=params.use_ge,
                                         use_dd=params.use_dd,
                                         ge_shape=ge_shape,
                                         dd_shape=dd_shape,
                                         model_type=params.model_type,
                                         NUM_CLASSES=NUM_CLASSES)
        x = {"ge_data": tr_ge.values, "dd_data": tr_dd.values}
        y = {"Response": ytr}
        validation_data = ({"ge_data": vl_ge.values,
                            "dd_data": vl_dd.values},
                           {"Response": yvl})
else:
    raise NotImplementedError('Need to check this method')
    model = build_model_rna()

print()
print(model.summary())

model.compile(loss=loss,
              optimizer=Adam(learning_rate=params.learning_rate),
              metrics=metrics)

# import ipdb; ipdb.set_trace()

t = time()
print('Start training ...')
if params.use_tile is True:
    history = model.fit(x=train_data,
                        steps_per_epoch=steps_per_epoch,
                        epochs=total_epochs,
                        verbose=1,
                        initial_epoch=toplayer_epochs,
                        validation_data=validation_data_for_training,
                        validation_steps=validation_steps,
                        callbacks=callbacks)
elif params.use_tile is False:
    history = model.fit(x=x,
                        y=y,
                        epochs=total_epochs,
                        verbose=1,
                        validation_data=validation_data,
                        callbacks=callbacks)
print('Runtime: {:.2f} mins'.format( (time() - t)/60) )


# --------------------------
# Evaluate
# --------------------------
# import ipdb; ipdb.set_trace()

def calc_per_tile_preds(data_with_smp_names, model, outdir):

    y_true, y_pred_prob, y_pred_label, smp_list = [], [], [], []
    for i, batch in enumerate(data_with_smp_names):
        fea = batch[0]
        label = batch[1]
        smp = batch[2]

        preds = model.predict(fea)
        y_pred_prob.append( preds )
        y_pred_label.extend( np.argmax(preds, axis=1).tolist() )
        y_true.extend( label[args.target[0]].numpy().tolist() )
        smp_list.extend( [smp_bytes.decode('utf-8') for smp_bytes in batch[2].numpy().tolist()] )

    # Put predictions in a dataframe
    y_pred_prob = np.vstack(y_pred_prob)
    y_pred_prob = pd.DataFrame(y_pred_prob, columns=[f'prob_{c}' for c in range(y_pred_prob.shape[1])])

    prd = pd.DataFrame({'smp': smp_list, 'y_true': y_true, 'y_pred_label': y_pred_label})
    prd = pd.concat([prd, y_pred_prob], axis=1)
    prd.to_csv(outdir/'test_preds_per_tiles.csv', index=False)
    return prd


def agg_per_smp_preds(prd, id_name, outdir):
    """ Agg predictions per smp. """
    aa = []
    for sample in prd[id_name].unique():
        dd = {id_name: sample}
        df = prd[prd[id_name] == sample]
        dd['y_true'] = df.y_true.unique()[0]
        dd['y_pred_label'] = np.argmax(np.bincount(df.y_pred_label))
        dd['pred_acc'] = sum(df.y_true == df.y_pred_label)/df.shape[0]
        aa.append(dd)

    agg_preds = pd.DataFrame(aa).sort_values(args.id_name).reset_index(drop=True)
    agg_preds.to_csv(outdir/'test_preds_per_smp.csv', index=False)

    # Efficient use of groupby().apply() !!
    # xx = prd.groupby('smp').apply(lambda x: pd.Series({
    #     'y_true': x['y_true'].unique()[0],
    #     'y_pred_label': np.argmax(np.bincount(x['y_pred_label'])),
    #     'pred_acc': sum(x['y_true'] == x['y_pred_label'])/x.shape[0]
    # })).reset_index().sort_values(args.id_name).reset_index(drop=True)
    # xx = xx.astype({'y_true': int, 'y_pred_label': int})
    # print(agg_preds.equals(xx))
    return agg_preds


import ipdb; ipdb.set_trace()
test_tile_preds = calc_per_tile_preds(test_data_with_smp_names, model=model, outdir=outdir)
test_smp_preds = agg_per_smp_preds(test_preds_tiles, id_name=args.id_name, outdir=outdir)

tile_scores = calc_scores(test_tile_preds['y_true'].values, test_tile_preds['prob_1'].values, mltype="cls")
smp_scores = calc_scores(test_smp_preds['y_true'].values, test_smp_preds['prob_1'].values, mltype="cls")
dump_dict(scores, outdir/"test_scores.txt")
dump_preds(yte_label, yte_prd[:, 1], te_meta, outpath=outdir/"test_preds.csv")

print('Done.')
