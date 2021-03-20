import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path
from pprint import pprint
import json
import csv
import glob
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
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

from tf_utils import _float_feature, _bytes_feature, _int64_feature, get_tfr_files
from sf_utils import (green, interleave_tfrecords,
                      _parse_tfrec_fn_rsp, _parse_tfrec_fn_rna)
from models import build_model_rsp, build_model_rna

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

args, other_args = parser.parse_known_args()
pprint(args)

use_ge, use_dd = True, True
# use_ge, use_dd = True, False
use_tile = True

# import ipdb; ipdb.set_trace()
# APPNAME = 'bin_ctype_balance_02'
APPNAME = 'bin_rsp_balance_01'
# APPNAME = 'bin_rsp_balance_02'

# Load data
appdir = cfg.MAIN_APPDIR/APPNAME
annotations_file = appdir/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

# Outdir
split_on = 'none' if args.split_on is (None or 'none') else args.split_on
outdir = appdir/f'results/split_on_{split_on}'
os.makedirs(outdir, exist_ok=True)

if args.target[0] == 'Response':
    pprint(data.groupby(['ctype', 'Response']).agg({
        'index': 'nunique'}).reset_index().rename(columns={'index': 'count'}))
else:
    pprint(data[args.target[0]].value_counts())

ge_cols = [c for c in data.columns if c.startswith('ge_')]
dd_cols = [c for c in data.columns if c.startswith('dd_')]
data = data.astype({'image_id': str, 'slide': str})

def get_scaler(fea_df):
    if fea_df.shape[0] == 0:
        # TODO: add warning!
        return None
    scaler = StandardScaler()
    scaler.fit(fea_df)
    return scaler

# RNA scaler
if use_ge:
    ge_scaler = get_scaler(data[ge_cols])

# Descriptors scaler
if use_dd:
    dd_scaler = get_scaler(data[dd_cols])


BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'

tile_px = 299
tile_um = 302
# finetune_epochs = 10
finetune_epochs = 1
toplayer_epochs = 0
model = 'Xception'
pooling = 'max'

loss = 'sparse_categorical_crossentropy'
# loss={'csite_label': tf.keras.losses.categorical_crossentropy,
#     'ctype_label': tf.keras.losses.categorical_crossentropy},
# loss = {'ctype': tf.keras.losses.SparseCategoricalCrossentropy()}

learning_rate = 0.0001
batch_size = 16
hidden_layers = 1
hidden_layer_width = 500
optimizer = 'Adam'
early_stop = True 
early_stop_patience = 10
early_stop_method = 'loss'  # uses only validation metrics
balanced_training = 'BALANCE_BY_CATEGORY'
balanced_validation = 'NO_BALANCE'
trainable_layers = 0
L2_weight = 0
augment = True

# outcome_header = ['Response']
outcome_header = args.target
# aux_headers = ['ctype', 'csite']  # (ap)
model_type = 'categorical'
use_fp16 = True
pretrain = 'imagenet'

label = f'{tile_px}px_{tile_um}um'


if args.target[0] == 'Response':
    tfr_dir = cfg.SF_TFR_DIR_RSP
    parse_fn = _parse_tfrec_fn_rsp
elif args.target[0] == 'ctype':
    tfr_dir = cfg.SF_TFR_DIR_RNA_NEW
    parse_fn = _parse_tfrec_fn_rna
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

# ooooooooooooooooooooooooooooooo
# (ap) outcome_labels 
# ooooooooooooooooooooooooooooooo
# import ipdb; ipdb.set_trace()
# unique_outcomes = data[header].unique().astype(str)
# outcome_labels = dict(zip(range(len(unique_outcomes)), unique_outcomes))
# print(outcome_labels)


# ---------------
# Create manifest - done
# ---------------
# __init__ --> _trainer --> get_manifest --> update_manifest_at_dir

directory = tfr_dir
tfr_files = list(directory.glob('*.tfrec*'))
manifest_path = directory/"manifest.json"
manifest = {}

if manifest_path.exists():
    print('Loading existing manifest.')
    with open(manifest_path, 'r') as data_file:
        global_manifest = json.load(data_file)
        
    MANIFEST = manifest = global_manifest

else:
    print('Creating manifest.')
    relative_tfrecord_paths = [f.name for f in tfr_files]
    slide_names_from_annotations = [n.split('.tfr')[0] for n in relative_tfrecord_paths]

    # ap
    n = len(relative_tfrecord_paths)

    for i, rel_tfr in enumerate(relative_tfrecord_paths):
        # print(f'processing {i}')
        tfr = str(directory/rel_tfr)  #join(directory, rel_tfr)

        manifest.update({rel_tfr: {}})
        try:
            raw_dataset = tf.data.TFRecordDataset(tfr)
        except Exception as e:
            print(f"Unable to open TFRecords file with Tensorflow: {str(e)}")
            
        print(f"\r\033[K + Verifying tiles in ({i+1} out of {n} tfrecords) {green(rel_tfr)}...", end="")
        total = 0
        
        try:
            for raw_record in raw_dataset:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')  # get the slide name
                if slide not in manifest[rel_tfr]:
                    manifest[rel_tfr][slide] = 1
                else:
                    manifest[rel_tfr][slide] += 1
                total += 1
                
        except tf.python.framework.errors_impl.DataLossError:
            print('\r\033[K', end="")
            log.error(f"Corrupt or incomplete TFRecord at {tfr}", 1)
            log.info(f"Deleting and removing corrupt TFRecord from manifest...", 1)
            del(raw_dataset)
            os.remove(tfr)
            del(manifest[rel_tfr])
            continue
            
        manifest[rel_tfr]['total'] = total
        print('\r\033[K', end="")

    del raw_dataset

    # Update manifest to use full paths to tfrecords
    tfrecord_dir = str(tfr_dir)
    global_manifest = {}
    for record in manifest:
        global_manifest.update({os.path.join(tfrecord_dir, record): manifest[record]})

    MANIFEST = manifest = global_manifest

    print(f'Items in manifest {len(manifest)}')
    
    # Save manifest
    with open(manifest_path, "w") as data_file:
        json.dump(manifest, data_file, indent=1)
        
print('\nmanifest:')
print(type(manifest))
print(len(manifest))
print(list(manifest.keys())[:3])
print(manifest[list(manifest.keys())[3]])


# ooooooooooooooooooooooooooooooo
# (ap) manifest
# ooooooooooooooooooooooooooooooo
# import ipdb; ipdb.set_trace()

# directory = tfr_dir
# tfr_files = list(directory.glob('*.tfrec*'))
# mn_path = directory/"mn.json"
# mn = {}

# if mn_path.exists():
#     print('Loading existing mn.')
#     with open(mn_path, 'r') as file:
#         global_mn = json.load(file)
#     MANIFEST = mn = global_mn

# else:
#     print('Creating mn.')
#     relative_tfrecord_paths = [f.name for f in tfr_files]
#     #slide_names_from_annotations = [n.split('.tfr')[0] for n in relative_tfrecord_paths]  # not used

#     # ap
#     n = len(relative_tfrecord_paths)

#     for i, rel_tfr in enumerate(relative_tfrecord_paths):
#         tfr = str(directory/rel_tfr)
#         mn.update({rel_tfr: {}})
#         raw_dataset = tf.data.TFRecordDataset(tfr)
            
#         print(f"\r\033[K + Verifying tiles in ({i+1} out of {n} tfrecords) {green(rel_tfr)}...", end="")
#         total = 0  # total tiles in the tfrecord
        
#         for raw_record in raw_dataset:
#             example = tf.train.Example()
#             example.ParseFromString(raw_record.numpy())
#             slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')  # get the slide name
#             if slide not in mn[rel_tfr]:
#                 mn[rel_tfr][slide] = 1
#             else:
#                 mn[rel_tfr][slide] += 1
#             total += 1
            
#         mn[rel_tfr]['total'] = total  # in case tfrecords contains tiles of multiple slides
#         print('\r\033[K', end="")

#     del raw_dataset

#     # Update mn to use pull paths to tfrecords
#     tfrecord_dir = str(tfr_dir)
#     global_mn = {}
#     for record in mn:
#         global_mn.update({os.path.join(tfrecord_dir, record): mn[record]})

#     MANIFEST = mn = global_mn
#     print(f'Items in mn {len(mn)}')
    
#     # Save mn
#     with open(mn_path, "w") as file:
#         json.dump(mn, file, indent=1)

# print('\nmanifest:')
# print(type(mn))
# print(len(mn))
# print(list(mn.keys())[:3])
# print(mn[list(mn.keys())[3]])


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
# splitdir = appdir/'annotations.splits'
splitdir = appdir/f'annotations.splits/split_on_{split_on}'
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
# assert len(single_split_files) >= 2, f'The split {s} contains only one file.'
for id_file in single_split_files:
    if 'tr_id' in id_file:
        # tr_id = pd.read_csv(id_file).values.reshape(-1,)
        tr_id = cast_list(read_lines(id_file), int)
    elif 'vl_id' in id_file:
        # vl_id = pd.read_csv(id_file).values.reshape(-1,)
        vl_id = cast_list(read_lines(id_file), int)
    elif 'te_id' in id_file:
        # te_id = pd.read_csv(id_file).values.reshape(-1,)
        te_id = cast_list(read_lines(id_file), int)

# -----------------------------------------------
# Get data based on splits
# -----------------------------------------------
# Dataframes of T/V/E samples
tr_df = data.iloc[tr_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
vl_df = data.iloc[vl_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
te_df = data.iloc[te_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
print('Total samples {}'.format(tr_df.shape[0] + vl_df.shape[0] + te_df.shape[0]))

# List of sample names for T/V/E
tr_smp_names = list(tr_df[args.id_name].values)
vl_smp_names = list(vl_df[args.id_name].values)
te_smp_names = list(te_df[args.id_name].values)

# TFRecords filenames
tr_tfr_files = get_tfr_files(tfr_dir, tr_smp_names)  # training_tfrecords
vl_tfr_files = get_tfr_files(tfr_dir, vl_smp_names)  # validation_tfrecords
te_tfr_files = get_tfr_files(tfr_dir, te_smp_names)
print('Total samples {}'.format(len(tr_tfr_files) + len(vl_tfr_files) + len(te_tfr_files)))

# Missing tfrecords
print('\nThese samples miss a tfrecord ...\n')
print(data.loc[~data[args.id_name].isin(tr_smp_names + vl_smp_names + te_smp_names), ['smp', 'image_id']])

train_tfrecords = tr_tfr_files
val_tfrecords = vl_tfr_files
test_tfrecords = te_tfr_files

import ipdb; ipdb.set_trace()

# data = data.astype({args.id_name: str})
# dtr = data[data[args.id_name].isin(tr_smp_names)]
# dvl = data[data[args.id_name].isin(vl_smp_names)]
# dte = data[data[args.id_name].isin(te_smp_names)]

# assert sorted(tr_smp_names) == sorted(dtr[args.id_name].values.tolist()), "Sample names \
#     in the s_tr and dtr don't match."
# assert sorted(vl_smp_names) == sorted(dvl[args.id_name].values.tolist()), "Sample names \
#     in the s_te and dte don't match."
# assert sorted(te_smp_names) == sorted(dte[args.id_name].values.tolist()), "Sample names \
#     in the s_te and dte don't match."

assert sorted(tr_smp_names) == sorted(tr_df[args.id_name].values.tolist()), "Sample names \
    in the s_tr and dtr don't match."
assert sorted(vl_smp_names) == sorted(vl_df[args.id_name].values.tolist()), "Sample names \
    in the s_te and dte don't match."
assert sorted(te_smp_names) == sorted(te_df[args.id_name].values.tolist()), "Sample names \
    in the s_te and dte don't match."

os.makedirs(outdir/f'split_{split_id}', exist_ok=True)
tr_df.to_csv(outdir/f'split_{split_id}/dtr.csv', index=False)
vl_df.to_csv(outdir/f'split_{split_id}/dvl.csv', index=False)
te_df.to_csv(outdir/f'split_{split_id}/dte.csv', index=False)

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
DATA_DIR = appdir
os.makedirs(DATA_DIR, exist_ok=True)
MANIFEST = manifest
IMAGE_SIZE = image_size = tile_px
DTYPE = 'float16' if use_fp16 else 'float32'
SLIDE_ANNOTATIONS = slide_annotations = outcomes
TRAIN_TFRECORDS = train_tfrecords
VALIDATION_TFRECORDS = val_tfrecords
model_type = MODEL_TYPE = model_type
#SLIDES = list(slide_annotations.keys())
SAMPLES = list(slide_annotations.keys())
DATASETS = {}
# NUM_SLIDE_INPUT = num_slide_input

#outcomes_ = [slide_annotations[slide]['outcome'] for slide in SLIDES]
outcomes_ = [slide_annotations[smp]['outcome'] for smp in SAMPLES]

normalizer = None

if model_type == 'categorical':
    NUM_CLASSES = len(list(set(outcomes_)))  # infer this from other variables

#ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SLIDES, outcomes_), -1)]
ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SAMPLES, outcomes_), -1)]


pretrain_model_format = None
resume_training = None
checkpoint = None
log_frequency = 100
# multi_image = False
validate_on_batch = 16
val_batch_size = 32
validation_steps = 200
max_tiles_per_slide = 0
min_tiles_per_slide = 0
starting_epoch = 0  # what's that??
ema_observations = 20
ema_smoothing = 2
steps_per_epoch_override = None  # what's that??

AUGMENT = True


# import ipdb; ipdb.set_trace()

if args.target[0] == 'Response':
    # Response
    parse_fn = _parse_tfrec_fn_rsp
    parse_fn_kwargs = {
        'use_tile': use_tile,
        'use_ge': use_ge,
        'use_dd': use_dd,
        'ge_scaler': ge_scaler,
        'dd_scaler': dd_scaler,
        'id_name': args.id_name,
        'AUGMENT': AUGMENT,
        'ANNOTATIONS_TABLES': ANNOTATIONS_TABLES
    }

else:
    # Ctype
    parse_fn = _parse_tfrec_fn_rna
    parse_fn_kwargs = {
        'use_tile': use_tile,
        'use_ge': use_ge,
        'ge_scaler': ge_scaler,
        'id_name': args.id_name,
        'MODEL_TYPE': MODEL_TYPE,
        'ANNOTATIONS_TABLES': ANNOTATIONS_TABLES,
        'AUGMENT': AUGMENT,
    }

train_data, _, num_tiles = interleave_tfrecords(
    tfrecords=TRAIN_TFRECORDS,
    batch_size=batch_size,
    balance=balanced_training,
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

bb = next(train_data.__iter__())

# import ipdb; ipdb.set_trace()
if use_ge:
    ge_shape = bb[0]['ge_data'].numpy().shape[1:]
if use_dd:
    dd_shape = bb[0]['dd_data'].numpy().shape[1:]

for i, item in enumerate(bb):
    print(f"\nItem {i}")
    if isinstance(item, dict):
        for k in item.keys():
            print(f"\t{k}: {item[k].numpy().shape}")

recs = []
for i, rec in enumerate(train_data.take(4)):
    recs.append(rec)
    tf.print(rec[1])

using_validation = 25

if using_validation:
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
        print(f"Using {validation_steps} batches ({validation_steps * batch_size} samples) each validation check")
    else:
        validation_data_for_training = val_data
        print(f"Using entire validation set each validation check")
else:
    log.info("Validation during training: None", 1)
    validation_data_for_training = None
    validation_steps = 0

test_data, te_data_with_smp_names, _ = interleave_tfrecords(
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
if max([finetune_epochs]) <= starting_epoch:
    print(f"Starting epoch ({starting_epoch}) cannot be greater than the \
          maximum target epoch ({max(finetune_epochs)})")
    
if early_stop and early_stop_method == 'accuracy' and model_type != 'categorical':
    print(f"Unable to use early stopping method 'accuracy' with a non-categorical \
          model type (type: '{model_type}')")
    
if starting_epoch != 0:
    print(f"Starting training at epoch {starting_epoch}")
    
total_epochs = toplayer_epochs + (max([finetune_epochs]) - starting_epoch)
# TODO: this might need to change??
steps_per_epoch = round(num_tiles/batch_size) if steps_per_epoch_override is None else steps_per_epoch_override
results_log = os.path.join(DATA_DIR, 'results_log.csv')
metrics = ['accuracy'] if model_type != 'linear' else [loss]


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
    model = build_model_rsp(ge_shape=ge_shape, dd_shape=dd_shape,
                            model_type=model_type, NUM_CLASSES=NUM_CLASSES)
else:
    raise NotImplementedError('Need to check this method')
    model = build_model_rna()

print()
print(model.summary())

model.compile(loss=loss,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=metrics)

# import ipdb; ipdb.set_trace()

t = time()
print('Start training ...')
history = model.fit(train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    verbose=1,
                    initial_epoch=toplayer_epochs,
                    validation_data=validation_data_for_training,
                    validation_steps=validation_steps,
                    callbacks=callbacks)
print('Runtime: {:.2f} mins'.format( (time() - t)/60) )


# import ipdb; ipdb.set_trace()

# Predict
y_true, y_pred_prob, y_pred_label, smp_list = [], [], [], []
for i, batch in enumerate(te_data_with_smp_names):
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
prd.to_csv(outdir/'te_preds_per_tiles.csv', index=False)

# Agg predictions per smp
aa = []
for smp in prd.smp.unique():
    dd = {'smp': smp}
    df = prd[prd.smp == smp]
    dd['y_true'] = df.y_true.unique()[0]
    dd['y_pred_label'] = np.argmax(np.bincount(df.y_pred_label))
    dd['pred_acc'] = sum(df.y_true == df.y_pred_label)/df.shape[0]
    aa.append(dd)

import ipdb; ipdb.set_trace()

te_prd_agg = pd.DataFrame(aa).sort_values(args.id_name).reset_index(drop=True)
te_prd_agg.to_csv(outdir/'te_preds_per_smp.csv', index=False)

# efficient use of groupby().apply()
xx = prd.groupby('smp').apply(lambda x: pd.Series({
    'y_true': int(x['y_true'].unique()[0]),
    'y_pred_label': int(np.argmax(np.bincount(x['y_pred_label']))),
    'pred_acc': sum(x['y_true'] == x['y_pred_label'])/x.shape[0]
})).reset_index().sort_values(args.id_name).reset_index(drop=True)

print(te_prd_agg.equals(xx))

print('Done.')
