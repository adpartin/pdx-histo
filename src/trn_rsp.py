import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path
from pprint import pprint
import json
import csv
from functools import partial
from typing import List, Optional, Union

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
assert tf.__version__ >= "2.0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
AUTO = tf.data.experimental.AUTOTUNE

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

from tf_utils import _float_feature, _bytes_feature, _int64_feature
from sf_utils import (read_annotations, green,
                      _parse_tfrec_fn_rsp, _parse_tfrec_fn_rna,
                      _process_image, _interleave_tfrecords)
# from tfrecords import FEA_SPEC_RSP

fdir = Path(__file__).resolve().parent
from config import cfg

# Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


parser = argparse.ArgumentParser("Traing NN.")
parser.add_argument('-t', '--target',
                    type=str,
                    nargs='+',
                    default=['ctype'],
                    # default=['Response'],
                    choices=['Response', 'ctype', 'csite'],
                    help='Name of target output.')
parser.add_argument('--id_name',
                    type=str,
                    default='slide',
                    # default='smp',
                    choices=['slide', 'smp'],
                    help='Column name of the ID.')

args, other_args = parser.parse_known_args()
pprint(args)

use_ge, use_dd = True, True
# use_ge, use_dd = True, False
use_tile = True

# import ipdb; ipdb.set_trace()
# APPNAME = 'bin_ctype_balance_02'
# APPNAME = 'bin_rsp_balance_01'
APPNAME = 'bin_rsp_balance_02'

# Load data
appdir = cfg.MAIN_APPDIR/APPNAME
annotations_file = appdir/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

if args.target[0] == 'Response':
    pprint(data.groupby(['ctype', 'Response']).agg({
        'index': 'nunique'}).reset_index().rename(columns={'index': 'count'}))
else:
    pprint(data[args.target[0]].value_counts())

ge_cols = [c for c in data.columns if c.startswith('ge_')]
dd_cols = [c for c in data.columns if c.startswith('dd_')]
# meta_df = data.drop(columns=[ge_cols + dd_cols])
# meta_df.astype(str)
data = data.astype({'image_id': str, 'slide': str})


# Scale attributes for RNA
if len(ge_cols) > 0 and use_ge:
    ge_fea = data[ge_cols]
    ge_scaler = StandardScaler()
    ge_scaler.fit(ge_fea)
    del ge_fea
else:
    ge_scaler = None

# Scale attributes for DD
if len(dd_cols) > 0 and use_dd:
    dd_fea = data[dd_cols]
    dd_scaler = StandardScaler()
    dd_scaler.fit(dd_fea)
    del dd_fea
else:
    dd_scaler = None


BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'

tile_px = 299
tile_um = 302
finetune_epochs = 10
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
    #tfr_dir = cfg.SF_TFR_DIR_RNA
    tfr_dir = cfg.SF_TFR_DIR_RNA_NEW
    parse_fn = _parse_tfrec_fn_rna
tfr_dir = tfr_dir/label

# Annotations
header, current_annotations = read_annotations(annotations_file)
ANNOTATIONS = current_annotations  # item in Dataset class instance
ann = ANNOTATIONS  # (ap)
print('Number of headers', len(header))
print('Number of samples', len(current_annotations))


# ---------------
# Create outcomes - done
# ---------------
# __init__ --> _trainer --> training_dataset.get_outcomes_from_annotations

# # Inputs
# headers = outcome_header
# use_float = False
# assigned_outcome=None

# slides = sorted([a['slide'] for a in ANNOTATIONS])
# filtered_annotations = ANNOTATIONS
# results = {}

# assigned_headers = {}
# unique_outcomes = None

# # We have only one header!
# # for header in headers:
# header = headers[0]

# assigned_headers[header] = {}
# filtered_outcomes = [a[header] for a in filtered_annotations]
# unique_outcomes = list(set(filtered_outcomes))
# unique_outcomes.sort()

# # Create function to process/convert outcome
# def _process_outcome(o):
#     if use_float:
#         return float(o)
#     elif assigned_outcome:
#         return assigned_outcome[o]
#     else:
#         return unique_outcomes.index(o)

# # Assemble results dictionary
# patient_outcomes = {}
# num_warned = 0
# warn_threshold = 3

# for annotation in filtered_annotations:
#     slide = annotation['slide']
#     patient = annotation['submitter_id']
#     annotation_outcome = _process_outcome(annotation[header])
#     print_func = print if num_warned < warn_threshold else None

#     # Mark this slide as having been already assigned an outcome with his header
#     assigned_headers[header][slide] = True

#     # Ensure patients do not have multiple outcomes
#     if patient not in patient_outcomes:
#         patient_outcomes[patient] = annotation_outcome
#     elif patient_outcomes[patient] != annotation_outcome:
#         log.error(f"Multiple different outcomes in header {header} found for patient {patient} ({patient_outcomes[patient]}, {annotation_outcome})", 1, print_func)
#         num_warned += 1
#     elif (slide in slides) and (slide in results) and (slide in assigned_headers[header]):
#         continue

#     if slide in slides:
#         if slide in results:
#             so = results[slide]['outcome']
#             results[slide]['outcome'] = [so] if not isinstance(so, list) else so
#             results[slide]['outcome'] += [annotation_outcome]
#         else:
#             results[slide] = {'outcome': annotation_outcome if not use_float else [annotation_outcome]}
#             results[slide]['submitter_id'] = patient
            
# if num_warned >= warn_threshold:
#     log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

# # import ipdb; ipdb.set_trace()
# outcomes = results
# del results
# print("\n'outcomes':")
# print(type(outcomes))
# print(len(outcomes))
# print(list(outcomes.keys())[:3])
# print(outcomes[list(outcomes.keys())[3]])

# ooooooooooooooooooooooooooooooo
# (ap) outcomes for drug response
# ooooooooooooooooooooooooooooooo
# import ipdb; ipdb.set_trace()
outcomes = {}
unique_outcomes = list(set(data[args.target[0]].values))
unique_outcomes.sort()
# for smp, o in zip(data[args.id_name], data[args.target[0]]):
#     outcomes[smp] = {'outcome': unique_outcomes.index(o), 'submitter_id': smp}
for smp, o in zip(data[args.id_name], data[args.target[0]]):
    outcomes[smp] = {'outcome': unique_outcomes.index(o)}

# import ipdb; ipdb.set_trace()
# outcomes = {smp: {'outcome': o} for smp, o in zip(data[args.id_name], data[header])}

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

from sklearn.model_selection import train_test_split
# slides = list(outcomes.keys())
# y = [outcomes[k]['outcome'] for k in outcomes.keys()]
samples = list(outcomes.keys())
y = [outcomes[k]['outcome'] for k in outcomes.keys()]
s_tr, s_te, y_tr, y_te = train_test_split(samples, y, test_size=0.2,
                                          stratify=y, random_state=seed)

training_tfrecords = [str(directory/s)+'.tfrecords' for s in s_tr]
validation_tfrecords = [str(directory/s)+'.tfrecords' for s in s_te]

data = data.astype({args.id_name: str})
dtr = data[data[args.id_name].isin(s_tr)]
dte = data[data[args.id_name].isin(s_te)]

assert sorted(s_tr) == sorted(dtr[args.id_name].values.tolist()), "Sample names \
    in the s_tr and dtr don't match."
assert sorted(s_te) == sorted(dte[args.id_name].values.tolist()), "Sample names \
    in the s_te and dte don't match."

print(f'Training files {len(training_tfrecords)}')
print(f'Validation files {len(validation_tfrecords)}')


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
TRAIN_TFRECORDS = training_tfrecords
VALIDATION_TFRECORDS = validation_tfrecords
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

train_data, _, num_tiles = _interleave_tfrecords(
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
    validation_data, validation_data_with_slidenames, _ = _interleave_tfrecords(
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
        validation_data_for_training = validation_data.repeat()
        print(f"Using {validation_steps} batches ({validation_steps * batch_size} samples) each validation check")
    else:
        validation_data_for_training = validation_data
        print(f"Using entire validation set each validation check")
else:
    log.info("Validation during training: None", 1)
    validation_data_for_training = None
    validation_steps = 0


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
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
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
# Build network - ap
# -------------

import ipdb; ipdb.set_trace()

def build_model_rna(pooling='max', pretrain='imagenet'):
    # Image layers
    image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
    tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
    base_img_model = tf.keras.applications.Xception(
        weights=pretrain, pooling=pooling, include_top=False,
        input_shape=None, input_tensor=None)
    x_im = base_img_model(tile_input_tensor)

    # RNA layers
    ge_input_tensor = tf.keras.Input(shape=(976,), name="ge_data")
    x_ge = Dense(512, activation=tf.nn.relu)(ge_input_tensor)

    model_inputs = [tile_input_tensor, ge_input_tensor]

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")([x_ge, x_im])

    hidden_layer_width = 1000
    merged_model = tf.keras.layers.Dense(hidden_layer_width, activation=tf.nn.relu,
                                         name="hidden_1")(merged_model)

    # Add the softmax prediction layer
    activation = 'linear' if model_type == 'linear' else 'softmax'
    final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
    softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name='ctype')(final_dense_layer)

    # Assemble final model
    model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
    return model


def build_model_rsp(pooling='max', pretrain='imagenet',
                    use_ge=True, use_dd=True, use_tile=True,
                    ge_shape=None, dd_shape=None):
    """ ... """
    model_inputs = []
    merge_inputs = []

    if use_tile:
        image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
        tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
        base_img_model = tf.keras.applications.Xception(
            weights=pretrain, pooling=pooling, include_top=False,
            input_shape=None, input_tensor=None)

        x_im = base_img_model(tile_input_tensor)
        model_inputs.append(tile_input_tensor)
        merge_inputs.append(x_im)

    if use_ge:
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(512, activation=tf.nn.relu, name="dense_ge_1")(ge_input_tensor)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)

    if use_dd:
        dd_input_tensor = tf.keras.Input(shape=dd_shape, name="dd_data")
        x_dd = Dense(512, activation=tf.nn.relu, name="dense_dd_1")(dd_input_tensor)
        model_inputs.append(dd_input_tensor)
        merge_inputs.append(x_dd)

    # model_inputs = [tile_input_tensor, ge_input_tensor, dd_input_tensor]

    # Merge towers
    # merged_model = layers.Concatenate(axis=1, name="merger")([x_ge, x_dd, x_im])
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    hidden_layer_width = 1000
    merged_model = tf.keras.layers.Dense(hidden_layer_width, activation=tf.nn.relu,
                                         name="hidden_1", kernel_regularizer=None)(merged_model)

    # Add the softmax prediction layer
    activation = 'linear' if model_type == 'linear' else 'softmax'
    final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
    softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name="Response")(final_dense_layer)

    # Assemble final model
    model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
    return model


if args.target[0] == 'Response':
    model = build_model_rsp(ge_shape=ge_shape, dd_shape=dd_shape)
else:
    model = build_model_rna()

print()
model.summary()

# Fine-tune the model
print("Beginning fine-tuning")

model.compile(loss=loss,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=metrics)

history = model.fit(train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=total_epochs,
                    verbose=1,
                    initial_epoch=toplayer_epochs,
                    validation_data=validation_data_for_training,
                    validation_steps=validation_steps,
                    callbacks=callbacks)


