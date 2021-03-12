import os
import sys
assert sys.version_info >= (3, 5)

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
from sf_utils import read_annotations, _parse_tfrecord_function, _process_image, _interleave_tfrecords
from tfrecords import FEA_SPEC_RSP

# from sf_utils import read_annotations#, _parse_tfrecord_function, _process_image

fdir = Path(__file__).resolve().parent
from config import cfg

# Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


APPNAME = 'bin_rsp_balance_02'

# Load data
appdir = cfg.MAIN_APPDIR/APPNAME
annotations_file = appdir/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

pprint(data.ctype.value_counts())

ge_cols = [c for c in data.columns if c.startswith('ge_')]
dd_cols = [c for c in data.columns if c.startswith('dd_')]
# GE_LEN = len(ge_cols)
# DD_LEN = len(dd_cols)

# Scale attributes for RNA
ge_fea = data[ge_cols]
ge_scaler = StandardScaler()
ge_scaler.fit(ge_fea)
# GE_FEA_MEAN = ge_scaler.mean_
# GE_FEA_SCALE = ge_scaler.scale_

# Scale attributes for DD
dd_fea = data[dd_cols]
dd_scaler = StandardScaler()
dd_scaler.fit(dd_fea)
# DD_FEA_MEAN = dd_scaler.mean_
# DD_FEA_SCALE = dd_scaler.scale_


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

outcome_header = ['Response']
aux_headers = ['ctype', 'csite']  # (ap)
model_type = 'categorical'
use_fp16 = True
pretrain = 'imagenet'

label = '299px_302um'
label = f'{tile_px}px_{tile_um}um'
tfr_dir = cfg.SF_TFR_DIR_RSP/label  # (ap)


# import ipdb; ipdb.set_trace()

header, current_annotations = read_annotations(annotations_file)
ANNOTATIONS = current_annotations  # item in Dataset class instance
ann = ANNOTATIONS  # (ap)
print('Number of headers', len(header))
print('Number of samples', len(current_annotations))


# ---------------
# Create outcomes - done
# ---------------
# __init__ --> _trainer --> training_dataset.get_outcomes_from_annotations
# Inputs
headers = outcome_header
use_float = False
assigned_outcome=None

slides = sorted([a['slide'] for a in ANNOTATIONS])
filtered_annotations = ANNOTATIONS
results = {}

assigned_headers = {}
unique_outcomes = None

# We have only one header!
# for header in headers:
header = headers[0]

assigned_headers[header] = {}
filtered_outcomes = [a[header] for a in filtered_annotations]
unique_outcomes = list(set(filtered_outcomes))
unique_outcomes.sort()

# Create function to process/convert outcome
def _process_outcome(o):
    if use_float:
        return float(o)
    elif assigned_outcome:
        return assigned_outcome[o]
    else:
        return unique_outcomes.index(o)

# Assemble results dictionary
patient_outcomes = {}
num_warned = 0
warn_threshold = 3

for annotation in filtered_annotations:
    slide = annotation['slide']
    patient = annotation['submitter_id']
    annotation_outcome = _process_outcome(annotation[header])
    print_func = print if num_warned < warn_threshold else None

    # Mark this slide as having been already assigned an outcome with his header
    assigned_headers[header][slide] = True

    # Ensure patients do not have multiple outcomes
    if patient not in patient_outcomes:
        patient_outcomes[patient] = annotation_outcome
    elif patient_outcomes[patient] != annotation_outcome:
        log.error(f"Multiple different outcomes in header {header} found for patient {patient} ({patient_outcomes[patient]}, {annotation_outcome})", 1, print_func)
        num_warned += 1
    elif (slide in slides) and (slide in results) and (slide in assigned_headers[header]):
        continue

    if slide in slides:
        if slide in results:
            so = results[slide]['outcome']
            results[slide]['outcome'] = [so] if not isinstance(so, list) else so
            results[slide]['outcome'] += [annotation_outcome]
        else:
            results[slide] = {'outcome': annotation_outcome if not use_float else [annotation_outcome]}
            results[slide]['submitter_id'] = patient
            
if num_warned >= warn_threshold:
    log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)

outcomes = results
del results
print("\n'outcomes':")
print(type(outcomes))
print(len(outcomes))
print(list(outcomes.keys())[:5])
print(outcomes[list(outcomes.keys())[5]])

# ooooooooooooooooooooooooooooooo
# (ap) outcomes for drug response
# ooooooooooooooooooooooooooooooo
# import ipdb; ipdb.set_trace()
outcomes = {smp: {'outcome': o} for smp, o in zip(data[cfg.ID_NAME], data['Response'])}
print(type(outcomes))
print(len(outcomes))
print(list(outcomes.keys())[:5])
print(outcomes[list(outcomes.keys())[5]])


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
unique_outcomes = data['Response'].unique().astype(str)
outcome_labels = dict(zip(range(len(unique_outcomes)), unique_outcomes))
print(outcome_labels)


# ---------------
# Create manifest - done
# ---------------
# __init__ --> _trainer --> get_manifest --> update_manifest_at_dir

GREEN = '\033[92m'
ENDC = '\033[0m'
def green(text):
    return GREEN + str(text) + ENDC

manifest = {}

directory = tfr_dir
tfr_files = list(directory.glob('*.tfrec*'))

manifest_path = directory/"manifest.json"

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

    # Update manifest to use pull paths to tfrecords
    tfrecord_dir = str(tfr_dir)
    global_manifest = {}
    for record in manifest:
        global_manifest.update({os.path.join(tfrecord_dir, record): manifest[record]})

    MANIFEST = manifest = global_manifest

    print(f'Items in manifest {len(manifest)}')
    
    # Save manifest
    with open(manifest_path, "w") as data_file:
        json.dump(manifest, data_file, indent=1)
        
print(type(manifest))
print(len(manifest))
print(list(manifest.keys())[:5])
print(manifest[list(manifest.keys())[5]])


# ooooooooooooooooooooooooooooooo
# (ap) manifest
# ooooooooooooooooooooooooooooooo
# import ipdb; ipdb.set_trace()

directory = tfr_dir
tfr_files = list(directory.glob('*.tfrec*'))
mn_path = directory/"mn.json"
mn = {}

if mn_path.exists():
    print('Loading existing mn.')
    with open(mn_path, 'r') as file:
        global_mn = json.load(file)
    MANIFEST = mn = global_mn

else:
    print('Creating mn.')
    relative_tfrecord_paths = [f.name for f in tfr_files]
    #slide_names_from_annotations = [n.split('.tfr')[0] for n in relative_tfrecord_paths]  # not used

    # ap
    n = len(relative_tfrecord_paths)

    for i, rel_tfr in enumerate(relative_tfrecord_paths):
        tfr = str(directory/rel_tfr)
        mn.update({rel_tfr: {}})
        raw_dataset = tf.data.TFRecordDataset(tfr)
            
        print(f"\r\033[K + Verifying tiles in ({i+1} out of {n} tfrecords) {green(rel_tfr)}...", end="")
        total = 0  # total tiles in the tfrecord
        
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')  # get the slide name
            if slide not in mn[rel_tfr]:
                mn[rel_tfr][slide] = 1
            else:
                mn[rel_tfr][slide] += 1
            total += 1
            
        mn[rel_tfr]['total'] = total  # in case tfrecords contains tiles of multiple slides
        print('\r\033[K', end="")

    del raw_dataset

    # Update mn to use pull paths to tfrecords
    tfrecord_dir = str(tfr_dir)
    global_mn = {}
    for record in mn:
        global_mn.update({os.path.join(tfrecord_dir, record): mn[record]})

    MANIFEST = mn = global_mn
    print(f'Items in mn {len(mn)}')
    
    # Save mn
    with open(mn_path, "w") as file:
        json.dump(mn, file, indent=1)
        
print(type(mn))
print(len(mn))
print(list(mn.keys())[:5])
print(mn[list(mn.keys())[5]])


# --------------------------------------------------
# Create training_tfrecords and validation_tfrecords - done
# --------------------------------------------------
# __init__ --> _trainer --> sfio.tfrecords.get_training_and_validation_tfrecords
# Note: I use my processing to split data

from sklearn.model_selection import train_test_split
slides = list(outcomes.keys())
y = [outcomes[k]['outcome'] for k in outcomes.keys()]
s_tr, s_te, y_tr, y_te = train_test_split(slides, y, test_size=0.2, stratify=y, random_state=seed)

training_tfrecords = [str(directory/s)+'.tfrecords' for s in s_tr]
validation_tfrecords = [str(directory/s)+'.tfrecords' for s in s_te]

print(f'Training files {len(training_tfrecords)}')
print(f'Validation files {len(validation_tfrecords)}')


num_slide_input = 0
input_labels = None


# import ipdb; ipdb.set_trace()
print(len(outcomes))
print(len(manifest))
print(outcomes[list(outcomes.keys())[5]])
print(manifest[list(manifest.keys())[5]])


# -------------------------------
# create SlideflowModel model SFM
# -------------------------------
# import ipdb; ipdb.set_trace()

DATA_DIR = '/vol/ml/apartin/projects/slideflow-proj/sf_pdx_bin_rsp2/project/models/ctype-Xception_v0-kfold1';
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
NUM_SLIDE_INPUT = num_slide_input

#outcomes_ = [slide_annotations[slide]['outcome'] for slide in SLIDES]
outcomes_ = [slide_annotations[smp]['outcome'] for smp in SAMPLES]

normalizer = None

if model_type == 'categorical':
    NUM_CLASSES = len(list(set(outcomes_)))  # infer this from other variables

#ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SLIDES, outcomes_), -1)]
ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SAMPLES, outcomes_), -1)]


pretrain_model_format=None;
resume_training=None;
checkpoint=None
log_frequency=100;
multi_image=False;
validate_on_batch=16;
val_batch_size=32;
validation_steps=200
max_tiles_per_slide=0;
min_tiles_per_slide=0;
starting_epoch=0;
ema_observations=20;
ema_smoothing=2;
steps_per_epoch_override=None

AUGMENT = True


# with_tile = False
# with_ge = True
# with_tile = True
# with_ge = False
with_tile = True
with_ge = True


tfrecords = training_tfrecords
finite = False
min_tiles = 0
max_tiles = 0
include_slidenames = False
drop_remainder = False

datasets = []
datasets_categories = []
num_tiles = []
global_num_tiles = 0
categories = {}
categories_prob = {}
categories_tile_fraction = {}
parse_fn = _parse_tfrecord_function

# import ipdb; ipdb.set_trace()

for i, filename in enumerate(tfrecords):
    smp = filename.split('/')[-1][:-10]
    # smp_data = data[data['smp'] == smp] # (ap)
    # data[data['smp'] == smp].Response # (ap)

    if smp not in SAMPLES:
        continue

    # Determine total number of tiles available in TFRecord
    try:
        tiles = MANIFEST[filename]['total']
    except KeyError:
        print(f"Manifest not finished, unable to find {green(filename)}")

    # Ensure TFRecord has minimum number of tiles; otherwise, skip
    if not min_tiles and tiles == 0:
        print(f"Skipping empty tfrecord {green(smp)}")
        continue
    elif tiles < min_tiles:
        print(f"Skipping tfrecord {green(smp)}; has {tiles} tiles (minimum: {min_tiles})")
        continue

    # Assign category by outcome if this is a categorical model.
    # Otherwise, consider all slides from the same category (effectively skipping balancing);
    # appropriate for linear models.
    # (ap) Get the category of the current sample
    category = SLIDE_ANNOTATIONS[smp]['outcome'] if MODEL_TYPE == 'categorical' else 1
    # data[data['smp'] == smp].Response # (ap)

    # Create a list of tf datasets and a corresponding list of categories (outcomes)
    if filename not in DATASETS:
        # buffer_size=1024*1024*100 num_parallel_reads=tf.data.experimental.AUTOTUNE
        DATASETS.update({filename: tf.data.TFRecordDataset(filename, num_parallel_reads=32)}) 
    datasets += [DATASETS[filename]]
    datasets_categories += [category]

    # Cap number of tiles to take from TFRecord at maximum specified
    if max_tiles and tiles > max_tiles:
        print(f"Only taking maximum of {max_tiles} (of {tiles}) tiles from {green(filename)}")
        tiles = max_tiles

    if category not in categories.keys():
        # categories.update({category: {'num_slides': 1,
        #                               'num_tiles': tiles}})
        categories.update({category: {'num_samples': 1,
                                      'num_tiles': tiles}})
    else:
        # categories[category]['num_slides'] += 1
        # categories[category]['num_tiles'] += tiles
        categories[category]['num_samples'] += 1
        categories[category]['num_tiles'] += tiles
    num_tiles += [tiles]


# import ipdb; ipdb.set_trace()

# Assign weight to each category
#lowest_category_slide_count = min([categories[i]['num_slides'] for i in categories])
lowest_category_sample_count = min([categories[i]['num_samples'] for i in categories])
lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
for category in categories:
    #categories_prob[category] = lowest_category_slide_count / categories[category]['num_slides']
    categories_prob[category] = lowest_category_sample_count / categories[category]['num_samples']
    categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']

print(categories_prob)
print(categories_tile_fraction)


# (ap) Assign per-slide weight based on sample count per outcome (category-based/outcome-based stratification)
balance = balanced_training  # ap
print(balance)

if balance == BALANCE_BY_CATEGORY:
    prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
    if finite:
        # Only take as many tiles as the number of tiles in the smallest category -->
        # (ap) i.e., balance by the number of tiles
        for i in range(len(datasets)):
            num_tiles[i] = int(num_tiles[i] * categories_tile_fraction[datasets_categories[i]])
            print(f"Tile fraction (dataset {i+1}/{len(datasets)}): \
                  {categories_tile_fraction[datasets_categories[i]]}, \
                  taking {num_tiles[i]}")
        print(f"Global num tiles: {global_num_tiles}")


# Take the calculcated number of tiles from each dataset and calculate global number of tiles
for i in range(len(datasets)):
    datasets[i] = datasets[i].take(num_tiles[i])  # Take the calculcated number of tiles from each dataset
    if not finite:
        datasets[i] = datasets[i].repeat()  # (ap) why repeat()??
global_num_tiles = sum(num_tiles)


import ipdb; ipdb.set_trace()

# Interleave datasets
try:
    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights)
except IndexError:
    print(f"No TFRecords found after filter criteria; please ensure all tiles have been \
          extracted and all TFRecords are in the appropriate folder")
    #sys.exit()

if include_slidenames:
    dataset_with_slidenames = dataset.map(
            partial(parse_fn, include_slidenames=True, multi_image=multi_image,
                    with_ge=with_ge, with_tile=with_tile),
            num_parallel_calls=32
    ) #tf.data.experimental.AUTOTUNE
    dataset_with_slidenames = dataset_with_slidenames.batch(batch_size, drop_remainder=drop_remainder)

else:
    dataset_with_slidenames = None

dataset = dataset.map(
    partial(parse_fn, include_slidenames=False, multi_image=multi_image,
            with_ge=with_ge, with_tile=with_tile, ge_scaler=ge_scaler,  dd_scaler=dd_scaler),
    num_parallel_calls = 8
)

dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

import ipdb; ipdb.set_trace()

# train_data, _, num_tiles = _interleave_tfrecords(
train_data, train_data_with_slidenames, num_tiles = _interleave_tfrecords(
    tfrecords=TRAIN_TFRECORDS,
    batch_size=batch_size,
    balance=balanced_training,
    finite=False,
    max_tiles=max_tiles_per_slide,
    min_tiles=min_tiles_per_slide,
    # include_slidenames=False,
    include_slidenames=include_slidenames,
    multi_image=multi_image,
    parse_fn=None,
    with_ge=with_ge,
    with_tile=with_tile,
    MANIFEST=MANIFEST,
    ANNOTATIONS_TABLES=ANNOTATIONS_TABLES,
    SLIDE_ANNOTATIONS=SLIDE_ANNOTATIONS,
    MODEL_TYPE=MODEL_TYPE
)

using_validation = 25

if using_validation:
    #validation_data, validation_data_with_slidenames, _ = _build_dataset_inputs(
    validation_data, validation_data_with_slidenames, _ = _interleave_tfrecords(
        tfrecords=VALIDATION_TFRECORDS,
        batch_size=val_batch_size,
        balance='NO_BALANCE',
        finite=True,
        max_tiles=max_tiles_per_slide,
        min_tiles=min_tiles_per_slide,
        include_slidenames=True,
        multi_image=multi_image,
        parse_fn=None,
        with_ge=with_ge,
        with_tile=with_tile,
        MANIFEST=MANIFEST,
        ANNOTATIONS_TABLES=ANNOTATIONS_TABLES,
        SLIDE_ANNOTATIONS=SLIDE_ANNOTATIONS,
        MODEL_TYPE=MODEL_TYPE
    )

    if validation_steps:
        validation_data_for_training = validation_data.repeat()
        print(f"Using {validation_steps} batches ({validation_steps * batch_size} samples) each validation check")
    else:
        validation_data_for_training = validation_data
        #log.empty(f"Using entire validation set each validation check", 2)
        print(f"Using entire validation set each validation check")
else:
    log.info("Validation during training: None", 1)
    validation_data_for_training = None
    validation_steps = 0


print('Number of tfrecords in "datasets"', len(datasets))
bb = next(train_data.__iter__())
# bb = next(train_data_with_slidenames.__iter__())
print(type(bb))
# print(bb[1])
# print(bb[0].keys())
# print(bb[0]['tile_image'].shape)
# print(bb[0]['ge_data'].shape)

for ii, item in enumerate(bb):
    print(f"\nItem {ii}")
    if isinstance(item, dict):
        for i, k in enumerate(item.keys()):
            print(f"\t{k}: {item[k].numpy().shape}")

recs = []
for i, rec in enumerate(train_data.take(4)):
    recs.append(rec)
    tf.print(rec[1])
