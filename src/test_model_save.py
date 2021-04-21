import os
import sys
import tempfile

import glob
from pathlib import Path
from pprint import pprint, pformat

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
assert tf.__version__ >= "2.0"

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

# import ipdb; ipdb.set_trace()

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
# from src.config import cfg
# from src.models import build_model_rsp, build_model_rsp_baseline, keras_callbacks
# from src.ml.scale import get_scaler
# from src.ml.evals import calc_scores, calc_preds, dump_preds, save_confusion_matrix
# from src.ml.keras_utils import plot_prfrm_metrics
# from src.utils.classlogger import Logger
# from src.utils.utils import (cast_list, create_outdir, create_outdir_2, dump_dict, get_print_func,
#                              read_lines, Params, Timer)
# from src.datasets.tidy import split_data_and_extract_fea, extract_fea, TidyData
# from src.tf_utils import get_tfr_files, calc_records_in_tfr_files, count_data_items
# from src.sf_utils import (create_tf_data, calc_class_weights,
#                           parse_tfrec_fn_rsp, parse_tfrec_fn_rna,
#                           create_manifest)
# from src.sf_utils import bold, green, blue, yellow, cyan, red

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip"
# cache_dir = fdir/"cache_test_datasets/"

tf.keras.utils.get_file(
    fname="creditcard.zip",
    origin=url,
    # cache_dir="/tmp/datasets/",
    cache_dir=None,
    extract=True)

# raw_df = pd.read_csv(cache_dir/"creditcard.csv")
raw_df = pd.read_csv(Path("~/.keras/datasets/")/"creditcard.csv")

neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

cleaned_df = raw_df.copy()
cleaned_df.pop('Time')
eps = 0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = make_model()
model.summary()

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

initial_bias = np.log([pos/neg])
initial_bias

model = make_model(output_bias=initial_bias)
model.predict(train_features[:10])

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

import ipdb; ipdb.set_trace()

# initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
initial_weights = fdir/'../initial_weights'
model.save_weights(initial_weights)

model.save(fdir/'../saved_model')
model1 = tf.keras.models.load_model(fdir/'../saved_model')

res  = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
res1 = model1.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)

model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0)

model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0)


