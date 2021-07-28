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
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip"

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
eps = 0.001
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
    # Define
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])
    # Compile
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
print(initial_bias)

model = make_model(output_bias=initial_bias)
model.predict(train_features[:10])

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

import ipdb; ipdb.set_trace()

# Save weights only
# initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
initial_weights = fdir/'../initial_weights'
model.save_weights(initial_weights)

# Save weights
model1 = make_model()
model1.load_weights(initial_weights)

# Save model
model.save(fdir/'../saved_model')
model2 = tf.keras.models.load_model(fdir/'../saved_model')

res  = model.evaluate( train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
res1 = model1.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
res2 = model2.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print(res)
print(res1)
print(res2)

# Create model and load weights
# epochs = 20
epochs = 1

model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=epochs,
    validation_data=(val_features, val_labels),
    verbose=0)

model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=epochs,
    validation_data=(val_features, val_labels),
    verbose=0)

# --------------------------------------------------------
# import ipdb; ipdb.set_trace()
# model = tf.keras.applications.Xception(weights="imagenet", pooling="avg")

# initial_weights = fdir/'../initial_weights'
# model.save_weights(initial_weights)
# # model_loaded_wts = tf.keras.applications.Xception(weights=initial_weights, pooling="avg")
# model_loaded_wts = tf.keras.applications.Xception(pooling="avg")
# model_loaded_wts.load_weights(initial_weights)
# print(type(model_loaded_wts))

# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = optimizers.Adam(learning_rate=0.1)
# model.compile(loss=loss, optimizer=optimizer)

# import ipdb; ipdb.set_trace()
# model.save(fdir/"../saved_model")
# model_loaded_full = tf.keras.models.load_model(fdir/"../saved_model")

# print("\nOriginal {}:".format(model.evaluate(val_data, steps=vl_steps)))
# print("\nWeights  {}:".format(model_loaded_wts.evaluate(val_data, steps=vl_steps)))
# print("\nFull     {}:".format(model_loaded_full.evaluate(val_data, steps=vl_steps)))
# --------------------------------------------------------

