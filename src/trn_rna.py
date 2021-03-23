"""
Train ctype classifier using rna data (w/o tfrecords).
This code can adjusted for drug response prediction.
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

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
from tensorflow.keras.models import Sequential, Model


fdir = Path(__file__).resolve().parent
from config import cfg

# Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


prjname = 'bin_ctype_balance_02'

# Load dataframe (annotations)
prjdir = cfg.MAIN_PRJDIR/prjname
annotations_file = prjdir/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

# Args
epochs = 3
batch_size = 32
# y_encoding = 'onehot'
y_encoding = 'label'


cls_label = 'ctype'
# display(data.groupby(['ctype', 'ctype_label', 'Response']).agg({'smp': 'nunique'}).reset_index().rename(columns={'smp': 'samples'}))
pprint(data[cls_label].value_counts())

ge_cols = [c for c in data.columns if c.startswith('ge_')]
GE_LEN = len(ge_cols)


# # Create balanced dataset on ctype
# n_classes = 2
# cls_label = 'ctype'
# min_count = pdx_meta[cls_label].value_counts().iloc[n_classes-1]

# dfs = []
# for ctype in pdx_meta[cls_label].value_counts().index[:n_classes]:
#     df = pdx_meta[pdx_meta[cls_label].isin([ctype])]
#     df = df.sample(n=min_count, random_state=seed)
#     dfs.append(df)

# pdx_meta = pd.concat(dfs, axis=0).sort_values('Sample', ascending=True).reset_index(drop=True)
# pdx_rna = pdx_rna[pdx_rna.Sample.isin(pdx_meta.Sample.values)].sort_values('Sample', ascending=True).reset_index(drop=True)

# del dfs, df

# print(pdx_meta.shape)
# print(pdx_rna.shape)
# pdx_meta.ctype.value_counts()

# Onehot encoding
ydata = data[cls_label].values
# y_onehot = pd.get_dummies(ydata_label)
y_onehot = pd.get_dummies(ydata)
ydata_label = np.argmax(y_onehot.values, axis=1)
n_classes = len(np.unique(ydata_label))

import ipdb; ipdb.set_trace()
# Scale RNA
xdata = data[ge_cols]
x_scaler = StandardScaler()
x1 = pd.DataFrame(x_scaler.fit_transform(xdata), columns=ge_cols, dtype=np.float32)

# X = xdata.values
# x_mean = np.nanmean(X, axis=0)
# x_scale = np.nanstd(X, axis=0, ddof=0)
# x2 = (X - x_mean)/x_scale
# print(sum(x_scale == 0))
# print(sum(np.ravel(np.isnan(x2))))

# print(x_mean[:10])
# print(x_scaler.mean_[:10])
# print(max(x_mean - x_scaler.mean_))
# print(x_scale[:10])
# print(x_scaler.scale_[:10])
# print(max(x_scale - x_scaler.scale_))
# print(x1.iloc[0, :10].values)
# print(x2[0, :10])
# print(max(np.ravel(x2 - x1.values)))
# x2 = pd.DataFrame(x2, columns=ge_cols, dtype=np.float32)
# print(np.isnan(x2).sum(axis=0).sort_values(ascending=False))

xdata = x1; del x1
# xdata = x2; del x2

# Split
tr_ids, te_ids = train_test_split(range(xdata.shape[0]), test_size=0.2, random_state=seed, stratify=ydata)
xtr = xdata.iloc[tr_ids, :].reset_index(drop=True)
xte = xdata.iloc[te_ids, :].reset_index(drop=True)

if y_encoding == 'onehot':
    ytr = y_onehot.iloc[tr_ids, :].reset_index(drop=True)
    yte = y_onehot.iloc[te_ids, :].reset_index(drop=True)
    loss = losses.CategoricalCrossentropy()
elif y_encoding == 'label':
    ytr = ydata_label[tr_ids]
    yte = ydata_label[te_ids]
    loss = losses.SparseCategoricalCrossentropy()
else:
    raise ValueError(f'Unknown value for y_encoding ({y_encoding}).')
    
ytr_label = ydata_label[tr_ids]
yte_label = ydata_label[te_ids]

# Model
inputs = Input(shape=(xtr.shape[1],), name='rna_input')
# x = Dense(4, activation=tf.nn.relu)(inputs)
x = Dense(256, activation=tf.nn.relu)(inputs)
x = Dropout(0.1)(x)
outputs = Dense(n_classes, activation=tf.nn.softmax, name='ctype')(x)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(loss=loss,
              optimizer=optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

pprint(model.summary())


# Train
history = model.fit(xtr, ytr,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(xte, yte))

yte_prd = model.predict(xte)
yte_prd_label = np.argmax(yte_prd, axis=1)
# yte_true_label = np.argmax(yte.values, axis=1)

cnf_mtrx = confusion_matrix(yte_label, yte_prd_label)
# disp = ConfusionMatrixDisplay(cnf_mtrx, display_labels=list(y_onehot.columns))
# disp.plot(xticks_rotation=65);
pprint(cnf_mtrx)

print('\nDone.')
