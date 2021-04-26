# https://www.codegrepper.com/code-examples/python/suppres+tensorflow+warnings
import os
import tempfile
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
fdir = Path(__file__).resolve().parent

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fdir = Path(__file__).resolve().parent

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/data/creditcard.zip',
    fname='creditcard.zip',
    extract=True)

raw_df = pd.read_csv(zip_path)


neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))


cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
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

sns.jointplot(pos_df['V5'], pos_df['V6'], kind='hex', xlim=(-5,5), ylim=(-5,5))
plt.suptitle("Positive distribution")

sns.jointplot(neg_df['V5'], neg_df['V6'], kind='hex', xlim=(-5,5), ylim=(-5,5))
_ = plt.suptitle("Negative distribution")


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


# import ipdb; ipdb.set_trace()


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


print(model.predict(train_features[:10]))


results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))


initial_bias = np.log([pos/neg])
print(initial_bias)


model = make_model(output_bias=initial_bias)
model.predict(train_features[:10])


results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))


initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)


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


def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'], color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'], color=colors[n], label='Val ' + label, linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)


model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels))


import csv
from collections import OrderedDict

class BatchCSVLogger(tf.keras.callbacks.Callback):
    """ Write training logs on every batch. """
    def __init__(self,
                 filename,
                 validate_on_batch=None,
                 validation_data=None,
                 validation_steps=None):
        """ ... """
        super(BatchCSVLogger, self).__init__()
        self.filename = filename
        self.validate_on_batch = validate_on_batch
        self.validation_data = validation_data
        self.validation_steps = validation_steps

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.step = 0  # global batch
        self.results = []
        
    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        self.epoch = epoch + 1

    def on_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        batch = batch + 1
        self.step += 1
        res = OrderedDict({"step": self.step, "epoch": self.epoch, "batch": batch})
        res.update(logs)  # logs contains the metrics for the training set

        if (self.validate_on_batch is not None) and (batch % self.validate_on_batch == 0):
            evals = self.model.evaluate(self.validation_data, verbose=0, steps=self.validation_steps)
            val_logs = {"val_"+str(k): v for k, v in zip(keys, evals)}
            res.update(val_logs)
        else:
            val_logs = {"val_"+str(k): np.nan for k in keys}
            res.update(val_logs)

        if self.step == 1:
            # keys = list(logs.keys())
            val_keys = ["val_"+str(k) for k in keys]
            fieldnames = ["step", "epoch", "batch"] + keys + val_keys + ["lr"]
            self.fieldnames = fieldnames

            self.csv_file = open(self.filename, "w")
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.csv_file.flush()

        # Get the current learning rate from model's optimizer
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        res.update({"lr": lr})
        self.writer.writerow(res)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()


import ipdb; ipdb.set_trace()
callbacks = []
outdir = fdir/"../test.out/test_callbacks"
os.makedirs(outdir, exist_ok=True)
val_data = (val_features, val_labels)
callbacks.append(BatchCSVLogger(filename=outdir/"batch_training.log", 
                                validate_on_batch=50,
                                validation_data=val_data))


# Test BatchCSVLogger
model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    # callbacks=[early_stopping],
    callbacks=callbacks,
    validation_data=val_data)

print("\nDone.")
