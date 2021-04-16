from pathlib import Path
from typing import Optional, List

import tensorflow as tf
assert tf.__version__ >= "2.0"
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# AUTO = tf.data.experimental.AUTOTUNE

import numpy as np

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

fdir = Path(__file__).resolve().parent
from config import cfg
from src.sf_utils import bold, green, blue, yellow, cyan, red


_ModelDict = {
    "Xception": tf.keras.applications.Xception,
    "EfficientNetB1": tf.keras.applications.EfficientNetB1,
    "EfficientNetB2": tf.keras.applications.EfficientNetB2,
    "EfficientNetB3": tf.keras.applications.EfficientNetB3,
    "EfficientNetB4": tf.keras.applications.EfficientNetB4
}


def keras_callbacks(outdir, monitor="val_loss", **mycallback_kwargs):
    """ ... """
    callbacks = []

    csv_logger = CSVLogger(outdir/"training.log")
    callbacks.append(csv_logger)

    # checkpointer = ModelCheckpoint(str(outdir/"model_best_at_{epoch}.ckpt"),
    #                                monitor=monitor,
    #                                verbose=0,
    #                                save_weights_only=False,
    #                                save_best_only=True,
    #                                save_freq="epoch")
    # callbacks.append(checkpointer)

    # reduce_lr = ReduceLROnPlateau(monitor=monitor,
    #                               factor=0.5,
    #                               patience=5,
    #                               verbose=1,
    #                               mode="auto",
    #                               min_delta=0.0001,
    #                               cooldown=0,
    #                               min_lr=0)
    # callbacks.append(reduce_lr)

    # early_stop = EarlyStopping(monitor=monitor,
    #                            patience=20,
    #                            mode="auto",
    #                            restore_best_weights=True,
    #                            verbose=1)
    # callbacks.append(early_stop)

    # mycallback = EarlyStopOnBatch(monitor="loss", **mycallback_kwargs)
    # callbacks.append(mycallback)

    return callbacks


class EarlyStopOnBatch(tf.keras.callbacks.Callback):
    """
    EarlyStopOnBatch(monitor="loss", batch_patience=20, validate_on_batch=10)
    """
    def __init__(self,
                 validation_data,
                 validation_steps=None,
                 validate_on_batch=100,
                 monitor="loss",
                 batch_patience=0,
                 print_fn=print):
        """ 
        Args:
            validate_on_batch : 
        """
        super(EarlyStopOnBatch, self).__init__()
        self.batch_patience = batch_patience
        self.best_weights = None
        # self.monitor = monitor  # ap
        self.validate_on_batch = validate_on_batch  # ap
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.print_fn = print_fn

    def on_train_begin(self, logs=None):
        # self.wait = 0           # number of batches it has waited when loss is no longer minimum
        self.stopped_epoch = 0  # epoch the training stops at
        self.stopped_batch = 0  # epoch the training stops at
        self.best = np.Inf      # init the best as infinity
        self.track_values = []  # ap
        self.epoch = None
        self.val_loss = np.Inf
        self.step_id = 0
        # self.print_fn("\n{}.".format(yellow("Start training")))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        self.epoch = epoch + 1
        self.wait = 0  # number of batches it has waited when loss is no longer minimum
        # self.print_fn("\n{} {}.\n".format( yellow("Start epoch"), yellow(epoch)) )

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        # outpath = str(outdir/f"model_at_epoch_{self.epoch}")
        # self.model.save(outpath)
        # self.print_fn("")

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())  # metrics/logs names
        batch = batch + 1

        self.step_id += 1
        results["step_id"] = self.step_id
        results["step_id"] = {"epoch": self.epoch, "batch": batch}
        # train_logs = {l: logs[l] for l in logs}
        results["step_id"].update(logs)

        if batch % self.validate_on_batch == 0:
            evals = self.model.evaluate(self.validation_data, verbose=0, steps=self.validation_steps)
            # self.track_values.append(self.val_loss)
            val_logs = {"val"+str(k): v for k, v in zip(keys, evals)}
            results["step_id"].update(val_logs)

            self.val_loss = evals[0]
            # current = logs.get("loss")
            # current = logs.get(self.monitor)
            current = self.val_loss

            if np.less(current, self.best):
                self.best = current
                self.wait = 0
                # Record the best weights if current results is better (less).
                self.best_weights = self.model.get_weights()

            else:
                self.wait += 1
                # if self.wait >= self.batch_patience:
                #     self.stopped_epoch = self.epoch
                #     self.stopped_batch = batch
                #     self.model.stop_training = True
                #     self.print_fn("\n{}".format(bold("Terminate training")))
                #     self.print_fn("Restores model weights from the best (epoch, batch).")
                #     self.model.set_weights(self.best_weights)

            if self.epoch == 1:
                self.print_fn("\repoch: {}, batch: {}, loss {:.4f}, val_loss {:.4f}, best_val_loss {:.4f} (wait: {})".format(
                    self.epoch, batch, logs["loss"], self.val_loss, self.best, yellow(self.wait)), end="")
            else:
                self.print_fn("\repoch: {}, batch: {}, loss {:.4f}, val_loss {:.4f}, best_val_loss {:.4f} (wait: {})".format(
                    self.epoch, batch, logs["loss"], self.val_loss, self.best, red(self.wait)), end="")

            # Don't terminate of the first epoch
            if (self.wait >= self.batch_patience) and (self.epoch > 1):
                self.stopped_epoch = self.epoch
                self.stopped_batch = batch
                self.model.stop_training = True
                self.print_fn("\n{}".format(red("Terminate training")))
                self.print_fn("Restores model weights from the best (epoch, batch).")
                self.model.set_weights(self.best_weights)

        else:
            val_logs = {"val"+str(k): np.nan for k in keys}
            results["step_id"].update(val_logs)

    def on_train_end(self, logs=None):
        if self.stopped_batch > 0:
            self.print_fn("Early stopping. Epoch: {}. Batch: {}.".format(self.stopped_epoch, self.stopped_batch))


class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    def on_train_batch_end(self, batch, logs=None):
        """ During training; when calling model.fit(). """
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        """ During evaluating; when calling model.evaluate(). """
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# def build_model_rsp_baseline(use_ge=True, use_dd=True,
#                              ge_shape=None, dd_shape=None, model_type="categorical",
#                              NUM_CLASSES=None, output_bias=None):
#     """ Doesn't use image data. """
#     if output_bias is not None:
#         output_bias = tf.keras.initializers.Constant(output_bias)

#     model_inputs = []
#     merge_inputs = []

#     if use_ge:
#         ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
#         x_ge = Dense(512, activation=tf.nn.relu, name="dense_ge_1")(ge_input_tensor)
#         # x_ge = BatchNormalization(0.2)(x_ge)
#         # x_ge = Dropout(0.4)(x_ge)
#         model_inputs.append(ge_input_tensor)
#         merge_inputs.append(x_ge)

#     if use_dd:
#         dd_input_tensor = tf.keras.Input(shape=dd_shape, name="dd_data")
#         x_dd = Dense(512, activation=tf.nn.relu, name="dense_dd_1")(dd_input_tensor)
#         # x_dd = BatchNormalization(0.2)(x_dd)
#         # x_dd = Dropout(0.4)(x_dd)
#         model_inputs.append(dd_input_tensor)
#         merge_inputs.append(x_dd)

#     # Merge towers
#     merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

#     # hidden_layer_width = 1000
#     hidden_layer_width = 500
#     merged_model = tf.keras.layers.Dense(hidden_layer_width, activation=tf.nn.relu,
#                                          name="hidden_1", kernel_regularizer=None)(merged_model)
#     merged_model = Dropout(0.4)(merged_model)

#     # Add the softmax prediction layer
#     # activation = "linear" if model_type == "linear" else "softmax"
#     # final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
#     # softmax_output = tf.keras.layers.Activation(activation, dtype="float32", name="Response")(final_dense_layer)

#     softmax_output = tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias, name="Response")(merged_model)

#     # Assemble final model
#     model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
#     return model


# def build_model_rsp(pooling='max', pretrain='imagenet',
#                     use_ge=True, use_dd=True, use_tile=True,
#                     ge_shape=None, dd_shape=None, model_type='categorical',
#                     NUM_CLASSES=None):
#     """ ... """
#     model_inputs = []
#     merge_inputs = []

#     if use_tile:
#         image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
#         tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
#         base_img_model = tf.keras.applications.Xception(
#             weights=pretrain, pooling=pooling, include_top=False,
#             input_shape=None, input_tensor=None)

#         x_im = base_img_model(tile_input_tensor)
#         model_inputs.append(tile_input_tensor)
#         merge_inputs.append(x_im)

#     if use_ge:
#         ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
#         x_ge = Dense(512, activation=tf.nn.relu, name="dense_ge_1")(ge_input_tensor)
#         model_inputs.append(ge_input_tensor)
#         merge_inputs.append(x_ge)

#     if use_dd:
#         dd_input_tensor = tf.keras.Input(shape=dd_shape, name="dd_data")
#         x_dd = Dense(512, activation=tf.nn.relu, name="dense_dd_1")(dd_input_tensor)
#         model_inputs.append(dd_input_tensor)
#         merge_inputs.append(x_dd)

#     # Merge towers
#     merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

#     hidden_layer_width = 1000
#     merged_model = tf.keras.layers.Dense(hidden_layer_width, activation=tf.nn.relu,
#                                          name="hidden_1", kernel_regularizer=None)(merged_model)

#     # Add the softmax prediction layer
#     activation = 'linear' if model_type == 'linear' else 'softmax'
#     final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
#     softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name="Response")(final_dense_layer)

#     # Assemble final model
#     model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
#     return model


def build_model_rsp_baseline(use_ge=True, use_dd1=True, use_dd2=True,
                             ge_shape=None, dd_shape=None, model_type="categorical",
                             # NUM_CLASSES=None,
                             output_bias=None):
    """ Doesn't use image data. """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model_inputs = []
    merge_inputs = []

    if use_ge:
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(512, activation=tf.nn.relu, name="dense_ge_1")(ge_input_tensor)
        x_ge = BatchNormalization()(x_ge)
        # x_ge = Dropout(0.4)(x_ge)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)
        del ge_input_tensor, x_ge

    if use_dd1:
        dd1_input_tensor = tf.keras.Input(shape=dd_shape, name="dd1_data")
        x_dd1 = Dense(256, activation=tf.nn.relu, name="dense_dd1_1")(dd1_input_tensor)
        x_dd1 = BatchNormalization()(x_dd1)
        # x_dd1 = Dropout(0.4)(x_dd1)
        model_inputs.append(dd1_input_tensor)
        merge_inputs.append(x_dd1)
        del dd1_input_tensor, x_dd1

    if use_dd2:
        dd2_input_tensor = tf.keras.Input(shape=dd_shape, name="dd2_data")
        x_dd2 = Dense(256, activation=tf.nn.relu, name="dense_dd2_1")(dd2_input_tensor)
        x_dd2 = BatchNormalization()(x_dd2)
        # x_dd2 = Dropout(0.4)(x_dd2)
        model_inputs.append(dd2_input_tensor)
        merge_inputs.append(x_dd2)
        del dd2_input_tensor, x_dd2

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    # hidden_layer_width = 1000
    hidden_layer_width = 500
    merged_model = tf.keras.layers.Dense(hidden_layer_width, activation=tf.nn.relu,
                                         name="top_hidden_1", kernel_regularizer=None)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    merged_model = Dropout(0.5)(merged_model)

    # Add the softmax prediction layer
    # activation = "linear" if model_type == "linear" else "softmax"
    # final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
    # softmax_output = tf.keras.layers.Activation(activation, dtype="float32", name="Response")(final_dense_layer)

    softmax_output = tf.keras.layers.Dense(1, activation="sigmoid", bias_initializer=output_bias, name="Response")(merged_model)

    # Assemble final model
    model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
    return model


def build_model_rsp(use_ge=True, use_dd1=True, use_dd2=True, use_tile=True,
                    ge_shape=None, dd_shape=None,
                    dense1_ge=512, dense1_dd1=256, dense1_dd2=256, dense1_top=1024,
                    top_hidden_dropout1=0.1,
                    output_bias=None,
                    # model_type="categorical",
                    base_image_model="Xception",
                    pooling="max",
                    pretrain="imagenet",
                    loss=losses.BinaryCrossentropy(),
                    optimizer=optimizers.Adam(learning_rate=0.0001)):
    """ ...
    refs:
        https://github.com/jkjung-avt/keras_imagenet/blob/master/utils/dataset.py
    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
        
    model_inputs = []
    merge_inputs = []

    if use_tile:
        image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
        tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
        # base_img_model = tf.keras.applications.Xception(
        #     weights=pretrain, pooling=pooling, include_top=False,
        #     input_shape=None, input_tensor=None)
        base_img_model = _ModelDict[base_image_model](
            include_top=False,
            weights=pretrain,
            input_shape=None,
            input_tensor=None,
            pooling=pooling, 
        )

        x_im = base_img_model(tile_input_tensor)
        model_inputs.append(tile_input_tensor)
        merge_inputs.append(x_im)
        del tile_input_tensor, x_im

    if use_ge:
        # dense1_ge = 512
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(dense1_ge, activation=tf.nn.relu, name="dense1_ge")(ge_input_tensor)
        x_ge = BatchNormalization(name="ge_batchnorm")(x_ge)
        # x_ge = Dropout(0.4)(x_ge)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)
        del ge_input_tensor, x_ge

    if use_dd1:
        # dense1_dd1 = 256
        dd1_input_tensor = tf.keras.Input(shape=dd_shape, name="dd1_data")
        x_dd1 = Dense(dense1_dd1, activation=tf.nn.relu, name="dense1_dd1")(dd1_input_tensor)
        x_dd1 = BatchNormalization(name="dd1_batchnorm")(x_dd1)
        # x_dd1 = Dropout(0.4)(x_dd1)
        model_inputs.append(dd1_input_tensor)
        merge_inputs.append(x_dd1)
        del dd1_input_tensor, x_dd1

    if use_dd2:
        # dense1_dd2 = 256
        dd2_input_tensor = tf.keras.Input(shape=dd_shape, name="dd2_data")
        x_dd2 = Dense(dense1_dd2, activation=tf.nn.relu, name="dense1_dd2")(dd2_input_tensor)
        x_dd2 = BatchNormalization(name="dd2_batchnorm")(x_dd2)
        # x_dd2 = Dropout(0.4)(x_dd2)
        model_inputs.append(dd2_input_tensor)
        merge_inputs.append(x_dd2)
        del dd2_input_tensor, x_dd2

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    # dense1_top = 1024
    merged_model = tf.keras.layers.Dense(dense1_top, activation=tf.nn.relu,
                                         name="dense1_top", kernel_regularizer=None)(merged_model)
    merged_model = BatchNormalization(name="top_batchnorm")(merged_model)
    # top_hidden_dropout1 = 0.1
    if top_hidden_dropout1 > 0:
        merged_model = Dropout(top_hidden_dropout1)(merged_model)

    # Add the softmax prediction layer
    # activation = 'linear' if model_type == 'linear' else 'softmax'
    # final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
    # softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name="Response")(final_dense_layer)

    softmax_output = tf.keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=output_bias, name="Response")(merged_model)

    # Assemble final model
    model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)

    # Compile
    # Note! When I put metrics in model.py, it immediately occupies a lot of the GPU memory!
    metrics = [
          # keras.metrics.TrueNegatives(name="tn"),
          keras.metrics.FalsePositives(name="fp"),
          # keras.metrics.FalseNegatives(name="fn"),
          keras.metrics.TruePositives(name="tp"),
          # keras.metrics.AUC(name="auc"),
    ]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


# def build_model_rna(pooling='max', pretrain='imagenet'):
#     # Image layers
#     image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
#     tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
#     base_img_model = tf.keras.applications.Xception(
#         weights=pretrain, pooling=pooling, include_top=False,
#         input_shape=None, input_tensor=None)
#     x_im = base_img_model(tile_input_tensor)

#     # RNA layers
#     ge_input_tensor = tf.keras.Input(shape=(976,), name="ge_data")
#     x_ge = Dense(512, activation=tf.nn.relu)(ge_input_tensor)

#     model_inputs = [tile_input_tensor, ge_input_tensor]

#     # Merge towers
#     merged_model = layers.Concatenate(axis=1, name="merger")([x_ge, x_im])

#     hidden_layer_width = 1000
#     merged_model = tf.keras.layers.Dense(hidden_layer_width, activation=tf.nn.relu,
#                                          name="hidden_1")(merged_model)

#     # Add the softmax prediction layer
#     activation = 'linear' if model_type == 'linear' else 'softmax'
#     final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
#     softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name='ctype')(final_dense_layer)

#     # Assemble final model
#     model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)
#     return model
