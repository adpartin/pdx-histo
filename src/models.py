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


def keras_callbacks(outdir, monitor="val_loss", patience=5):
    """ ... """
    callbacks = []

    csv_logger = CSVLogger(outdir/"training.log")
    callbacks.append(csv_logger)

    checkpointer = ModelCheckpoint(str(outdir/"model_best_at_{epoch}.ckpt"),
                                   monitor=monitor,
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=True,
                                   save_freq="epoch")
    callbacks.append(checkpointer)

    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=0.5,
                                  patience=patience,
                                  verbose=1,
                                  mode="auto",
                                  min_delta=0.0001,
                                  cooldown=0,
                                  min_lr=0)
    callbacks.append(reduce_lr)

    early_stop = EarlyStopping(monitor=monitor,
                               patience=20,
                               mode="auto",
                               restore_best_weights=True,
                               verbose=1)
    callbacks.append(early_stop)

    return callbacks


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
                             dense1_ge=512, dense1_dd1=256, dense1_dd2=256, dense1_top=500,
                             top_hidden_dropout1=0.1,
                             # NUM_CLASSES=None,
                             output_bias=None):
    """ Doesn't use image data. """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model_inputs = []
    merge_inputs = []

    if use_ge:
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(dense1_ge, activation=tf.nn.relu, name="dense1_ge")(ge_input_tensor)
        x_ge = BatchNormalization(name="ge_batchnorm")(x_ge)
        # x_ge = Dropout(0.4)(x_ge)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)
        del ge_input_tensor, x_ge

    if use_dd1:
        dd1_input_tensor = tf.keras.Input(shape=dd_shape, name="dd1_data")
        x_dd1 = Dense(dense1_dd1, activation=tf.nn.relu, name="dense1_dd1")(dd1_input_tensor)
        x_dd1 = BatchNormalization(name="dd1_batchnorm")(x_dd1)
        # x_dd1 = Dropout(0.4)(x_dd1)
        model_inputs.append(dd1_input_tensor)
        merge_inputs.append(x_dd1)
        del dd1_input_tensor, x_dd1

    if use_dd2:
        dd2_input_tensor = tf.keras.Input(shape=dd_shape, name="dd2_data")
        x_dd2 = Dense(dense1_dd2, activation=tf.nn.relu, name="dense1_dd2")(dd2_input_tensor)
        x_dd2 = BatchNormalization(name="dd2_batchnorm")(x_dd2)
        # x_dd2 = Dropout(0.4)(x_dd2)
        model_inputs.append(dd2_input_tensor)
        merge_inputs.append(x_dd2)
        del dd2_input_tensor, x_dd2

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    merged_model = tf.keras.layers.Dense(dense1_top, activation=tf.nn.relu,
                                         name="dense1_top", kernel_regularizer=None)(merged_model)
    merged_model = BatchNormalization()(merged_model)
    if top_hidden_dropout1 > 0:
        merged_model = Dropout(top_hidden_dropout1)(merged_model)

    # Add the softmax prediction layer
    # activation = "linear" if model_type == "linear" else "softmax"
    # final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
    # softmax_output = tf.keras.layers.Activation(activation, dtype="float32", name="Response")(final_dense_layer)

    softmax_output = tf.keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=output_bias, name="Response")(merged_model)

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
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(dense1_ge, activation=tf.nn.relu, name="dense1_ge")(ge_input_tensor)
        x_ge = BatchNormalization(name="ge_batchnorm")(x_ge)
        # x_ge = Dropout(0.4)(x_ge)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)
        del ge_input_tensor, x_ge

    if use_dd1:
        dd1_input_tensor = tf.keras.Input(shape=dd_shape, name="dd1_data")
        x_dd1 = Dense(dense1_dd1, activation=tf.nn.relu, name="dense1_dd1")(dd1_input_tensor)
        x_dd1 = BatchNormalization(name="dd1_batchnorm")(x_dd1)
        # x_dd1 = Dropout(0.4)(x_dd1)
        model_inputs.append(dd1_input_tensor)
        merge_inputs.append(x_dd1)
        del dd1_input_tensor, x_dd1

    if use_dd2:
        dd2_input_tensor = tf.keras.Input(shape=dd_shape, name="dd2_data")
        x_dd2 = Dense(dense1_dd2, activation=tf.nn.relu, name="dense1_dd2")(dd2_input_tensor)
        x_dd2 = BatchNormalization(name="dd2_batchnorm")(x_dd2)
        # x_dd2 = Dropout(0.4)(x_dd2)
        model_inputs.append(dd2_input_tensor)
        merge_inputs.append(x_dd2)
        del dd2_input_tensor, x_dd2

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    merged_model = tf.keras.layers.Dense(dense1_top, activation=tf.nn.relu,
                                         name="dense1_top", kernel_regularizer=None)(merged_model)
    merged_model = BatchNormalization(name="top_batchnorm")(merged_model)
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

    # # Compile
    # # Note! When I put metrics in model.py, it immediately occupies a lot of the GPU memory!
    # metrics = [
    #       # keras.metrics.TrueNegatives(name="tn"),
    #       keras.metrics.FalsePositives(name="fp"),
    #       # keras.metrics.FalseNegatives(name="fn"),
    #       keras.metrics.TruePositives(name="tp"),
    #       # keras.metrics.AUC(name="auc"),
    # ]

    # model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

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
