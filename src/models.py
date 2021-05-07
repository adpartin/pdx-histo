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
# from config import cfg
from src.config import cfg
from src.sf_utils import bold, green, blue, yellow, cyan, red

_ModelDict = {
    "Xception": tf.keras.applications.Xception,
    "ResNet50": tf.keras.applications.ResNet50,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "ResNet101": tf.keras.applications.ResNet101,
    "ResNet101V2": tf.keras.applications.ResNet101V2,
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

    checkpointer = ModelCheckpoint(str(outdir/"model_{epoch:02d}-{val_loss:.3f}.ckpt"),
                                   monitor=monitor,
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=True,
                                   save_freq="epoch")
    callbacks.append(checkpointer)

    reduce_lr = ReduceLROnPlateau(monitor=monitor,
                                  factor=0.5,
                                  patience=10,
                                  verbose=1,
                                  mode="auto",
                                  min_delta=0.0001,
                                  cooldown=0,
                                  min_lr=0)
    callbacks.append(reduce_lr)

    early_stop = EarlyStopping(monitor=monitor,
                               patience=patience,
                               mode="auto",
                               restore_best_weights=True,
                               verbose=1)
    callbacks.append(early_stop)

    return callbacks


def load_best_model(models_dir):
    """ Load the best checkpointed model where best is defined as a model with
    the lowest val_loss. The names of checkpointed models follow the same naming
    convention that contains the val_loss: model_{epoch:02d}-{val_loss:.3f}.ckpt
    """
    model_paths = sorted(models_dir.glob("model*.ckpt"))
    values = np.array([float(p.name.split(".ckpt")[0].split("-")[1]) for p in model_paths])
    # best_value = min(values)
    model_path = model_paths[np.argmin(values)]
    model = tf.keras.models.load_model(model_path)
    return model


def build_model_rsp_baseline(use_ge=True, use_dd1=True, use_dd2=True,
                             ge_shape=None, dd_shape=None, model_type="categorical",
                             dense1_ge=512, dense1_dd1=256, dense1_dd2=256, dense1_top=500,
                             dropout1_top=0.1,
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
    if dropout1_top > 0:
        merged_model = Dropout(dropout1_top)(merged_model)

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
                    dense1_img=1024, dense2_img=512,
                    dense1_ge=500,
                    dense1_dd1=250, dense1_dd2=250,
                    dense1_top=1000,
                    dropout1_top=0.1,
                    output_bias=None,
                    # model_type="categorical",
                    base_image_model="Xception",
                    pooling="max",
                    pretrain="imagenet",
                    loss=losses.BinaryCrossentropy(),
                    optimizer="SGD",
                    learning_rate=0.0001):
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
        base_img_model = _ModelDict[base_image_model](
            include_top=False,
            weights=pretrain,
            input_shape=None,
            input_tensor=None,
            pooling=pooling)

        # import ipdb; ipdb.set_trace()
        # print(len(base_img_model.trainable_weights))
        # print(len(base_img_model.non_trainable_weights))
        # print(len(base_img_model.layers))
        # base_img_model.trainable = False
        # x_tile = keras.layers.GlobalAveragePooling2D()(tile_input_tensor)

        x_tile = base_img_model(tile_input_tensor)
        model_inputs.append(tile_input_tensor)

        if dense1_img > 0:
            x_tile = Dense(dense1_img, activation=tf.nn.relu, name="dense1_img")(x_tile)
            # x_tile = BatchNormalization(name="batchnorm_im")(x_tile)
        if dense2_img > 0:
            x_tile = Dense(dense2_img, activation=tf.nn.relu, name="dense2_img")(x_tile)
        if (dense1_img > 0) or (dense2_img > 0):
            x_tile = BatchNormalization(name="batchnorm_im")(x_tile)
        merge_inputs.append(x_tile)
        del tile_input_tensor, x_tile

    if use_ge:
        ge_input_tensor = tf.keras.Input(shape=ge_shape, name="ge_data")
        x_ge = Dense(dense1_ge, activation=tf.nn.relu, name="dense1_ge")(ge_input_tensor)
        x_ge = BatchNormalization(name="batchnorm_ge")(x_ge)
        # x_ge = Dropout(0.4)(x_ge)
        model_inputs.append(ge_input_tensor)
        merge_inputs.append(x_ge)
        del ge_input_tensor, x_ge

    if use_dd1:
        dd1_input_tensor = tf.keras.Input(shape=dd_shape, name="dd1_data")
        x_dd1 = Dense(dense1_dd1, activation=tf.nn.relu, name="dense1_dd1")(dd1_input_tensor)
        x_dd1 = BatchNormalization(name="batchnorm_dd1")(x_dd1)
        # x_dd1 = Dropout(0.4)(x_dd1)
        model_inputs.append(dd1_input_tensor)
        merge_inputs.append(x_dd1)
        del dd1_input_tensor, x_dd1

    if use_dd2:
        dd2_input_tensor = tf.keras.Input(shape=dd_shape, name="dd2_data")
        x_dd2 = Dense(dense1_dd2, activation=tf.nn.relu, name="dense1_dd2")(dd2_input_tensor)
        x_dd2 = BatchNormalization(name="batchnorm_dd2")(x_dd2)
        # x_dd2 = Dropout(0.4)(x_dd2)
        model_inputs.append(dd2_input_tensor)
        merge_inputs.append(x_dd2)
        del dd2_input_tensor, x_dd2

    # Merge towers
    merged_model = layers.Concatenate(axis=1, name="merger")(merge_inputs)

    merged_model = tf.keras.layers.Dense(dense1_top, activation=tf.nn.relu,
                                         name="dense1_top", kernel_regularizer=None)(merged_model)
    merged_model = BatchNormalization(name="batchnorm_top")(merged_model)
    if dropout1_top > 0:
        merged_model = Dropout(dropout1_top)(merged_model)

    # Add the softmax prediction layer
    # activation = 'linear' if model_type == 'linear' else 'softmax'
    # final_dense_layer = tf.keras.layers.Dense(NUM_CLASSES, name="prelogits")(merged_model)
    # softmax_output = tf.keras.layers.Activation(activation, dtype='float32', name="Response")(final_dense_layer)

    softmax_output = tf.keras.layers.Dense(
        1, activation="sigmoid", bias_initializer=output_bias, name="Response")(merged_model)

    # Assemble final model
    model = tf.keras.Model(inputs=model_inputs, outputs=softmax_output)

    metrics = [
          # keras.metrics.FalsePositives(name="fp"),
          # keras.metrics.TruePositives(name="tp"),
          keras.metrics.AUC(name="roc-auc", curve="ROC"),
          keras.metrics.AUC(name="pr-auc", curve="PR"),
    ]
    if optimizer == "SGD":
        optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "Adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


# def build_model_rna(pooling='max', pretrain='imagenet'):
#     # Image layers
#     image_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)
#     tile_input_tensor = tf.keras.Input(shape=image_shape, name="tile_image")
#     base_img_model = tf.keras.applications.Xception(
#         weights=pretrain, pooling=pooling, include_top=False,
#         input_shape=None, input_tensor=None)
#     x_tile = base_img_model(tile_input_tensor)

#     # RNA layers
#     ge_input_tensor = tf.keras.Input(shape=(976,), name="ge_data")
#     x_ge = Dense(512, activation=tf.nn.relu)(ge_input_tensor)

#     model_inputs = [tile_input_tensor, ge_input_tensor]

#     # Merge towers
#     merged_model = layers.Concatenate(axis=1, name="merger")([x_ge, x_tile])

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
