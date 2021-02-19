# https://github.com/jkjung-avt/keras_imagenet/blob/master/config.py
# https://github.com/jkjung-avt/keras_imagenet/blob/master/utils/dataset.py

import os
import sys
import types
from pathlib import Path

fdir = Path(__file__).resolve().parent

MAIN_APPDIR = fdir/'../apps'
DATADIR = fdir/'../data'
ANNOTATIONS_FILENAME = 'annotations.csv'
TILES_DIR = DATADIR/'tiles_png'
TFR_DIR = DATADIR/'tfrecords'

cfg = types.SimpleNamespace()
cfg.MAIN_APPDIR = MAIN_APPDIR
cfg.DATADIR = DATADIR
cfg.ANNOTATIONS_FILENAME = ANNOTATIONS_FILENAME
cfg.TILES_DIR = TILES_DIR
cfg.TFR_DIR = TFR_DIR
