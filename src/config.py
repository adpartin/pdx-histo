# https://github.com/jkjung-avt/keras_imagenet/blob/master/config.py
# https://github.com/jkjung-avt/keras_imagenet/blob/master/utils/dataset.py

import os
import sys
import types
from pathlib import Path
import numpy as np
import tensorflow as tf

fdir = Path(__file__).resolve().parent

cfg = types.SimpleNamespace()

cfg.MAIN_APPDIR = (fdir/'../apps').resolve()
cfg.DATADIR = (fdir/'../data').resolve()
cfg.SLIDES_DIR = (cfg.DATADIR/'doe-globus-pdx-data').resolve()
# cfg.TILES_DIR = cfg.DATADIR/'tiles_png'  # old pipeline

# Annotaions
cfg.ANNOTATIONS_FILENAME = 'annotations.csv'
cfg.SF_ANNOTATIONS_FILENAME = 'annotations_slideflow.csv'

# TFRecords
cfg.TFR_DIR = (cfg.DATADIR/'tfrecords').resolve()
cfg.SF_TFR_DIR = (fdir/'../../slideflow-proj/PDX_FIXED').resolve()
# cfg.SF_TFR_DIR = (fdir/'../../slideflow-proj/PDX_FIXED_updated').resolve()
cfg.SF_TFR_DIR_RNA = (fdir/'../../slideflow-proj/PDX_FIXED_RNA').resolve()
cfg.SF_TFR_DIR_RSP = (fdir/'../../slideflow-proj/PDX_FIXED_RSP').resolve()

# cfg.MAIN_APPDIR = MAIN_APPDIR
# cfg.DATADIR = DATADIR
# cfg.ANNOTATIONS_FILENAME = ANNOTATIONS_FILENAME
# cfg.SF_ANNOTATIONS_FILENAME = SF_ANNOTATIONS_FILENAME
# cfg.TILES_DIR = TILES_DIR
# cfg.SF_TFR_DIR = SF_TFR_DIR

# Meta file names
# cfg.CROSSREF_FNAME = 'ImageID_PDMRID_CrossRef.csv'
cfg.CROSSREF_FNAME = '_ImageID_PDMRID_CrossRef.xlsx'
# cfg.PDX_META_FNAME = 'PDX_Meta_Information.xlsx'
cfg.PDX_META_FNAME = 'PDX_Meta_Information2.csv'
cfg.SLIDES_META_FNAME = 'meta_from_wsi_slides.csv'

cfg.RSP_DPATH = cfg.DATADIR/'studies/pdm/ncipdm_drug_response'
# cfg.RNA_DPATH = cfg.DATADIR/'combined_rnaseq_data_lincs1000'
cfg.RNA_DPATH = cfg.DATADIR/'combined_rnaseq_data_lincs1000_combat'
cfg.DD_DPATH = cfg.DATADIR/'dd.mordred.with.nans'

cfg.METAPATH = cfg.DATADIR/'meta'
# cfg.META_DPATH = cfg.METAPATH/'meta_merged.csv'

# Slides that were identified as bad with either bad staining or poor quality. 
# These notes were provided by Pearson's group in PDX_FIXED/slide_problems.txt
cfg.BAD_SLIDES = [45983, 83742, 83743,  # poor quality
                  20729, 21836, 22232,  # staining off
                  ]
cfg.BAD_SLIDES = [str(s) for s in cfg.BAD_SLIDES]
# 10415, 13582, 9113,   # "no corresponding slide annotations" --> output from slideflow

# Data types
cfg.GE_DTYPE = np.float32
cfg.DD_DTYPE = np.float32
cfg.FEA_DTYPE = np.float32

# App globals  # TODO: this should be set per app??
app_cfg = types.SimpleNamespace()
cfg.ID_NAME = 'smp'
cfg.TILE_PX = 299
cfg.TILE_UM = 302
cfg.IMAGE_SIZE = cfg.TILE_PX 

# TF names
tf_cfg = types.SimpleNamespace()
tf_cfg.TILE_DATA_NAME = "tile_data"
tf_cfg.GE_DATA_NAME = "ge_data"
tf_cfg.DD_DATA_NAME = "dd_data"
tf_cfg.RESPONSE_NAME = 'Response'
