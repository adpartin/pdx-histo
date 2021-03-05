# https://github.com/jkjung-avt/keras_imagenet/blob/master/config.py
# https://github.com/jkjung-avt/keras_imagenet/blob/master/utils/dataset.py

import os
import sys
import types
from pathlib import Path

fdir = Path(__file__).resolve().parent

cfg = types.SimpleNamespace()

cfg.MAIN_APPDIR = (fdir/'../apps').resolve()
cfg.DATADIR = (fdir/'../data').resolve()

cfg.TILES_DIR = cfg.DATADIR/'tiles_png'

cfg.ANNOTATIONS_FILENAME = 'annotations.csv'
cfg.SF_ANNOTATIONS_FILENAME = 'annotations_slideflow.csv'

cfg.TFR_DIR = (cfg.DATADIR/'tfrecords').resolve()
# cfg.SF_TFR_DIR = (fdir/'../../slideflow-proj/PDX_FIXED').resolve()
cfg.SF_TFR_DIR = (fdir/'../../slideflow-proj/PDX_FIXED_updated').resolve()

# cfg.MAIN_APPDIR = MAIN_APPDIR
# cfg.DATADIR = DATADIR
# cfg.ANNOTATIONS_FILENAME = ANNOTATIONS_FILENAME
# cfg.SF_ANNOTATIONS_FILENAME = SF_ANNOTATIONS_FILENAME
# cfg.TILES_DIR = TILES_DIR
# cfg.SF_TFR_DIR = SF_TFR_DIR

# Meta file names
# cfg.CROSSREF_FNAME = 'ImageID_PDMRID_CrossRef.csv'
cfg.CROSSREF_FNAME = '_ImageID_PDMRID_CrossRef.xlsx'
# cfg.PDX_META_FNAME = 'PDX_Meta_Information.csv'
cfg.PDX_META_FNAME = 'PDX_Meta_Information.xlsx'
cfg.SLIDES_META_FNAME = 'meta_from_wsi_slides.csv'

# Slides that were identified as bad with either bad staining or poor quality. 
# These notes were provided by Pearson's group in PDX_FIXED/slide_problems.txt
cfg.BAD_SLIDES = [45983, 83742, 83743,  # poor quality
                  22232, 21836, 20729,  # staining off
                  ]
# 10415, 13582, 9113,   # "no corresponding slide annotations" --> output from slideflow
