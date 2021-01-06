"""
1. Tiling
   Generate tiles from slides (WSI files).

2. Build df that merges response, rna-seq, and desciptors.

3. Create tfrecords.
    For each slide, load the tiles, expression, and meta,
    and store each example into a tfrec file (signle tfrec per slide).
"""

import os
import sys
from pathlib import Path
import glob

import numpy as np
import pandas as pd

# To get around renderer issue on OSX going from Matplotlib image to NumPy image.
import matplotlib
matplotlib.use('Agg')

# import deephistopath.wsi
# from deephistopath.wsi.filter import *
# from deephistopath.wsi.slide import *
# from deephistopath.wsi.tiles import *
# from deephistopath.wsi.util import *

from deephistopath.wsi import filter
from deephistopath.wsi import slide
from deephistopath.wsi import tiles
from deephistopath.wsi import util

fpath = Path(__file__).resolve().parent


def get_slide_num_from_path(slide_filepath):
    return int(os.path.basename(slide_filepath).split('.svs')[0])  ## ap


import ipdb; ipdb.set_trace(context=11)

# Path to tiles
tiles_dirpath = fpath/'../data/tiles_png'
tiles_path_list = glob.glob(os.path.join(slides_dirpath, '*-tile_*.png'))

