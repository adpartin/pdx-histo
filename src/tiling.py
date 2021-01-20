"""
1. Tiling
Generate tiles from slides (WSI files).

Update the default (hard-coded) tiling parameters in deephistopath. For example:
    - SCALE_FACTOR (32 -> 16)
    - ROW_TILE_SIZE (1024 -> 300)
    - COL_TILE_SIZE (1024 -> 300)
    - NUM_TOP_TILES (50 -> 40)

Notes:
    - some slides contain very few tissue (8657). make sure to keep tiles with enough tissue.
    - some tiles contain regions with very different texture (34796).

Specify the path to the tiles.
Specify the path to gene expression.
Specify the path to meta.

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
    return int(os.path.basename(slide_filepath).split('.svs')[0])


# import ipdb; ipdb.set_trace(context=11)

slides_dirpath = fpath/'../data/training_slides'
slides_path_list = glob.glob(os.path.join(slides_dirpath, '*.svs'))
image_num_list = [get_slide_num_from_path(slide_filepath) for slide_filepath in slides_path_list]
print('Total svs slides', len(image_num_list))


# ================================================
# Tiling (generate tiles from slides)
# ================================================

#
# Part 1 (whole-slide image preprocessing in python)
#

# import ipdb; ipdb.set_trace(context=11)
slide.singleprocess_training_slides_to_images(slides_dirpath)
# slide.multiprocess_training_slides_to_images(slides_path)  # didn't try

#
# Part 3 (morphology operators)
#

# import ipdb; ipdb.set_trace(context=11)
filter.singleprocess_apply_filters_to_images(image_num_list=image_num_list)
# filter.multiprocess_apply_filters_to_images(image_num_list=image_num_list)

#
# Part 4 (top tile retrieval)
#

# import ipdb; ipdb.set_trace(context=11)
tiles.singleprocess_filtered_images_to_tiles(display=False,
                                             save_summary=True,
                                             save_data=True,
                                             save_top_tiles=True,
                                             html=True,
                                             image_num_list=image_num_list)

# tiles.multiprocess_filtered_images_to_tiles(display=False,
#                                             save_summary=True,
#                                             save_data=True,
#                                             save_top_tiles=True,
#                                             html=True,
#                                             image_num_list=image_num_list)

print('\nDone.')
