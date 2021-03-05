"""
Generate tiles from histology slides (WSI files).

Updated the default (hard-coded) tiling parameters in deephistopath.
For example:
    * SCALE_FACTOR (32 -> 16)      ; in slide.py ; done
    * ROW_TILE_SIZE (1024 -> 300)  ; in tiles.py ; done
    * COL_TILE_SIZE (1024 -> 300)  ; in tiles.py ; done
    * NUM_TOP_TILES (50 -> 40)     ; in tiles.py ; done

Note:
    * some slides contain very small amount tissue (8657); make sure to keep tiles with enough tissue
    * some tiles contain regions with very different texture (34796)

Specify the path to tiles.
Specify the path to gene expression.
Specify the path to meta.

For each slide, load the tiles, expression, and meta,
and store each example into a tfrec file (signle tfrec per slide).
"""

import os
import sys
from pathlib import Path
import glob
from pprint import pprint
import pandas as pd
import numpy as np

# To get around renderer issue on OSX going from Matplotlib image to NumPy image.
import matplotlib
matplotlib.use('Agg')

from deephistopath.wsi import filter
from deephistopath.wsi import slide
from deephistopath.wsi import tiles
from deephistopath.wsi import util

from config import cfg

fdir = Path(__file__).resolve().parent

# DATADIR = fdir/'../data'
DATADIR = cfg.DATADIR


def get_slide_num_from_path(slide_filepath):
    return int(os.path.basename(slide_filepath).split('.svs')[0])

# slides_dirpath = cfg.DATADIR/'training_slides'
slides_dirpath = cfg.DATADIR/'doe-globus-pdx-data'
slides_path_list = glob.glob(os.path.join(slides_dirpath, '*.svs'))
image_num_list = [get_slide_num_from_path(slide_filepath) for slide_filepath in slides_path_list]
print('Total svs slides', len(image_num_list))

# n_slides = 2
n_slides = None
exclude_slides = []
# image_num_list = [9970, 9926]

proc_slides = True
filter_slides = True
gen_tiles = True


# ================================================
# Tiling (generate tiles from histology slides)
# ================================================

# Timer for each processing step
timer = {}

#
# Part 1 (whole-slide image preprocessing in python)
#
"""
Pass dirpath that contains the slides to slide.singleprocess_training_slides_to_images().
Glob all svs files in the dirpath and iter through them.
For each file:
    * extract slide number from fname (used as id)
    * scale the slide, assign into pil var, and obtain the scaled dims (new_w, new_h)
    * save scaled image to:
          data/training_png/{slide_number}-{scale_factor}-{large_w}x{large_h}-{new_w}x{new_h}.png
    * save thumbnail to:
          data/training_thumbnail_jpg/{slide_number}-{scale_factor}-{large_w}x{large_h}-{new_w}x{new_h}.png
"""

# import ipdb; ipdb.set_trace(context=11)

t = util.Time()

if proc_slides:
    slide.singleprocess_training_slides_to_images(slides_dirpath, n_slides=n_slides)
    # slide.multiprocess_training_slides_to_images(slides_path)  # didn't try

timer['process slides'] = t.elapsed

#
# Part 3 (morphology operators)
#
"""
Pass list of slide numbers to filter.singleprocess_apply_filters_to_images().
For each image:
    * load the (downsampled) image into pil from:
          data/training_png/{slide_number}-{scale_factor}-{large_w}x{large_h}-{new_w}x{new_h}.png
    * convert pil to np
    * create mask with filter_green_channel
    * create mask with filter_grays
    * create mask with filter_red_pen
    * create mask with filter_green_pen
    * create mask with filter_blue_pen
    * combine filter masks and apply to the (downsampled) image
    * apply filter_remove_small_objects
    * save all intermediate stages of the filtered image to:
      data/filter_png/{slide_number}-[stage-id]-[filter-type].png
      the final filtered image is saved to:
          data/filter_png/{slide_number}-{scale_factor}-{large_w}x{large_h}-{new_w}x{new_h}-filtered.png
    * save all filtered images above also as thumbnails to
          data/filter_thumbnail_jpg/...
"""

t = util.Time()

# import ipdb; ipdb.set_trace(context=11)

if filter_slides:
    filter.singleprocess_apply_filters_to_images(image_num_list=image_num_list)
    # filter.multiprocess_apply_filters_to_images(image_num_list=image_num_list)

timer['filter slides'] = t.elapsed

#
# Part 4 (top tile retrieval)
#
"""
Pass list of slide numbers to tiles.singleprocess_filtered_images_to_tiles().
For each image:
    ... = image_list_to_tiles( image_num_list, ... )
    * load the filtered (downsampled) image into np from:
          data/filter_png/{slide_number}-{scale_factor}-{large_w}x{large_h}-{new_w}x{new_h}-filtered.png
    * score tiles with --> tile_sum = score_tiles( slide_num, np_img ):
        * obtain tile dimensions in the downsampled image space:
            row_tile_size = round(ROW_TILE_SIZE / slide.SCALE_FACTOR)  # 300 / 16 = 19
            col_tile_size = round(COL_TILE_SIZE / slide.SCALE_FACTOR)  # 300 / 16 = 19
        * create an object of TileSummary that will contain all the tiles:
            tile_sum = TileSummary( slide_num, ... )
        * obtain the indices of all tiles in the downsampled image space:
            tile_indices = get_tile_indices( rows=h, cols=w, row_tile_size=row_tile_size, col_tile_size=col_tile_size )
            returns a list of tuples (start row, end row, start column, end column, row number, column number)
        * get a tile from the downsampled numpy image:
            np_tile = np_img[r_s:r_e, c_s:c_e]  # get a tile
        * map a downsampled pixel width and height to the corresponding pixel of the original whole-slide image:
            (WHY DO WE DO THAT??)
            o_c_s, o_r_s = slide.small_to_large_mapping( (c_s, r_s), (o_w, o_h) )
            o_c_e, o_r_e = slide.small_to_large_mapping( (c_e, r_e), (o_w, o_h) )
        * pixel adjustment in case tile dimension is too large (??)
            if (o_c_e - o_c_s) > COL_TILE_SIZE:
                o_c_e -= 1
            if (o_r_e - o_r_s) > ROW_TILE_SIZE:
                o_r_e -= 1
        * score the tile
            ... = score_tile( np_tile, t_p, slide_num, r, c )
        * for each np tile image create an object of class Tile and store in TileSummary:
            tile = Tile( tile_sum, slide_num, np_scaled_tile, ... )
            tile_sum.tiles.append( tile )
    * save tiles info in csv for each slide and save in data/tile_data:
        save_tile_data( tile_sum )
    * using the filtered downsampled image, generate summary images/thumbnails
      showing a 'heatmap' representation of the tissue segmentation of all tiles:
          generate_tile_summaries( tile_sum, np_img, display=display, save_summary=save_summary )
          data/tile_summary_on_original_png
          data/tile_summary_png
    * same as previous command, but now generate heatmap for the top tiles:
          generate_top_tile_summaries(tile_sum, np_img, display=display, save_summary=save_summary)
          data/top_tile_summary_on_original_png
          data/top_tile_summary_png
    * finally, iter over the top tiles and save each tile to png:
        tile.save_tile()
        Note! the final tile that is saved in png is NOT extracted from the downsampled image,
        but rather from the original WSI inside:
        tile_to_pil_tile( tile )
"""

t = util.Time()

# import ipdb; ipdb.set_trace(context=11)

if gen_tiles:
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

timer['generate tiles'] = t.elapsed


# ================================================
# Final things
# ================================================

# import ipdb; ipdb.set_trace(context=11)

# Keep only tiles_png in ../data/ and copy the rest to ../data/tiles_other
import shutil
items = ['filter_png', 'filters.html', 'filter_thumbnail_jpg',
         'tile_data', 'tiles.html',
         'tile_summary_on_original_png', 'tile_summary_on_original_thumbnail_jpg',
         'tile_summary_png', 'tile_summary_thumbnail_jpg',
         'top_tile_summary_on_original_png', 'top_tile_summary_on_original_thumbnail_jpg',
         'top_tile_summary_png', 'top_tile_summary_thumbnail_jpg',
         'training_thumbnail_jpg']
src_path = cfg.DATADIR
dst_path = cfg.DATADIR/'tiles_other'
os.makedirs(dst_path, exist_ok=True)
for item in items:
    if (src_path/item).exists():
        shutil.move(src_path/item, dst_path/item)

file_patterns = ['filters*.html', 'tiles*.html']
for fp in file_patterns:
    gfiles = list(src_path.glob(fp))
    for gf in gfiles:
        if gf.exists():
            shutil.move(gf, dst_path/gf.name)

print('\nProcessing time.')
for k, v in timer:
    print(f'{k}: {v}')

print('\nDone.')
