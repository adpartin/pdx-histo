import os
from pathlib import Path
from glob import glob
from pprint import pprint
import numpy as np
import pandas as pd

# WSI utils
import histolab
from histolab.slide import Slide
from histolab.tiler import RandomTiler, GridTiler, ScoreTiler
from histolab.scorer import NucleiScorer
import openslide
# from openslide import OpenSlide

import matplotlib
import matplotlib.pyplot as plt
import PIL

dirpath = Path(__file__).resolve().parent
print(dirpath)

# Path
datapath = dirpath/'../data'
imgpath = datapath/'doe-globus-pdx-data'  # path to raw WSI data
metapath = datapath/'meta'

# outpath = datapath/'processed'        # path to save processed images
tmp_outpath = datapath/'tmp_out'
os.makedirs(tmp_outpath, exist_ok=True)

# Glob images
wsi_format = 'svs'
files = sorted(imgpath.glob(f'*.{wsi_format}'))
print('Total {} files: {}'.format(wsi_format.upper(), len(files)))


# -----------------------------------
# Explore the Slide object
# -----------------------------------
"""
- biggest_tissue_box_mask:   YES
- dimensions:                YES
- extract_tile:              explore
- level_dimensions:          YES
- levels:                    YES
- locate_biggest_tissue_box: explore
- name:                      YES
- processed_path:            YES
- resampled_array:           YES
- save_scaled_image:         YES
- save_thumbnail:            YES
- scaled_image_path:         YES
- show:                      no
- thumbnail_path:            YES
"""
# Slide instance
# path: path to WSI file
# processed_path: path to save thumbnails and scaled images
fname = files[0]
img_inpath = str(fname)
img_outpath = os.path.join(str(datapath), 'processed', fname.with_suffix('').name)
pdx_slide = Slide(path=img_inpath, processed_path=tmp_outpath)

# Methods and properties of slide object
# pprint(dir(pdx_slide))

# Slide properties
print(f"Type:                  {type(pdx_slide)}")
print(f"Slide name:            {pdx_slide.name}")
print(f"Levels:                {pdx_slide.levels}")
print(f"Dimensions at level 0: {pdx_slide.dimensions}")
print(f"Dimensions at level 1: {pdx_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {pdx_slide.level_dimensions(level=2)}")


# Openslide properties (_wsi is openslide object)
print(f"type(slide._wsi):            {type(pdx_slide._wsi)}")
print(f"type(slide._wsi.properties): {type(pdx_slide._wsi.properties)}")
print(f"Total properties:            {len(pdx_slide._wsi.properties)}")
print(f"AppMag:                      {pdx_slide._wsi.properties['aperio.AppMag']}")

# Methods and properties of openslide object
# pprint(pdx_slide._wsi.properties._keys())  # all properties

mag = int(pdx_slide._wsi.properties['aperio.AppMag'])
print(f"Level count:       {pdx_slide._wsi.level_count}")
print(f"Level downsamples: {pdx_slide._wsi.level_downsamples}")
print(f"Level dimensions:  {pdx_slide._wsi.level_dimensions}")


# Thumbnail
print('thumbnail_size:         ', pdx_slide._thumbnail_size)  # thumbnail size proportionally to the slide dimensions
print('biggest_tissue_box_mask:', pdx_slide.biggest_tissue_box_mask.shape)  # thumbnail binary mask of the box containing the max tissue area
print('thumbnail_path:         ', pdx_slide.thumbnail_path)
pdx_slide.save_thumbnail()

# Inside the save_thumbnail
pil_img = pdx_slide._wsi.get_thumbnail(pdx_slide._thumbnail_size)
print(type(pil_img))
print(pil_img.size)
# pil_img.show() # why this doesn't work??
# plt.imshow(pil_img); plt.axis('off');
pil_img.save(tmp_outpath/'pil_img.png', format='png')

# Apply mask to image
pil_img_boxed = histolab.util.apply_mask_image(pil_img, pdx_slide.biggest_tissue_box_mask)
pil_img_boxed.save(tmp_outpath/'pil_img_boxed.png', format='png')


# Scale and save image
scale_factor = 16
np_img_scaled = pdx_slide.resampled_array(scale_factor=scale_factor)  # scale and return ndarray
print(type(np_img_scaled))
print(np_img_scaled.shape)
# out image file path: {name}-{scale_factor}x-{large_w}x{large_h}-{new_w}x{new_h}.{IMG_EXT}
pdx_slide.save_scaled_image(scale_factor=scale_factor)             # scale and save into file

# Show img from array
# plt.imshow(img_scaled); plt.axis('off');
fig, ax = plt.subplots(figsize=(5,5)); ax.imshow(np_img_scaled); ax.axis('off');
# PIL.Image.fromarray(np_img_scaled)


# -----------------------------------
# Metadata
# -----------------------------------
# Load meta
if (metapath/'meta_from_wsi_images.csv').exists():
    meta_df = pd.read_csv(metapath/'meta_from_wsi_images.csv')


# -----------------------------------
# Tiling
# -----------------------------------

"""
https://histolab.readthedocs.io/en/latest/api/tiler.html
Extract tiles arranged in a grid and save them to disk, following this filename pattern:
{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}
"""
tile_sz = 300
level = 0
# check_tissue = True
check_tissue = False

grid_tiler = GridTiler(
    tile_size=(tile_sz, tile_sz), # (width, height) of the extracted tiles
    level=level,                  # Level from which extract the tiles. Default is 0.
    check_tissue=check_tissue,    # Whether to check if the tile has enough tissue to be saved. Default is True.
    pixel_overlap=0,              # Number of overlapping pixels (for both height and width) between two adjacent tiles.
    prefix='',                    # Prefix to be added to the tile filename. Default is an empty string.
    suffix=".png" # default
)

# Find the smallest slide
idx = meta_df['openslide.level[0].height'].isin( [meta_df['openslide.level[0].height'].min()] )
img_name = meta_df.loc[idx, 'aperio.ImageID'].values[0]
fname = imgpath/f'{img_name}.svs'

# Slide instance
# path: path to WSI file
# processed_path: path to save thumbnails and scaled images
img_inpath = str(fname)
sfx = 'tissue' if check_tissue else 'all'
img_outpath = os.path.join(str(datapath), 'tiles', sfx, f'{tile_sz}px', fname.with_suffix('').name)
# os.makedirs(grid_tiles_path, exist_ok=True)
pdx_slide = Slide(path=img_inpath, processed_path=img_outpath)
print(pdx_slide.dimensions)

# Extract
import ipdb; ipdb.set_trace(context=11)
grid_tiler.extract(pdx_slide);
