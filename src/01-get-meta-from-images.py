import os
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob
from pprint import pprint

# WSI utils
from histolab.slide import Slide
# from histolab.tiler import RandomTiler, GridTiler, ScoreTiler
# from histolab.scorer import NucleiScorer
import openslide
# from openslide import OpenSlide

dirpath = Path(__file__).resolve().parent
print(dirpath)

# Path
datapath = dirpath/'../data'
imgpath = datapath/'doe-globus-pdx-data'  # path to raw WSI data
metapath = datapath/'meta'

# Glob images
files = sorted(imgpath.glob('*.svs'))
print(f'Total files: {len(files)}')
print(files[0].with_suffix('').name)

# -----------------------------------
# Explore the Slide object
# -----------------------------------
# Slide instance
# path: path to WSI file
# processed_path: path to save thumbnails and scaled images
fname = files[0]
img_inpath = str(fname)
img_outpath = os.path.join(str(datapath), 'processed', fname.with_suffix('').name)
pdx_slide = Slide(path=img_inpath, processed_path=img_outpath)

# Slide properties
print(f"Type:                  {type(pdx_slide)}")
print(f"Slide name:            {pdx_slide.name}")
print(f"Levels:                {pdx_slide.levels}")
print(f"Dimensions at level 0: {pdx_slide.dimensions}")
print(f"Dimensions at level 1: {pdx_slide.level_dimensions(level=1)}")
print(f"Dimensions at level 2: {pdx_slide.level_dimensions(level=2)}")

# Access the openslide properties through the Slide object
# (histopath inherits from the openslide)
print(f"Type:             {type(pdx_slide._wsi.properties)}")
print(f"Total properties: {len(pdx_slide._wsi.properties)}")
print(f"Property value:   {pdx_slide._wsi.properties['aperio.AppMag']}")  # access a property
mag = int(pdx_slide._wsi.properties['aperio.AppMag'])

# print(pdx_slide._wsi.properties[openslide.PROPERTY_NAME_MPP_X], '\n')
# print(pdx_slide._wsi.properties, '\n')

print(f"Level count:       {pdx_slide._wsi.level_count}")  # access a property
print(f"Level downsamples: {pdx_slide._wsi.level_downsamples}")  # access a property
print(f"Level dimensions:  {pdx_slide._wsi.level_dimensions}")  # access a property


# Sampling and resolution
def calc_eff_mpp(slide, level=0):
    """ effective MPP = downsample x MPP """
    mpp_eff = slide._wsi.level_downsamples[level] * float(slide._wsi.properties[openslide.PROPERTY_NAME_MPP_X])  # effective magnification
    print('Downsample:', slide._wsi.level_downsamples[level])
    print('Level:     ', level)
    print('MPP (um):  ', mpp_eff)
    return mpp_eff


for level in range(pdx_slide._wsi.level_count):
    mpp_eff = calc_eff_mpp(slide=pdx_slide, level=level)
    
# Calc tile size
tile_px = 300
level = 0
mpp_eff = calc_eff_mpp(pdx_slide, level=0)
tile_um = mpp_eff * tile_px
print('Tile (um):', tile_um)


# ------------------------------------------
# Aggregate metadata into df from all slides
# ------------------------------------------
meta_list = []  # list of dicts
for i, fname in enumerate(files):
    if i % 50 == 0:
        print(f'slide {i}: {fname.name}')
    # print(f'slide {i}: {fname.name}')
    
    # Load slide
    img_inpath = str(fname)
    img_outpath = os.path.join(str(datapath), 'processed', fname.with_suffix('').name)
    pdx_slide = Slide(path=img_inpath, processed_path=img_outpath)

    # Create dict that contains the slide metadata (properties)
    ignore_property = ['aperio.User', 'openslide.comment', 'openslide.quickhash-1', 'tiff.ImageDescription']
    meta = {}
    for p_name in pdx_slide._wsi.properties:
        # print('{}: {}'.format( p_name, pdx_slide._wsi.properties[p_name] ))
        if p_name in ignore_property:
            continue
        meta[p_name] = pdx_slide._wsi.properties[p_name]
        
    # Append the slide meta to a list
    meta_list.append(meta)
    del pdx_slide
    
# Create df    
meta_df = pd.DataFrame(meta_list)
meta_df = meta_df[[c for c in sorted(meta_df.columns)]]
cols = ['aperio.ImageID'] + [c for c in meta_df.columns if c != 'aperio.ImageID']
meta_df = meta_df[cols]
print('Shape', meta_df.shape)
pprint(meta_df.T.iloc[:4, :7])

# Save
meta_df.to_csv(metapath/'meta_from_wsi_images.csv', index=False)
print('Done.')
