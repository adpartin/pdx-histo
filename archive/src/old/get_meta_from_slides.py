""" 
Extract metadata from wsi slides and save the summary
in ../data/meta/meta_from_wsi_slides.
"""

import os
import sys
from pathlib import Path
import glob
from pprint import pprint
import pandas as pd
import numpy as np

import openslide
# from deephistopath.wsi import filter
from deephistopath.wsi import slide
# from deephistopath.wsi import tiles
from deephistopath.wsi import util

from config import cfg

fdir = Path(__file__).resolve().parent

SLIDESPATH = cfg.DATADIR/'doe-globus-pdx-data'  # path to raw WSI slides
METAPATH = cfg.DATADIR/'meta'


def calc_eff_mpp(s, level=0, verbose=False):
    """ Calc the effective MPP for a given level.
    effective MPP = downsample x MPP 
    Args:
        slide: openslide object
        level: downsample level
    """
    mag_0 = float(s.properties['aperio.AppMag'])
    mpp_eff = s.level_downsamples[level] * float(s.properties[openslide.PROPERTY_NAME_MPP_X])  # effective magnification
    if verbose:
        print('Downsample:', s.level_downsamples[level])
        print('Level:     ', level)
        print('MPP (um):  ', mpp_eff)
        print('Mag:       ', mag_0/s.level_downsamples[level])
    return mpp_eff


if __name__ == "__main__":

    t = util.Time()

    # import ipdb; ipdb.set_trace()

    # Glob slides
    slides_path_list = glob.glob(os.path.join(SLIDESPATH, '*.svs'))
    print(f'Total slides: {len(slides_path_list)}')
    # print(os.path.basename(slides_path_list[0]))

    # Confirm that svs file names match and the 'Image ID' column in excel file
    s1 = set([int(os.path.basename(x).split('.')[0]) for x in slides_path_list])
    # df_img = pd.read_excel(METAPATH/crossref_meta_fname)  # csv
    df_img = pd.read_excel(METAPATH/cfg.CROSSREF_FNAME, engine='openpyxl', header=2)  # excel
    df_img = df_img.rename(columns={'Image ID': 'image_id'})
    df_img = df_img.dropna(axis=0, how='all').reset_index(drop=True)
    df_img['image_id'] = [int(x) if ~np.isnan(x) else x for x in df_img['image_id'].values]
    s2 = set(df_img['image_id'].values)

    print("SVS slides that are present in the folder but not in the 'Image ID' " \
          "column: {}".format(s1.difference(s2)))
    print("Slide ids that are in the 'Image ID' column but not present in the " \
          "folder:  {}".format(s2.difference(s1)))

    # Explore the Slide object
    s = slide.open_slide(slides_path_list[0])
    print(f"\nFile type: {type(s.properties)}")
    print(f"Properties:  {len(s.properties)}")
    print(f"AppMag:      {s.properties['aperio.AppMag']}")  # access a property
    mag = int(s.properties['aperio.AppMag'])

    print(f"Level count:       {s.level_count}")         # access a property
    print(f"Level downsamples: {s.level_downsamples}")   # access a property
    print(f"Level dimensions:  {s.level_dimensions}\n")  # access a property

    for level in range(s.level_count):
        print()
        mpp_eff = calc_eff_mpp(s=s, level=level, verbose=True)
        
    # Calc tile size
    tile_px = 300
    level = 0
    mpp_eff = calc_eff_mpp(s, level=0)
    tile_um = mpp_eff * tile_px
    print(f'\nTile (um): {tile_um}\n')


    # ------------------------------------------
    # Aggregate metadata into df from all slides
    # ------------------------------------------

    # import ipdb; ipdb.set_trace(context=11)

    meta_list = []  # list of dicts
    print_after = 50

    for i, sname in enumerate(slides_path_list):
        if i % print_after == 0:
            print(f'slide {i}: {sname.split(os.sep)[-1]}')
        
        # Create dict to contain slide metadata (properties)
        s = slide.open_slide(sname)
        ignore_property = ['aperio.User', 'openslide.comment',
                           'openslide.quickhash-1', 'tiff.ImageDescription']
        meta = {pname: s.properties[pname] for pname in s.properties if pname not in ignore_property}
        meta.update({'disk_memory': os.path.getsize(sname)})  # get the disk memory the file takes
        meta_list.append(meta)  # append dict with slide meta to a list
        del s
        
    # Create df    
    meta_df = pd.DataFrame(meta_list)
    meta_df = meta_df[[c for c in sorted(meta_df.columns)]]
    cols = ['aperio.ImageID'] + [c for c in meta_df.columns if c != 'aperio.ImageID']
    meta_df = meta_df[cols]
    print('\nShape', meta_df.shape)
    pprint(meta_df.T.iloc[:4, :7])

    # Save
    print('\nSave slides metadata in csv.')
    meta_df.to_csv(METAPATH/cfg.SLIDES_META_FNAME, index=False)
    t.elapsed_display()
    print('Done.')
