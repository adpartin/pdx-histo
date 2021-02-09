Multimodal neural network for drug response prediction in PDX with histology images and gene expressions.

## Create dataframes with data
Extract metadata from wsi slides and save the summary in `data/meta/meta_from_wsi_slides.csv`.
```
$ python src/get_meta_from_slides.py
```

Merge 3 metadata files to create `data/meta/meta_merged.csv`:
1. _ImageID_PDMRID_CrossRef.xlsx:  meta comes with PDX slides; crossref
2. PDX_Meta_Information.csv:       meta from Yitan; pdx_meta
3. meta_from_wsi_slides.csv:       meta extracted from SVS slides using openslide; slides_meta
```
$ python src/merge_meta_files.py
```

Build df and save in `data/data_merged.csv`.
```
$ python src/build_df.py
```

## Tiling
Generate image tiles from WSI slides and save to `data/tiles_png`.
```
$ python src/tiling.py
```
