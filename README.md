Multimodal neural network for drug response prediction in PDX with histology images and gene expressions.

## Create master dataframe (meta, tabular features, and response)
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

Finally, build master df and save in `data/data_merged.csv`.
```
$ python src/build_df.py
```

## Tiling
Generate image tiles from WSI slides and save to `data/tiles_png`.
```
$ python src/tiling.py
```

## Build TFRecords
Build tfrecords to use with TensorFlow2 for all the samples in the master dataframe
and save in `data/data_merged.csv`.
```
$ python src/build_tfrec.py
```

## Generate dataset for an application
Using the master dataframe as the strating point, generate a subset of samples for
further processing. For example, in build_mm_01.py, we create a balanced dataset
(in terms responders and non-responders). For the non-responders, we extract
samples of the same ctype as the responders. The created subset dataframe is stored
in an appropriate foler such as `apps/mm_01/annotations.csv` (we name the file
`annotations.csv` to conform with the naming convention that is used in SlideFlow.
```
$ python src/apps/build_mm_01.py
```

## Generate data splits
```python
$ python src/cv_splits.py --appname mm_01
```
