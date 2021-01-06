"""
Merge three metadata files to create ../data/meta/meta_merged.csv:

1. ImageID_PDMRID_CrossRef.csv:  meta comes with PDX slides
2. PDX_Meta_Information.csv:     meta from Yitan
3. meta_from_wsi_slides.csv:     meta extracted from SVS slides using openslide

Note! Before running this code, generate meta_from_wsi_slides.csv with get_meta_from_slides.py.

Note! Yitan's file has some missing samples for which we do have the slides.
The missing samples either don't have response data or expression data.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import glob
from pprint import pprint

