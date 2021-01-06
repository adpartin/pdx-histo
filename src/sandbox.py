import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# To get around renderer issue on OSX going from Matplotlib image to NumPy image.
import matplotlib
matplotlib.use('Agg')

# import deephistopath.wsi
from deephistopath.wsi.filter import *
from deephistopath.wsi.slide import *
from deephistopath.wsi.tiles import *
from deephistopath.wsi.util import *

from deephistopath.wsi import filter
from deephistopath.wsi import slide
from deephistopath.wsi import tiles
from deephistopath.wsi import util

fpath = Path(__file__).resolve().parent

def get_slide_num_from_path(slide_filepath):
    return int(os.path.basename(slide_filepath).split('.svs')[0])  ## ap

## ap
slides_path = fpath/'data/training_slides'
slide_path_list = glob.glob(os.path.join(slides_path, '*.svs'))
image_num_list = [get_slide_num_from_path(slide_filepath) for slide_filepath in slide_path_list]

# Choose sample slide
slide_number = 8657
spath = str(slides_path/f'{slide_number}.{SRC_TRAIN_EXT}')


# ================================================
# Part 1
# Whole-slide image preprocessing in Python
# ================================================

# pdx_slide = open_slide(spath)

# show_slide(slide_number)

## slide_info(display_all_properties=True)  ## original
# slide_info(slides_path, display_all_properties=True)

# slide_stats(slides_path)

# slide.training_slide_to_image(slide_number)
# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# img.show()

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, name="RGB")

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import ipdb; ipdb.set_trace(context=11)
singleprocess_training_slides_to_images(slides_path)

# multiprocess_training_slides_to_images(slides_path)  # didn't try

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ================================================
# Part 2
# Apply filters for tissue segmentation
# ================================================

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# util.display_img(grayscale, "Grayscale")

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# complement = filter.filter_complement(grayscale)
# util.display_img(complement, "Complement")

# ------------
# Thresholding
# ------------

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# complement = filter.filter_complement(grayscale)
# thresh = filter.filter_threshold(complement, threshold=100)
# util.display_img(thresh, "Threshold")

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# complement = filter.filter_complement(grayscale)
# hyst = filter.filter_hysteresis_threshold(complement)
# util.display_img(hyst, "Hysteresis Threshold")

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# complement = filter.filter_complement(grayscale)
# otsu = filter.filter_otsu_threshold(complement)
# util.display_img(otsu, "Otsu Threshold")

# --------
# Contrast
# --------

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# complement = filter.filter_complement(grayscale)
# contrast_stretch = filter.filter_contrast_stretch(complement, low=100, high=200)
# util.display_img(contrast_stretch, "Contrast Stretch")

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# util.display_img(grayscale, "Grayscale")
# hist_equ = filter.filter_histogram_equalization(grayscale)
# util.display_img(hist_equ, "Histogram Equalization")

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# grayscale = filter.filter_rgb_to_grayscale(rgb)
# util.display_img(grayscale, "Grayscale")
# adaptive_equ = filter.filter_adaptive_equalization(grayscale)
# util.display_img(adaptive_equ, "Adaptive Equalization")

# -----
# Color
# -----

# 
# RGB to HED
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# hed = filter.filter_rgb_to_hed(rgb)
# hema = filter.filter_hed_to_hematoxylin(hed)
# util.display_img(hema, "Hematoxylin Channel")
# eosin = filter.filter_hed_to_eosin(hed)
# util.display_img(eosin, "Eosin Channel")

# 
# Green channel filter
# 

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_green = filter.filter_green_channel(rgb)
# util.display_img(not_green, "Green Channel Filter")

# 
# Grays filter
# 

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_grays = filter.filter_grays(rgb)
# util.display_img(not_grays, "Grays Filter")

# 
# Red filter
# 

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_red = filter.filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90, display_np_info=True)
# util.display_img(not_red, "Red Filter (150, 80, 90)")
# util.display_img(util.mask_rgb(rgb, not_red), "Not Red")
# util.display_img(util.mask_rgb(rgb, ~not_red), "Red")

#
# Red pen filter
#

# result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
#          filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
#          filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
#          filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
#          filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
#          filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
#          filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
#          filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
#          filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45)

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_blue_pen = filter.filter_blue_pen(rgb)
# util.display_img(not_blue_pen, "Blue Pen Filter")
# util.display_img(util.mask_rgb(rgb, not_blue_pen), "Not Blue Pen")
# util.display_img(util.mask_rgb(rgb, ~not_blue_pen), "Blue Pen")

#
# Blue filter
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_blue = filter.filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180, display_np_info=True)
# util.display_img(not_blue, "Blue Filter (130, 155, 180)")
# util.display_img(util.mask_rgb(rgb, not_blue), "Not Blue")
# util.display_img(util.mask_rgb(rgb, ~not_blue), "Blue")

# 
# Blue pen filter
# 

# result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
#          filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
#          filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
#          filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
#          filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
#          filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
#          filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
#          filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
#          filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
#          filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
#          filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
#          filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_blue_pen = filter.filter_blue_pen(rgb)
# util.display_img(not_blue_pen, "Blue Pen Filter")
# util.display_img(util.mask_rgb(rgb, not_blue_pen), "Not Blue Pen")
# util.display_img(util.mask_rgb(rgb, ~not_blue_pen), "Blue Pen")

# not_blue = filter.filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180, display_np_info=True)
# not_blue_pen = filter.filter_blue_pen(rgb)
# print("filter_blue: " + filter.mask_percentage_text(filter.mask_percent(not_blue)))
# print("filter_blue_pen: " + filter.mask_percentage_text(filter.mask_percent(not_blue_pen)))

# 
# Green filter
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_green = filter.filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140, display_np_info=True)
# util.display_img(not_green, "Green Filter (150, 160, 140)")
# util.display_img(util.mask_rgb(rgb, not_green), "Not Green")
# util.display_img(util.mask_rgb(rgb, ~not_green), "Green")

#
# Green pen filter
#

# result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
#          filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
#          filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
#          filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
#          filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
#          filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
#          filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
#          filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
#          filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
#          filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
#          filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
#          filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
#          filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
#          filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
#          filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195)

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "RGB")
# not_green_pen = filter.filter_green_pen(rgb)
# util.display_img(not_green_pen, "Green Pen Filter")
# util.display_img(util.mask_rgb(rgb, not_green_pen), "Not Green Pen")
# util.display_img(util.mask_rgb(rgb, ~not_green_pen), "Green Pen")

# 
# K-means segmentation
# 

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# kmeans_seg = filter.filter_kmeans_segmentation(rgb, n_segments=3000)
# util.display_img(kmeans_seg, "K-Means Segmentation", bg=True)
# otsu_mask = util.mask_rgb(rgb, filter.filter_otsu_threshold(filter.filter_complement(filter.filter_rgb_to_grayscale(rgb)), output_type="bool"))
# util.display_img(otsu_mask, "Image after Otsu Mask", bg=True)
# kmeans_seg_otsu = filter.filter_kmeans_segmentation(otsu_mask, n_segments=3000)
# util.display_img(kmeans_seg_otsu, "K-Means Segmentation after Otsu Mask", bg=True)

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# rag_thresh = filter.filter_rag_threshold(rgb)
# util.display_img(rag_thresh, "RAG Threshold (9)", bg=True)
# rag_thresh = filter.filter_rag_threshold(rgb, threshold=1)
# util.display_img(rag_thresh, "RAG Threshold (1)", bg=True)
# rag_thresh = filter.filter_rag_threshold(rgb, threshold=20)
# util.display_img(rag_thresh, "RAG Threshold (20)", bg=True)

#
# RGB to HSV
#

# from deephistopath.wsi import slide
# from deephistopath.wsi import tiles
# from deephistopath.wsi import util

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# tiles.display_image_with_rgb_and_hsv_histograms(rgb)


# ================================================
# Part 3
# Morphology operators
# ================================================

#
# Erosion
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# bin_erosion_5 = filter.filter_binary_erosion(no_grays, disk_size=5)
# util.display_img(bin_erosion_5, "Binary Erosion (5)", bg=True)
# bin_erosion_20 = filter.filter_binary_erosion(no_grays, disk_size=20)
# util.display_img(bin_erosion_20, "Binary Erosion (20)", bg=True)

#
# Dilation
# 

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# bin_dilation_5 = filter.filter_binary_dilation(no_grays, disk_size=5)
# util.display_img(bin_dilation_5, "Binary Dilation (5)", bg=True)
# bin_dilation_20 = filter.filter_binary_dilation(no_grays, disk_size=20)
# util.display_img(bin_dilation_20, "Binary Dilation (20)", bg=True)

# 
# Opening
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# bin_opening_5 = filter.filter_binary_opening(no_grays, disk_size=5)
# util.display_img(bin_opening_5, "Binary Opening (5)", bg=True)
# bin_opening_20 = filter.filter_binary_opening(no_grays, disk_size=20)
# util.display_img(bin_opening_20, "Binary Opening (20)", bg=True)

#
# Closing
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# bin_closing_5 = filter.filter_binary_closing(no_grays, disk_size=5)
# util.display_img(bin_closing_5, "Binary Closing (5)", bg=True)
# bin_closing_20 = filter.filter_binary_closing(no_grays, disk_size=20)
# util.display_img(bin_closing_20, "Binary Closing (20)", bg=True)

#
# Remove small objects
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# remove_small_100 = filter.filter_remove_small_objects(no_grays, min_size=100)
# util.display_img(remove_small_100, "Remove Small Objects (100)", bg=True)
# remove_small_10000 = filter.filter_remove_small_objects(no_grays, min_size=10000)
# util.display_img(remove_small_10000, "Remove Small Objects (10000)", bg=True)

#
# Remove small holes
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# no_grays = filter.filter_grays(rgb, output_type="bool")
# util.display_img(no_grays, "No Grays", bg=True)
# remove_small_100 = filter.filter_remove_small_holes(no_grays, min_size=100)
# util.display_img(remove_small_100, "Remove Small Holes (100)", bg=True)
# remove_small_10000 = filter.filter_remove_small_holes(no_grays, min_size=10000)
# util.display_img(remove_small_10000, "Remove Small Holes (10000)", bg=True)

#
# Fill holes
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# no_grays = filter.filter_grays(rgb, output_type="bool")
# fill_holes = filter.filter_binary_fill_holes(no_grays)
# util.display_img(fill_holes, "Fill Holes", bg=True)

# remove_holes_100 = filter.filter_remove_small_holes(no_grays, min_size=100, output_type="bool")
# util.display_img(fill_holes ^ remove_holes_100, "Differences between Fill Holes and Remove Small Holes (100)", bg=True)

# remove_holes_10000 = filter.filter_remove_small_holes(no_grays, min_size=10000, output_type="bool")
# util.display_img(fill_holes ^ remove_holes_10000, "Differences between Fill Holes and Remove Small Holes (10000)", bg=True)

#
# Entropy
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original")
# gray = filter.filter_rgb_to_grayscale(rgb)
# util.display_img(gray, "Grayscale")
# entropy = filter.filter_entropy(gray, output_type="bool")
# util.display_img(entropy, "Entropy")
# util.display_img(util.mask_rgb(rgb, entropy), "Original with Entropy Mask")
# util.display_img(util.mask_rgb(rgb, ~entropy), "Original with Inverse of Entropy Mask")

#
# Canny edge detection
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original", bg=True)
# gray = filter.filter_rgb_to_grayscale(rgb)
# canny = filter.filter_canny(gray, output_type="bool")
# util.display_img(canny, "Canny", bg=True)
# rgb_crop = rgb[300:900, 300:900]
# canny_crop = canny[300:900, 300:900]
# util.display_img(rgb_crop, "Original", size=24, bg=True)
# util.display_img(util.mask_rgb(rgb_crop, ~canny_crop), "Original with ~Canny Mask", size=24, bg=True)

#
# Combining filters
#

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original")
# no_green_pen = filter.filter_green_pen(rgb)
# util.display_img(no_green_pen, "No Green Pen")
# no_blue_pen = filter.filter_blue_pen(rgb)
# util.display_img(no_blue_pen, "No Blue Pen")
# no_gp_bp = no_green_pen & no_blue_pen
# util.display_img(no_gp_bp, "No Green Pen, No Blue Pen")
# util.display_img(util.mask_rgb(rgb, no_gp_bp), "Original with No Green Pen, No Blue Pen")

# img_path = slide.get_training_image_path(slide_number)
# img = slide.open_image(img_path)
# rgb = util.pil_to_np_rgb(img)
# util.display_img(rgb, "Original")
# mask = filter.filter_grays(rgb) & filter.filter_green_channel(rgb) & filter.filter_green_pen(rgb) & filter.filter_blue_pen(rgb)
# mask = filter.filter_remove_small_objects(mask, min_size=100, output_type="bool")
# util.display_img(mask, "No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects")
# util.display_img(util.mask_rgb(rgb, mask), "Original with No Grays, Green Channel, No Green Pen, No Blue Pen, No Small Objects")
# util.display_img(util.mask_rgb(rgb, ~mask), "Original with Inverse Mask")

# mask_not_green = filter_green_channel(rgb)
# mask_not_gray = filter_grays(rgb)
# mask_no_red_pen = filter_red_pen(rgb)
# mask_no_green_pen = filter_green_pen(rgb)
# mask_no_blue_pen = filter_blue_pen(rgb)
# mask_gray_green_pens = mask_not_gray & mask_not_green & mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
# mask_remove_small = filter_remove_small_objects(mask_gray_green_pens, min_size=500, output_type="bool")

# ... more code

#
# Applying filters to multiple images
#

# apply_filters_to_image_list(image_num_list, save, display)
# apply_filters_to_image_range(start_ind, end_ind, save, display)
# singleprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None)
# multiprocess_apply_filters_to_images(save=True, display=False, html=True, image_num_list=None)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import ipdb; ipdb.set_trace(context=11)
filter.singleprocess_apply_filters_to_images(image_num_list=image_num_list)

# filter.multiprocess_apply_filters_to_images(image_num_list=image_num_list)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#
# Overmask avoidance
#

# ... code


# ================================================
# Part 4
# Top tile retrieval
# ================================================

"""
- The tiling params are stored in tiles.py
- Saving tiles:
  1. singleprocess_filtered_images_to_tiles( image_num_list )
  2. image_list_to_tiles( image_num_list )
  3. summary_and_tiles()
  4. save_tile()
  5. save_display_tile(self, save=True, display=False)
  6. tile_to_np_tile( tile )
- Convert tile information (Tile object) into the corresponding
  tile as a NumPy image read from the whole-slide image file:
  tiles.tile_to_np_tile( tile )
"""

#
# Tiles
#

# tiles.summary_and_tiles(slide_number, display=True, save_summary=True, save_data=False, save_top_tiles=False)

#
# Tile scoring
#

# ...
# tile = tiles.dynamic_tile(slide_num=8068, row=29, col=16, small_tile_in_tile=False)
# tiles.display_image_with_hsv_hue_histogram(tile.get_np_scaled_tile(), scale_up=True)  # doesnt work

# ...
# tile = tiles.dynamic_tile(slide_num=8068, row=29, col=16, small_tile_in_tile=False)
# tile.display_with_histograms();

# ...
# tiles.summary_and_tiles(8068, display=True, save_summary=True, save_data=True, save_top_tiles=False)

# ...
# tiles.summary_and_tiles(8068, display=True, save_summary=True, save_data=False, save_top_tiles=False)

#
# Top tile retrieval
#

# ...
# tiles.summary_and_tiles(8068, display=True, save_summary=True, save_data=False, save_top_tiles=True)

# ...
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import ipdb; ipdb.set_trace(context=11)
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

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ...
# tile_summary = tiles.dynamic_tiles(8068)
# top_tiles = tile_summary.top_tiles()
# for t in top_tiles:
#   print(t)

# ...
# tile_summary = tiles.dynamic_tiles(8068)
# top_tiles = tile_summary.top_tiles()
# for t in top_tiles:
#   print(t)
#   np_tile = t.get_np_tile()

# ...
# tile_summary = tiles.dynamic_tiles(8068)
# top = tile_summary.top_tiles()[:2]
# for t in top:
#   t.display_with_histograms()

# ...
# tile_summary = tiles.dynamic_tiles(8068)
# tile_summary.display_summaries()
# ts = tile_summary.tiles_by_tissue_percentage()
# ts[999].display_with_histograms()
# ts[1499].display_with_histograms()

# ...
# tile_summary = tiles.dynamic_tiles(8068)
# tile_summary.get_tile(25, 30).display_tile()
# tile_summary.get_tile(25, 31).display_tile()

# Scaling, filtering, tiling, scoring, and saving the top tiles
# can be accomplished in batch mode using multiprocessing in the
# following manner.
# slide.multiprocess_training_slides_to_images()
# filter.multiprocess_apply_filters_to_images()
# tiles.multiprocess_filtered_images_to_tiles()
