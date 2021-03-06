import sys
import csv
import numpy as np
from functools import partial

import tensorflow as tf
assert tf.__version__ >= "2.0"

from config import cfg
from tfrecords import FEA_SPEC_RSP, FEA_SPEC_RNA, FEA_SPEC_RNA_NEW

GREEN = '\033[92m'
ENDC = '\033[0m'
def green(text):
    return GREEN + str(text) + ENDC

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'


# Create training_dataset (instance of class Dataset)
# Part of the Dataset class construction
def read_annotations(annotations_file):
    '''
    sfutil.read_annotations(annotations_file)
    Read an annotations file.
    '''
    results = []
    # Open annotations file and read header
    with open(annotations_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # First, try to open file
        try:
            header = next(csv_reader, None)
        except OSError:
            log.error(f"Unable to open annotations file {green(annotations_file)}, is it open in another program?")
            sys.exit()

        for row in csv_reader:
            row_dict = {}
            for i, key in enumerate(header):
                row_dict[key] = row[i]
            results += [row_dict]
    return header, results


def _process_image(image_string, augment):
    '''Converts a JPEG-encoded image string into RGB array, using normalization if specified.'''
    image = tf.image.decode_jpeg(image_string, channels = 3)

    # if self.normalizer:
    #     image = tf.py_function(self.normalizer.tf_to_rgb, [image], tf.int32)

    image = tf.image.per_image_standardization(image)

    if augment:
        # Apply augmentations
        # Rotate 0, 90, 180, 270 degrees
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Random flip and rotation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])
    return image


def decode_np_arr(tensor):
    return np.frombuffer(tensor.numpy(), dtype=cfg.FEA_DTYPE)


def scale_fea(data, scaler):
    """ Scaler is an object of class sklearn.preprocessing.StandardScaler. """
    fea_mean = tf.constant(scaler.mean_, tf.float32)
    fea_scale = tf.constant(scaler.scale_, tf.float32)
    return (data - fea_mean) / fea_scale


def _parse_tfrec_fn_rna(record,
                        include_smp_names=True,
                        use_tile=True,
                        use_ge=False,
                        ge_scaler=None,
                        id_name='slide',
                        MODEL_TYPE=None,
                        ANNOTATIONS_TABLES=None,
                        AUGMENT=True
                        ):
    '''Parses raw entry read from TFRecord.'''
    # features = tf.io.parse_single_example(record, FEA_SPEC_RNA)
    features = tf.io.parse_single_example(record, FEA_SPEC_RNA_NEW)
    slide = features[id_name]

    #if self.MODEL_TYPE == 'linear':
    if MODEL_TYPE == 'linear':
        label = [ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(NUM_CLASSES)]
    else:
        label = ANNOTATIONS_TABLES[0].lookup(slide)
    # label = {outcome_header[0]: label}  # ap, or {'ctype': label}
    label = {'ctype': label}

    image_dict = {}	

    if use_tile:
        image_string = features['image_raw']      
        image = _process_image(image_string, AUGMENT)
        image_dict.update({'tile_image': image})

    if use_ge:
        # ge_data = tf.cast(features['ge_data'], tf.float32)
        # ge_data = (ge_data-tf.constant(ge_scaler.mean_, tf.float32))/tf.constant(ge_scaler.scale_, tf.float32)
        # image_dict.update({'ge_data': ge_data})      

        # new
        ge_data = tf.py_function(func=decode_np_arr, inp=[features['ge_data']],
                                 Tout=[tf.float32])
        ge_data = tf.reshape(ge_data, [-1])

        if ge_scaler is not None:
            ge_data = scale_fea(ge_data, ge_scaler)

        ge_data = tf.cast(ge_data, tf.float32)
        image_dict.update({'ge_data': ge_data})      
        # new

    if include_smp_names:
        return image_dict, label, slide
    else:
        return image_dict, label


def _parse_tfrec_fn_rsp(record,
                        include_smp_names=True,
                        use_tile=True,
                        use_ge=False,
                        use_dd=False,
                        ge_scaler=None,
                        dd_scaler=None,
                        id_name='smp',
                        MODEL_TYPE=None,
                        ANNOTATIONS_TABLES=None,
                        AUGMENT=True
                        ):
    ''' Parses raw entry read from TFRecord. '''
    feature_description = FEA_SPEC_RSP

    features = tf.io.parse_single_example(record, feature_description)
    #slide = features['slide']
    #smp = features[cfg.ID_NAME]
    smp = features[id_name]

    #if MODEL_TYPE == 'linear':
    #    #label = [ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(NUM_CLASSES)]
    #    label = [ANNOTATIONS_TABLES[oi].lookup(smp) for oi in range(NUM_CLASSES)]
    #else:
    #    #label = ANNOTATIONS_TABLES[0].lookup(slide)
    #    label = ANNOTATIONS_TABLES[0].lookup(smp)
    # label = {outcome_header[0]: label}  # ap, or {'ctype': label}
    label = {'Response': tf.cast(features['Response'], tf.int64)}  # ap, or {'ctype': label}

    image_dict = {}

    if use_tile:
        image_string = features['image_raw']      
        image = _process_image(image_string, AUGMENT)
        image_dict.update({'tile_image': image})
        del image

    if use_ge:
        ge_data = tf.py_function(func=decode_np_arr, inp=[features['ge_data']],
                                 Tout=[tf.float32])
        ge_data = tf.reshape(ge_data, [-1])

        if ge_scaler is not None:
            ge_data = scale_fea(ge_data, ge_scaler)

        ge_data = tf.cast(ge_data, tf.float32)
        image_dict.update({'ge_data': ge_data})      
        del ge_data

    if use_dd:
        dd_data = tf.py_function(func=decode_np_arr, inp=[features['dd_data']],
                                 Tout=[tf.float32])
        dd_data = tf.reshape(dd_data, [-1])

        if dd_scaler is not None:
            dd_data = scale_fea(dd_data, dd_scaler)

        dd_data = tf.cast(dd_data, tf.float32)
        image_dict.update({'dd_data': dd_data})      
        del dd_data

    if include_smp_names:
        return image_dict, label, smp
    else:
        return image_dict, label


# def _interleave_tfrecords(tfrecords, batch_size, balance, finite,
#                           max_tiles=None, min_tiles=None,
#                           include_smp_names=False,
#                           parse_fn=None, drop_remainder=False,
#                           use_dd=False,
#                           use_ge=False,
#                           use_tile=True,
#                           id_name=None,
#                           MANIFEST=None, # global var of SlideFlowModel
#                           ANNOTATIONS_TABLES=None, # global var of SlideFlowModel
#                           SLIDE_ANNOTATIONS=None, # global var of SlideFlowModel
#                           MODEL_TYPE=None, # global var of SlideFlowModel
#                           SAMPLES=None): # global var of SlideFlowModel
def _interleave_tfrecords(tfrecords, batch_size, balance, finite,
                          max_tiles=None, min_tiles=None,
                          include_smp_names=False,
                          drop_remainder=False,
                          parse_fn=None,
                          MANIFEST=None, # global var of SlideFlowModel
                          # ANNOTATIONS_TABLES=None, # global var of SlideFlowModel
                          SLIDE_ANNOTATIONS=None, # global var of SlideFlowModel
                          MODEL_TYPE=None, # global var of SlideFlowModel
                          SAMPLES=None,
                          **parse_fn_kwargs): # global var of SlideFlowModel
    ''' Generates an interleaved dataset from a collection of tfrecord files,
    sampling from tfrecord files randomly according to balancing if provided.
    Requires self.MANIFEST. Assumes TFRecord files are named by slide.

    Args:
        tfrecords:				Array of paths to TFRecord files
        batch_size:				Batch size
        balance:				Whether to use balancing for batches. Options are BALANCE_BY_CATEGORY,
                                    BALANCE_BY_PATIENT, and NO_BALANCE. If finite option is used, will drop
                                    tiles in order to maintain proportions across the interleaved dataset.
        augment:					Whether to use data augmentation (random flip/rotate)
        finite:					Whether create finite or infinite datasets. WARNING: If finite option is
                                    used with balancing, some tiles will be skipped.
        max_tiles:				Maximum number of tiles to use per slide.
        min_tiles:				Minimum number of tiles that each slide must have to be included.
        include_smp_names:		Bool, if True, dataset will include slidename (each entry will return image,
                                label, and slidename)
        multi_image:			Bool, if True, will read multiple images from each TFRecord record.
    '''
    DATASETS = {}  # global var of SlideFlowModel

    print(f"Interleaving {len(tfrecords)} tfrecords: finite={finite}, max_tiles={max_tiles}, min_tiles={min_tiles}")
    datasets = []
    datasets_categories = []
    num_tiles = []
    global_num_tiles = 0
    categories = {}
    categories_prob = {}
    categories_tile_fraction = {}

    # if not parse_fn:
    #     parse_fn = _parse_tfrec_fn_rsp

    if tfrecords == []:
        print(f"No TFRecords found.")
        sys.exit()

    for filename in tfrecords:
        smp = filename.split('/')[-1][:-10]
        # smp_data = data[data['smp'] == smp] # (ap)
        # data[data['smp'] == smp].Response # (ap)

        if smp not in SAMPLES:
            continue

        # Determine total number of tiles available in TFRecord
        try:
            tiles = MANIFEST[filename]['total']
        except KeyError:
            print(f"Manifest not finished, unable to find {green(filename)}")

        # Ensure TFRecord has minimum number of tiles; otherwise, skip
        if not min_tiles and tiles == 0:
            print(f"Skipping empty tfrecord {green(smp)}")
            continue
        elif tiles < min_tiles:
            print(f"Skipping tfrecord {green(smp)}; has {tiles} tiles (minimum: {min_tiles})")
            continue

        # Assign category by outcome if this is a categorical model.
        # Otherwise, consider all slides from the same category (effectively skipping balancing);
        # appropriate for linear models.
        # (ap) Get the category of the current sample
        category = SLIDE_ANNOTATIONS[smp]['outcome'] if MODEL_TYPE == 'categorical' else 1
        # data[data['smp'] == smp].Response # (ap)

        # Create a list of tf datasets and a corresponding list of categories (outcomes)
        if filename not in DATASETS:
            # buffer_size=1024*1024*100 num_parallel_reads=tf.data.experimental.AUTOTUNE
            DATASETS.update({filename: tf.data.TFRecordDataset(filename, num_parallel_reads=32)}) 
        datasets += [DATASETS[filename]]
        datasets_categories += [category]

        # Cap number of tiles to take from TFRecord at maximum specified
        if max_tiles and tiles > max_tiles:
            print(f"Only taking maximum of {max_tiles} (of {tiles}) tiles from {sfutil.green(filename)}")
            tiles = max_tiles

        if category not in categories.keys():
            # categories.update({category: {'num_slides': 1,
            #                               'num_tiles': tiles}})
            categories.update({category: {'num_samples': 1,
                                          'num_tiles': tiles}})
        else:
            # categories[category]['num_slides'] += 1
            # categories[category]['num_tiles'] += tiles
            categories[category]['num_samples'] += 1
            categories[category]['num_tiles'] += tiles
        num_tiles += [tiles]

    # Assign weight to each category
    #lowest_category_slide_count = min([categories[i]['num_slides'] for i in categories])
    lowest_category_sample_count = min([categories[i]['num_samples'] for i in categories])
    lowest_category_tile_count = min([categories[i]['num_tiles'] for i in categories])
    for category in categories:
        #categories_prob[category] = lowest_category_slide_count / categories[category]['num_slides']
        categories_prob[category] = lowest_category_sample_count / categories[category]['num_samples']
        categories_tile_fraction[category] = lowest_category_tile_count / categories[category]['num_tiles']

    # Balancing
    if balance == NO_BALANCE:
        print(f"Not balancing input")
        prob_weights = [i/sum(num_tiles) for i in num_tiles]

    if balance == BALANCE_BY_PATIENT:
        print(f"Balancing input across slides")
        prob_weights = [1.0] * len(datasets)
        if finite:
            # Only take as many tiles as the number of tiles in the smallest dataset
            minimum_tiles = min(num_tiles)
            for i in range(len(datasets)):
                num_tiles[i] = minimum_tiles

    # (ap) Assign per-slide weight based on sample count per
    # outcome (category-based/outcome-based stratification)
    if balance == BALANCE_BY_CATEGORY:
        print(f"Balancing input across categories")
        prob_weights = [categories_prob[datasets_categories[i]] for i in range(len(datasets))]
        if finite:
            # Only take as many tiles as the number of tiles in the smallest category -->
            # (ap) i.e., balance by the number of tiles
            for i in range(len(datasets)):
                num_tiles[i] = int(num_tiles[i] * categories_tile_fraction[datasets_categories[i]])
                print(f"Tile fraction (dataset {i+1}/{len(datasets)}): \
                      {categories_tile_fraction[datasets_categories[i]]}, \
                      taking {num_tiles[i]}")
            print(f"Global num tiles: {global_num_tiles}")

    # Take the calculcated number of tiles from each dataset and calculate
    # global number of tiles
    for i in range(len(datasets)):
        datasets[i] = datasets[i].take(num_tiles[i])
        if not finite:
            datasets[i] = datasets[i].repeat()  # (ap) why repeat()??
    global_num_tiles = sum(num_tiles)

    # Interleave datasets
    # sample_from_datasets() returns a dataset that interleaves elements from
    # datasets at random, according to weights if provided, otherwise with
    # uniform probability.
    try:
        dataset = tf.data.experimental.sample_from_datasets(datasets, weights=prob_weights, seed=None)
    except IndexError:
        print(f"No TFRecords found after filter criteria; please ensure all tiles \
              have been extracted and all TFRecords are in the appropriate folder")
        sys.exit()

    # if include_smp_names:
    #     dataset_with_slidenames = dataset.map(
    #             partial(parse_fn, include_smp_names=True,
    #                     use_dd=use_dd, use_ge=use_ge, use_tile=use_tile, id_name=id_name,
    #                     AUGMENT=AUTMENT, ANNOTATIONS_TABLES=ANNOTATIONS_TABLES),
    #             num_parallel_calls=32
    #     ) #tf.data.experimental.AUTOTUNE
    #     dataset_with_slidenames = dataset_with_slidenames.batch(batch_size, drop_remainder=drop_remainder)
    # else:
    #     dataset_with_slidenames = None

    # dataset = dataset.map(
    #     partial(parse_fn, include_smp_names=False,
    #             use_dd=use_dd, use_ge=use_ge, use_tile=use_tile, id_name=id_name,
    #             AUGMENT=AUTMENT, ANNOTATIONS_TABLES=ANNOTATIONS_TABLES),
    #     num_parallel_calls=8
    # )

    if include_smp_names:
        dataset_with_slidenames = dataset.map(
                partial(parse_fn, include_smp_names=True, **parse_fn_kwargs),
                num_parallel_calls=32
        ) #tf.data.experimental.AUTOTUNE
        dataset_with_slidenames = dataset_with_slidenames.batch(batch_size, drop_remainder=drop_remainder)
    else:
        dataset_with_slidenames = None

    dataset = dataset.map(
        partial(parse_fn, include_smp_names=False, **parse_fn_kwargs),
        num_parallel_calls=8
    )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset, dataset_with_slidenames, global_num_tiles
