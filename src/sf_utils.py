import csv
import json
import numpy as np
import os
import sys
from functools import partial
from typing import Optional
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
assert tf.__version__ >= "2.0"

# from config import cfg
# from tfrecords import FEA_SPEC_RSP, FEA_SPEC_RNA, FEA_SPEC_RNA_NEW, FEA_SPEC_RSP_DRUG_PAIR
from src.config import cfg
from src.tfrecords import FEA_SPEC_RSP, FEA_SPEC_RNA, FEA_SPEC_RNA_NEW, FEA_SPEC_RSP_DRUG_PAIR

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


def parse_tfrec_fn_rna(record,
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


#def parse_tfrec_fn_rsp(record,
#                       include_smp_names=True,
#                       use_tile=True,
#                       use_ge=False,
#                       use_dd=False,
#                       ge_scaler=None,
#                       dd_scaler=None,
#                       id_name='smp',
#                       MODEL_TYPE=None,
#                       ANNOTATIONS_TABLES=None,
#                       AUGMENT=True
#                       ):
#    ''' Parses raw entry read from TFRecord. '''
#    feature_description = FEA_SPEC_RSP

#    features = tf.io.parse_single_example(record, feature_description)
#    #slide = features['slide']
#    #smp = features[cfg.ID_NAME]
#    smp = features[id_name]

#    #if MODEL_TYPE == 'linear':
#    #    #label = [ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(NUM_CLASSES)]
#    #    label = [ANNOTATIONS_TABLES[oi].lookup(smp) for oi in range(NUM_CLASSES)]
#    #else:
#    #    #label = ANNOTATIONS_TABLES[0].lookup(slide)
#    #    label = ANNOTATIONS_TABLES[0].lookup(smp)
#    # label = {outcome_header[0]: label}  # ap, or {'ctype': label}
#    label = {'Response': tf.cast(features['Response'], tf.int64)}  # ap, or {'ctype': label}

#    image_dict = {}

#    if use_tile:
#        image_string = features['image_raw']      
#        image = _process_image(image_string, AUGMENT)
#        image_dict.update({'tile_image': image})
#        del image

#    if use_ge:
#        ge_data = tf.py_function(func=decode_np_arr, inp=[features['ge_data']],
#                                 Tout=[tf.float32])
#        ge_data = tf.reshape(ge_data, [-1])

#        if ge_scaler is not None:
#            ge_data = scale_fea(ge_data, ge_scaler)

#        ge_data = tf.cast(ge_data, tf.float32)
#        image_dict.update({'ge_data': ge_data})      
#        del ge_data

#    if use_dd:
#        dd_data = tf.py_function(func=decode_np_arr, inp=[features['dd_data']],
#                                 Tout=[tf.float32])
#        dd_data = tf.reshape(dd_data, [-1])

#        if dd_scaler is not None:
#            dd_data = scale_fea(dd_data, dd_scaler)

#        dd_data = tf.cast(dd_data, tf.float32)
#        image_dict.update({'dd_data': dd_data})      
#        del dd_data

#    if include_smp_names:
#        return image_dict, label, smp
#    else:
#        return image_dict, label


def interleave_tfrecords(tfrecords, batch_size, balance, finite,
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
    # TODO: optimizing with data api
    # https://www.tensorflow.org/guide/data_performance
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
        # (ap)
        # tfr_files = tf.data.Dataset.from_tensor_slices(tfrecords)
        # dataset = tf.data.experimental.sample_from_datasets(tfr_files, weights=prob_weights, seed=None)
    except IndexError:
        print(f"No TFRecords found after filter criteria; please ensure all tiles \
              have been extracted and all TFRecords are in the appropriate folder")
        sys.exit()

    # (ap) recommended to use batch before map
    # dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if include_smp_names:
        dataset_with_smp_names = dataset.map(
                partial(parse_fn, include_smp_names=True, **parse_fn_kwargs),
                num_parallel_calls=32
        ) #tf.data.experimental.AUTOTUNE
        dataset_with_smp_names = dataset_with_smp_names.batch(batch_size, drop_remainder=drop_remainder)
    else:
        dataset_with_smp_names = None

    num_parallel_calls = 8
    # num_parallel_calls = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(
        partial(parse_fn, include_smp_names=False, **parse_fn_kwargs),
        num_parallel_calls=num_parallel_calls
    )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # (ap)
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, dataset_with_smp_names, global_num_tiles


def parse_tfrec_fn_rsp(record,
                       # include_smp_names=True,
                       include_meta=True,
                       use_tile=True,
                       use_ge=False,
                       use_dd2=False,
                       use_dd1=False,
                       ge_scaler=None,
                       dd1_scaler=None,
                       dd2_scaler=None,
                       id_name="smp",
                       MODEL_TYPE=None,
                       ANNOTATIONS_TABLES=None,
                       AUGMENT=True
                       ):
    ''' Parses raw entry read from TFRecord. '''
    feature_description = FEA_SPEC_RSP_DRUG_PAIR

    features = tf.io.parse_single_example(record, feature_description)
    #slide = features['slide']
    #smp = features[cfg.ID_NAME]
    smp = features[id_name]

    # Meta
    meta = {}
    meta_fields = ["index", "smp", "Group", "grp_name",
                   "Response",
                   "Sample", "model", "patient_id", "specimen_id", "sample_id", "image_id",
                   "ctype", "csite",
                   "Drug1", "Drug2", "trt", "aug"]
    for f in meta_fields:
        meta[f] = features[f] if f in features.keys() else None
    
    #if MODEL_TYPE == 'linear':
    #    #label = [ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(NUM_CLASSES)]
    #    label = [ANNOTATIONS_TABLES[oi].lookup(smp) for oi in range(NUM_CLASSES)]
    #else:
    #    #label = ANNOTATIONS_TABLES[0].lookup(slide)
    #    label = ANNOTATIONS_TABLES[0].lookup(smp)
    
    # TODO: {"Response": ...} doesn't work with class_weight!
    label = {"Response": tf.cast(features["Response"], tf.int64)}  # ap, or {'ctype': label}
    # label = tf.cast(features["Response"], tf.int64)

    image_dict = {}

    if use_tile:
        image_string = features["image_raw"]
        image = _process_image(image_string, AUGMENT)
        image_dict.update({"tile_image": image})
        del image

    if use_ge:
        ge_data = tf.py_function(func=decode_np_arr, inp=[features["ge_data"]],
                                 Tout=[tf.float32])
        ge_data = tf.reshape(ge_data, [-1])

        if ge_scaler is not None:
            ge_data = scale_fea(ge_data, ge_scaler)

        ge_data = tf.cast(ge_data, tf.float32)
        image_dict.update({"ge_data": ge_data})      
        del ge_data

    if use_dd1:
        dd_data = tf.py_function(func=decode_np_arr, inp=[features["dd1_data"]],
                                 Tout=[tf.float32])
        dd_data = tf.reshape(dd_data, [-1])

        if dd1_scaler is not None:
            dd_data = scale_fea(dd_data, dd1_scaler)

        dd_data = tf.cast(dd_data, tf.float32)
        image_dict.update({"dd1_data": dd_data})
        del dd_data

    if use_dd2:
        dd_data = tf.py_function(func=decode_np_arr, inp=[features["dd2_data"]],
                                 Tout=[tf.float32])
        dd_data = tf.reshape(dd_data, [-1])

        if dd2_scaler is not None:
            dd_data = scale_fea(dd_data, dd2_scaler)

        dd_data = tf.cast(dd_data, tf.float32)
        image_dict.update({"dd2_data": dd_data})
        del dd_data

    # if include_smp_names:
    #     return image_dict, label, smp
    if include_meta:
        return image_dict, label, meta
    else:
        return image_dict, label


def create_tf_data(tfrecords,
                   n_concurrent_shards=16,
                   shuffle_size=8192,  # 1024*8
                   repeat=True,
                   epochs=1,
                   batch_size=32,
                   drop_remainder=False,
                   seed=None,
                   prefetch=1,
                   # ANNOTATIONS_TABLES=None, # global var of SlideFlowModel
                   parse_fn=None,
                   include_meta=False,
                   **parse_fn_kwargs): # global var of SlideFlowModel
    """ ...

    Args:
        ...
    """
    # TODO: optimizing with data api
    # https://www.tensorflow.org/guide/data_performance

    # -------------------------------------------------------------------------
    # https://docs.google.com/presentation/d/16kHNtQslt-yuJ3w8GIx-eEH6t_AvFeQOchqGRFpAD7U/edit#slide=id.g254d08e080_0_370
    # -------------------------------------------------------------------------
    # # Use interleave() and prefetch() to read many files concurrently.
    # files = tf.data.Dataset.list_files("*.tfrecord")
    # dataset = files.interleave(lambda x: TFRecordDataset(x).prefetch(100),
    #                            cycle_length=8)
    # # Use num_parallel_calls to parallelize map().
    # dataset = dataset.map(lambda record: tf.parse_single_example(record),
    #                       num_parallel_calls=64)
    # dataset = dataset.shuffle(10000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.batch(128)
    # dataset = dataset.prefetch(1)  # Use prefetch() to overlap the producer and consumer.

    # -------------------------------------------------------------------------
    # https://github.com/jkjung-avt/keras_imagenet/blob/master/utils/dataset.py
    # -------------------------------------------------------------------------
    # files = tf.io.matching_files(os.path.join(tfrecords_dir, '%s-*' % subset))
    # shards = tf.data.Dataset.from_tensor_slices(files)
    # shards = shards.shuffle(tf.cast(tf.shape(files)[0], tf.int64))
    # shards = shards.repeat()
    # dataset = shards.interleave(tf.data.TFRecordDataset, cycle_length=4)
    # dataset = dataset.shuffle(buffer_size=8192)
    # parser = partial(
    #     _parse_fn, is_training=True if subset == 'train' else False)
    # dataset = dataset.apply(
    #     tf.data.experimental.map_and_batch(
    #         map_func=parser,
    #         batch_size=batch_size,
    #         num_parallel_calls=config.NUM_DATA_WORKERS))
    # dataset = dataset.prefetch(batch_size)

    # -------------------------------------------------------------------------
    # (AP)
    # -------------------------------------------------------------------------
    # Notes:
    # Apply batch() before map():
    #   https://www.tensorflow.org/guide/data_performance#vectorizing_mapping
    #   https://stackoverflow.com/questions/58014123/how-to-improve-data-input-pipeline-performance
    
    # 1. Randomly shuffle the entire data once using a MapReduce/Spark/Beam/etc. job to create a set of roughly equal-sized files ("shards")
    shards = tf.data.Dataset.from_tensor_slices(tfrecords)

    # 2. In each epoch:
    # i. Randomly shuffle the list of shard filenames, using Dataset.list_files(...).shuffle(num_shards).
    # shards = shards.shuffle(tf.shape(shards)[0]), seed=None)
    shards = shards.shuffle(len(tfrecords), seed=seed)  # shuffle files
    # shards = shards.repeat()

    # ii. Use dataset.interleave(lambda filename: tf.data.TextLineDataset(filename), cycle_length=N)
    #     to mix together records from N different shards.
    # If num_parallel_calls = tf.data.AUTOTUNE, the cycle_length argument identifies the maximum degree of parallelism
    dataset = shards.interleave(
        map_func=lambda x: tf.data.TFRecordDataset(x),
        cycle_length=n_concurrent_shards,
        block_length=None,  # defaults to 1
        num_parallel_calls=None,
        deterministic=None
    )

    # iii. Use dataset.shuffle(B) to shuffle the resulting dataset. Setting B might require some experimentation,
    #      but you will probably want to set it to some value larger than the number of records in a single shard.
    # dataset = dataset.shuffle(buffer_size=512)  # (ap) shuffles the examples in the relevant filenames
    dataset = dataset.shuffle(buffer_size=shuffle_size)  # (ap) shuffles the examples in the relevant filenames

    # (ap) recommended to use batch before map
    # dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # num_parallel_calls = tf.data.experimental.AUTOTUNE
    # num_parallel_calls = 64
    num_parallel_calls = 8
    dataset = dataset.map(
        map_func=partial(parse_fn, include_meta=include_meta, **parse_fn_kwargs),
        num_parallel_calls=num_parallel_calls,
        deterministic=None
    )

    # Apply batch after repeat (https://www.tensorflow.org/guide/data)
    if repeat:
        # dataset = dataset.repeat(epochs)
        dataset = dataset.repeat()
        
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # (ap)
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(prefetch)

    return dataset



def calc_class_weights(tfrecords,
                       class_weights_method="BY_TILE",
                       MANIFEST=None,
                       SLIDE_ANNOTATIONS=None,
                       MODEL_TYPE=None):
    ''' ... '''
    num_tiles = []
    categories = {}

    for filename in tfrecords:
        smp = filename.split("/")[-1][:-10]

        # Determine total number of tiles available in TFRecord
        tiles = MANIFEST[filename]["total"]

        # Get the category of the current sample
        category = SLIDE_ANNOTATIONS[smp]["outcome"] if MODEL_TYPE == "categorical" else 1

        if category not in categories.keys():
            categories.update({category: {"num_samples": 1,
                                          "num_tiles": tiles}})
        else:
            categories[category]["num_samples"] += 1
            categories[category]["num_tiles"] += tiles
        num_tiles += [tiles]

    if class_weights_method == "NONE":
        class_weight = None

    elif class_weights_method == "BY_SAMPLE":
        n_samples = len(tfrecords)
        n_classes = len(categories)
        bins = np.array([categories[c]["num_samples"] for c in categories.keys()])
        weights = n_samples / (n_classes * bins)
        class_weight = {c: value for c, value in zip(categories.keys(), weights)}

    elif class_weights_method == "BY_TILE":
        n_samples = np.array(num_tiles).sum()
        n_classes = len(categories)
        bins = np.array([categories[c]["num_tiles"] for c in categories.keys()])
        weights = n_samples / (n_classes * bins)
        class_weight = {c: value for c, value in zip(categories.keys(), weights)}

    return class_weight


def create_manifest(directory, n_files: Optional[int]=None):
    """
    __init__ --> _trainer --> get_manifest --> update_manifest_at_dir
    """
    # directory = tfr_dir
    tfr_files = list(directory.glob('*.tfrec*'))
    manifest_path = directory/"manifest.json"
    manifest = {}

    if manifest_path.exists():
        print('Loading existing manifest.')
        with open(manifest_path, 'r') as data_file:
            global_manifest = json.load(data_file)
            
        MANIFEST = manifest = global_manifest

    else:
        print('Creating manifest.')
        relative_tfrecord_paths = [f.name for f in tfr_files]

        # ap
        if n_files is None:
            n_files = len(relative_tfrecord_paths)
        else:
            relative_tfrecord_paths = relative_tfrecord_paths[:n_files]
            n_files = len(relative_tfrecord_paths)

        slide_names_from_annotations = [n.split('.tfr')[0] for n in relative_tfrecord_paths]

        for i, rel_tfr in enumerate(relative_tfrecord_paths):
            # print(f'processing {i}')
            tfr = str(directory/rel_tfr)  #join(directory, rel_tfr)

            manifest.update({rel_tfr: {}})
            try:
                raw_dataset = tf.data.TFRecordDataset(tfr)
            except Exception as e:
                print(f"Unable to open TFRecords file with Tensorflow: {str(e)}")
                
            print(f"\r\033[K + Verifying tiles in ({i+1} out of {n_files} tfrecords) {green(rel_tfr)}...", end="")
            total = 0
            
            try:
                for raw_record in raw_dataset:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    slide = example.features.feature['slide'].bytes_list.value[0].decode('utf-8')  # get the slide name
                    if slide not in manifest[rel_tfr]:
                        manifest[rel_tfr][slide] = 1
                    else:
                        manifest[rel_tfr][slide] += 1
                    total += 1
                    
            except tf.python.framework.errors_impl.DataLossError:
                print('\r\033[K', end="")
                log.error(f"Corrupt or incomplete TFRecord at {tfr}", 1)
                log.info(f"Deleting and removing corrupt TFRecord from manifest...", 1)
                del(raw_dataset)
                os.remove(tfr)
                del(manifest[rel_tfr])
                continue
                
            manifest[rel_tfr]['total'] = total
            print('\r\033[K', end="")

        del raw_dataset

        # Update manifest to use full paths to tfrecords
        # tfrecord_dir = str(tfr_dir)
        tfrecord_dir = str(directory)
        global_manifest = {}
        for record in manifest:
            global_manifest.update({os.path.join(tfrecord_dir, record): manifest[record]})

        MANIFEST = manifest = global_manifest

        print(f'Items in manifest {len(manifest)}')
        
        # Save manifest
        with open(manifest_path, "w") as data_file:
            json.dump(manifest, data_file, indent=1)

    return manifest
