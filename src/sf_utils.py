import sys
import csv
import numpy as np

import tensorflow as tf
assert tf.__version__ >= "2.0"

from config import cfg
from tfrecords import FEA_SPEC_RSP

BALANCE_BY_CATEGORY = 'BALANCE_BY_CATEGORY'
BALANCE_BY_PATIENT = 'BALANCE_BY_PATIENT'
NO_BALANCE = 'NO_BALANCE'


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


def _parse_tfrecord_function(record,
                             include_slidenames=True,
                             multi_image=False,
                             with_tile=True,
                             with_ge=False,
                             ge_scaler=None,
                             dd_scaler=None,
                             AUGMENT=True):
    ''' Parses raw entry read from TFRecord. '''
    #feature_description = tfrecords.FEATURE_DESCRIPTION if not multi_image else tfrecords.FEATURE_DESCRIPTION_MULTI
    ##feature_description = FEATURE_DESCRIPTION if not multi_image else FEATURE_DESCRIPTION_MULTI

    # if with_ge:
    #     feature_description = FEATURE_DESCRIPTION_RNA
    # elif multi_image:
    #     feature_description = FEATURE_DESCRIPTION_MULTI
    # else:
    #     feature_description = FEATURE_DESCRIPTION

    feature_description = FEA_SPEC_RSP

    features = tf.io.parse_single_example(record, feature_description)
    #slide = features['slide']
    smp = features[cfg.ID_NAME]

    #if MODEL_TYPE == 'linear':
    #    #label = [ANNOTATIONS_TABLES[oi].lookup(slide) for oi in range(NUM_CLASSES)]
    #    label = [ANNOTATIONS_TABLES[oi].lookup(smp) for oi in range(NUM_CLASSES)]
    #else:
    #    #label = ANNOTATIONS_TABLES[0].lookup(slide)
    #    label = ANNOTATIONS_TABLES[0].lookup(smp)
    # label = {outcome_header[0]: label}  # ap, or {'ctype': label}
    label = {'Response': tf.cast(features['Response'], tf.int64)}  # ap, or {'ctype': label}

    image_dict = {}
    
    if multi_image:
        #image_dict = {}
        inputs = [inp for inp in list(feature_description.keys()) if inp != 'slide']
        for i in inputs:
            image_string = features[i]
            #image = self._process_image(image_string, self.AUGMENT)
            image = _process_image(image_string, AUGMENT)
            image_dict.update({
                i: image
            })
        if include_slidenames:
            return image_dict, label, slide
        else:
            return image_dict, label	

    else:
        if with_tile:
            image_string = features['image_raw']      
            image = _process_image(image_string, AUGMENT)
            #image_dict = {'tile_image': image}
            image_dict.update({'tile_image': image})

            NUM_SLIDE_INPUT=None # (ap)
            if NUM_SLIDE_INPUT:
                def slide_lookup(s):
                    return SLIDE_INPUT_TABLE[s.numpy().decode('utf-8')]            
                slide_input_val = tf.py_function(func=slide_lookup, inp=[slide], Tout=[tf.float32] * NUM_SLIDE_INPUT)
                image_dict.update({'slide_input': slide_input_val})

        if with_ge:
            def decode_np_arr(tensor):
                # import ipdb; ipdb.set_trace()
                ge_data = np.frombuffer(tensor.numpy(), dtype=cfg.GE_DTYPE)
                return ge_data
            ge_data = tf.py_function(func=decode_np_arr, inp=[features['ge_data']],
                                     Tout=[tf.float32])
            ge_data = tf.cast(ge_data, tf.float32)
            # ge_data = _process_rna(ge_data)
            if ge_scaler is not None:
                ge_fea_mean = tf.constant(ge_scaler.mean_, tf.float32)
                ge_fea_scale = tf.constant(ge_scaler.scale_, tf.float32)
                ge_data = (ge_data - ge_fea_mean) / ge_fea_scale
            image_dict.update({'ge_data': ge_data})      
            
        if include_slidenames:
            return image_dict, label, slide
        else:
            return image_dict, label


def _process_image(image_string, augment):
    '''Converts a JPEG-encoded image string into RGB array, using normalization if specified.'''
    image = tf.image.decode_jpeg(image_string, channels = 3)

    #if self.normalizer:
    normalizer = None  # (ap)
    if normalizer:
        #image = tf.py_function(self.normalizer.tf_to_rgb, [image], tf.int32)
        print('(ap) not normalizing!')

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


def _interleave_tfrecords(tfrecords, batch_size, balance, finite, max_tiles=None,
                          min_tiles=None, include_slidenames=False, multi_image=False,
                          parse_fn=None, drop_remainder=False,
                          with_ge=False, with_tile=True,
                          MANIFEST=None, ANNOTATIONS_TABLES=None, SLIDE_ANNOTATIONS=None,
                          MODEL_TYPE=None):
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
        include_slidenames:		Bool, if True, dataset will include slidename (each entry will return image,
                                label, and slidename)
        multi_image:			Bool, if True, will read multiple images from each TFRecord record.
    '''
    # import ipdb; ipdb.set_trace()
    print(f"Interleaving {len(tfrecords)} tfrecords: finite={finite}, max_tiles={max_tiles}, min_tiles={min_tiles}")
    datasets = []
    datasets_categories = []
    num_tiles = []
    global_num_tiles = 0
    categories = {}
    categories_prob = {}
    categories_tile_fraction = {}

    if not parse_fn:
        parse_fn = _parse_tfrecord_function

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

    if include_slidenames:
        dataset_with_slidenames = dataset.map(
                partial(parse_fn, include_slidenames=True, multi_image=multi_image,
                        with_ge=with_ge, with_tile=with_tile),
                num_parallel_calls=32
        ) #tf.data.experimental.AUTOTUNE
        dataset_with_slidenames = dataset_with_slidenames.batch(batch_size, drop_remainder=drop_remainder)
    else:
        dataset_with_slidenames = None

    dataset = dataset.map(
        partial(parse_fn, include_slidenames=False, multi_image=multi_image,
                with_ge=with_ge, with_tile=with_tile),
        num_parallel_calls=8
    )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset, dataset_with_slidenames, global_num_tiles
