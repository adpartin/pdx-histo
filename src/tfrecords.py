import tensorflow as tf
# from config import cfg
from src.config import cfg


FEA_SPEC = {
    'slide':     tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None)
}

FEA_SPEC_RNA = {
    'slide': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    'Sample': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'model': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'patient_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'specimen_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'sample_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'image_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    # 'ge_data': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'ge_data': tf.io.FixedLenFeature(shape=(976,), dtype=tf.float32),
}

FEA_SPEC_RNA_NEW = {
    'slide': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    'Sample': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'model': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'patient_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'specimen_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'sample_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'image_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    # 'ge_data': tf.io.FixedLenFeature(shape=(976,), dtype=tf.float32),
    'ge_data': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
}

FEA_SPEC_RSP = {
    'slide': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    'smp': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    'Sample': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'model': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'patient_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'specimen_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'sample_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'image_id': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    'ctype': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'csite': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'ctype_src': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'csite_src': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    'Drug1': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'NAME': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'CLEAN_NAME': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'ID': tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    'Response': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),

    # 'ge_data': tf.io.FixedLenFeature(shape=(GE_LEN,), dtype=tf.float32),
    # 'dd_data': tf.io.FixedLenFeature(shape=(DD_LEN,), dtype=tf.float32),
    'ge_data': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    'dd_data': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
}


FEA_SPEC_RSP_DRUG_PAIR = {
    "slide":     tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "image_raw": tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    "index":    tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "smp":      tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "Group":    tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "grp_name": tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    "tile_id":  tf.io.FixedLenFeature(shape=[], dtype=tf.int64),

    "Sample":      tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "model":       tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "patient_id":  tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "specimen_id": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "sample_id":   tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "image_id":    tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    "ctype":     tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "csite":     tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "ctype_src": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "csite_src": tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    "Drug1": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "Drug2": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "trt":   tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "aug":   tf.io.FixedLenFeature(shape=[], dtype=tf.string),

    "Response": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),

    # 'ge_data': tf.io.FixedLenFeature(shape=(GE_LEN,), dtype=tf.float32),
    # 'dd_data': tf.io.FixedLenFeature(shape=(DD_LEN,), dtype=tf.float32),
    "ge_data":  tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "dd1_data": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    "dd2_data": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
}


def original_tfr_names(label='299px_302um'):
    """ Return list of the original tfrecords file names. """

    # Obtain slide names for which we need to update the tfrecords
    directory = cfg.SF_TFR_DIR/label
    tfr_files = list(directory.glob('*.tfrec*'))
    print(f'A total of {len(tfr_files)} original tfrecords.')

    # Slide names from tfrecords
    slides = [s.name.split('.tfrec')[0] for s in tfr_files]
    return slides
