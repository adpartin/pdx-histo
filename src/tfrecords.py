import tensorflow as tf
from config import cfg


FEA_SPEC = {
    'slide':     tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None)
}

FEA_SPEC_NEW = {
    'slide':     tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'image_raw': tf.io.FixedLenFeature(shape=[], dtype=tf.string, default_value=None)
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
