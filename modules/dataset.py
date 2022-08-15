import os
from pathlib import Path

import tensorflow as tf
import numpy as np


### FASHION MNIST DATASET

def get_fashion_mnist_train_dataset(shuffle=True, buffer_size=1000):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train/255.

    x_train = x_train[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    
    train_dataset = tf.data.Dataset.from_tensor_slices(((x_train, y_train), y_train))
    
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size)
    
    return train_dataset

### MILLION SONG DATASET

# Don't touch me anymore
def _get_MSD_raw_dataset(local=True):
    """
    Process the folder containing the tfrecord files,
    Return a tf.data.TFRecordDataset object.

    :param local: running on local machine or on boden.ma.ic.ac.uk, 
    :type local: bool, optional, defaults to True
    :return: tf.data.TFRecordDataset object
    """
    if local==True:
        waveforms_path = Path("./msd_data")
    else:
        # Training on Imperial College's Boden server
        waveforms_path = Path("/srv/data/msd/tfrecords/waveform-complete")
    
    feature_description = {
        'audio': tf.io.VarLenFeature(tf.float32),
        'tags': tf.io.VarLenFeature(tf.string),
        'tid': tf.io.VarLenFeature(tf.string)
    }

    def _waveform_parse_function(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        parsed_features['audio'] = tf.sparse.to_dense(parsed_features['audio'])
        parsed_features['tid'] = tf.sparse.to_dense(parsed_features['tid'])
        parsed_features['tags'] = tf.sparse.to_dense(parsed_features['tags'])
        return parsed_features

    filenames = [waveforms_path / f for f in os.listdir(waveforms_path) if f.endswith('tfrecord')]
    dataset = tf.data.TFRecordDataset(filenames)
    
    dataset = dataset.map(_waveform_parse_function)
    
    return dataset


def get_MSD_train_dataset(input_size=59049, num_classes=10, shuffle=True, buffer_size=1000, local=True):
    """
    Build an MSD dataset for training ArcSong. 
    Complete with data augmentation.

    :param local: running on local machine or on boden.ma.ic.ac.uk, 
    :param shuffle: bool, whether to shuffle the dataset,
    :param buffer_size: int, buffer size for shuffling, 
    :return: tf.data.TFRecordDataset object
    """
    # The sample rate used is 16kHz
    dataset = _get_MSD_raw_dataset(local=local)

    # TODO processing to make sure the songs are the same length
    # TODO handle the labels = convert tid into a unique artist number
    # TODO filter out the mismatches

    return dataset


# TODO delete me
# dataset = get_MSD_train_dataset(local=True)
# 
# for data_example in dataset.take(1):
#     print(data_example['audio'])
#     print(data_example['tags'])
#     print(data_example['tid'])
# End TODO

# TODO functions to return test dataset
