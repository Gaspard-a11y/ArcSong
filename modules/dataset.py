import os
from pathlib import Path

import tensorflow as tf
import numpy as np


def get_fashion_mnist_train_dataset(shuffle=True, buffer_size=1000):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train/255.

    x_train = x_train[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    
    train_dataset = tf.data.Dataset.from_tensor_slices(((x_train, y_train), y_train))
    
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size)
    
    return train_dataset


feature_description = {
    'audio': tf.io.VarLenFeature(tf.float32),
    'tags': tf.io.VarLenFeature(tf.string),
    'tid': tf.io.VarLenFeature(tf.string),
}


def _waveform_parse_function(example_proto):
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    parsed_features['audio'] = tf.sparse.to_dense(parsed_features['audio'])
    parsed_features['tid'] = tf.sparse.to_dense(parsed_features['tid'])
    parsed_features['tags'] = tf.sparse.to_dense(parsed_features['tags'])
    
    return parsed_features


def get_MSD_train_dataset(shuffle=True, buffer_size=1000, local=True):
    # The sample rate used is 16kHz
    if local==True:
        waveforms_path = Path("./msd")
    else:
        # Training on Imperial College's Boden server
        waveforms_path = Path("/srv/data/msd/tfrecords/waveform-complete")
    
    filenames = [f for f in os.listdir(waveforms_path) if f.endswith('tfrecord')]
    
    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(_waveform_parse_function)

    for data_example in dataset.take(1):
        print(data_example['audio'])
        print(data_example['tags'])
        print(data_example['tid'])
    # TODO finish

    return


# TODO functions to return test dataset
