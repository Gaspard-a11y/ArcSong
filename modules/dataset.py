import os
from pathlib import Path

import tensorflow as tf
import numpy as np

from modules.utils import load_json

# Magic ?
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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

# Don't touch me anymore, used elsewhere
def _get_MSD_raw_dataset(local=True):
    """
    Process the folder containing the tfrecord files,
    Return a tf.data.TFRecordDataset object.
    Returns the WHOLE dataset, should NOT be used for training.

    :param local: running on local machine or on boden.ma.ic.ac.uk, 
    :type local: bool, optional, defaults to True
    :return: tf.data.TFRecordDataset object
    """
    if local==True:
        waveforms_path = Path("./data_tfrecord")
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


def build_lookup_table_from_dict(dico, name):
    keys = list(dico.keys())
    values = list(dico.values())
    
    if type(values[0]) == int:
        default_value=tf.constant(-1)
    elif type(values[0]) == str:
        default_value=tf.constant('Unknown')
    
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values)),
        default_value=default_value,
        name=name)
    return table


def build_lookup_table_from_list(li, name):
    keys = [k for k in range(len(li))]
    values = li
    
    if type(values[0]) == int:
        default_value=tf.constant(-1)
    elif type(values[0]) == str:
        default_value=tf.constant('Unknown')
    
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keys),
            values=tf.constant(values)),
        default_value=default_value,
        name=name)
    return table


def _get_MSD_raw_split_dataset(local=True, train_size=0.9):
    """
    Process the folder containing the tfrecord files,
    Return a tf.data.TFRecordDataset object.
    Returns the TRAINING dataset.

    :param local: running on local machine or on boden.ma.ic.ac.uk, 
    :type local: bool, optional, defaults to True
    :return: tf.data.TFRecordDataset object
    """
    if local==True:
        waveforms_path = Path("./data_tfrecord")
        filenames = [waveforms_path / f for f in os.listdir(waveforms_path) if f.endswith('tfrecord')]
    else:
        # Training on Imperial College's Boden server
        waveforms_path = "/srv/data/msd/tfrecords/waveform-complete"
        filenames = [waveforms_path +'/' + f for f in os.listdir(waveforms_path) if f.endswith('tfrecord')]
    
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


    stop = round(train_size*len(filenames))

    train_dataset = tf.data.TFRecordDataset(filenames[:stop])
    test_dataset = tf.data.TFRecordDataset(filenames[stop:])

    train_dataset = train_dataset.map(_waveform_parse_function)
    test_dataset = test_dataset.map(_waveform_parse_function)

    return train_dataset, test_dataset

## DATASET PROCESSING FUNCTIONS

def extract_audio_and_label(lookup_name, lookup_number):
    def fun(item):
        audio = item['audio']
        tid = item['tid'][0]
        artist_name = lookup_name.lookup(tid)
        label = lookup_number.lookup(artist_name)
        return audio, label
    return fun

def filter_classes(num_classes=None):
    def fun(_, label):
        return label < num_classes
    return fun

def random_crop(size=None):
    def fun(audio, label):
        audio = tf.image.random_crop(audio, [size])
        return audio, label
    return fun

def setup_dataset_for_training(audio, label):
    return ((audio, label), label)


def get_MSD_train_dataset(config=None):
    """
    Build an MSD dataset for training ArcSong. 
    Complete with data augmentation.

    :param local: running on local machine or on boden.ma.ic.ac.uk, 
    :param shuffle: bool, whether to shuffle the dataset,
    :param buffer_size: int, buffer size for shuffling, 
    :return: tf.data.TFRecordDataset object
    """

    # Parse config
    if config is not None:
        input_size = config['input_size']
        num_classes = config['num_classes']
        buffer_size = config['buffer_size']
        order_by_count = config['order_by_count']
        local = config['local']
        train_size = config["train_size"]

    # Get the metadata lookup tables
    trackID_to_artistName = load_json("data_echonest/track_id_to_artist_name.json")
    if order_by_count==1:
        artistName_to_artistNumber = load_json("data_tfrecord_x_echonest/artist_name_to_artist_number_by_count.json")
    else:
        artistName_to_artistNumber = load_json("data_tfrecord_x_echonest/artist_name_to_artist_number_by_length.json")

    trackID_to_artistName_table = build_lookup_table_from_dict(trackID_to_artistName, "trackID_to_artistName")
    artistName_to_artistNumber_table = build_lookup_table_from_dict(artistName_to_artistNumber, "artistName_to_artistNumber")

    # Process dataset
    dataset, _ = _get_MSD_raw_split_dataset(local=(local==1), train_size=train_size)
    dataset = dataset.map(extract_audio_and_label(trackID_to_artistName_table, artistName_to_artistNumber_table))
    dataset = dataset.filter(filter_classes(num_classes=num_classes))
    
    if order_by_count:
        dataset = dataset.map(random_crop(size=input_size))

    # else:
    # TODO : data augmentation

    dataset = dataset.map(setup_dataset_for_training)

    if buffer_size is not None:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    return dataset


def get_audio(audio, label):
    return audio

def get_label(audio, label):
    return label


def get_MSD_test_data(config=None, extra_classes=0):
    """
    Build an MSD dataset for training ArcSong. 
    Complete with data augmentation.

    :param local: running on local machine or on boden.ma.ic.ac.uk, 
    :param shuffle: bool, whether to shuffle the dataset,
    :param buffer_size: int, buffer size for shuffling, 
    :return: tf.data.TFRecordDataset object
    """

    # Parse config
    if config is not None:
        input_size = config['input_size']
        num_classes = config['num_classes']+extra_classes
        order_by_count = config['order_by_count']
        local = config['local']
        train_size = config["train_size"]

    # Get the metadata lookup tables
    trackID_to_artistName = load_json("data_echonest/track_id_to_artist_name.json")
    if order_by_count==1:
        artistName_to_artistNumber = load_json("data_tfrecord_x_echonest/artist_name_to_artist_number_by_count.json")
    else:
        artistName_to_artistNumber = load_json("data_tfrecord_x_echonest/artist_name_to_artist_number_by_length.json")

    trackID_to_artistName_table = build_lookup_table_from_dict(trackID_to_artistName, "trackID_to_artistName")
    artistName_to_artistNumber_table = build_lookup_table_from_dict(artistName_to_artistNumber, "artistName_to_artistNumber")

    # Process dataset
    _, dataset = _get_MSD_raw_split_dataset(local=(local==1), train_size=train_size)
    dataset = dataset.map(extract_audio_and_label(trackID_to_artistName_table, artistName_to_artistNumber_table))
    dataset = dataset.filter(filter_classes(num_classes=num_classes))
    
    if order_by_count:
        dataset = dataset.map(random_crop(size=input_size))
    # TODO else: order by discography length?

    audio_dataset = dataset.map(get_audio)
    label_dataset = dataset.map(get_label)

    x_test = np.array(list(audio_dataset.as_numpy_iterator()))
    y_test = np.array(list(label_dataset.as_numpy_iterator()))

    return x_test, y_test

