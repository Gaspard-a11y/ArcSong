import os
from pathlib import Path

import tensorflow as tf
import numpy as np

from modules.utils import load_json


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

    # Get the metadata lookup tables
    trackID_to_artistName = load_json("msd_data/track_id_to_artist_name.json")
    artistName_to_artistNumber = load_json("msd_data/artist_name_to_artist_number.json")
    artistName_to_songCount = load_json("msd_data/artist_name_to_song_count.json")
    artist_list = load_json("msd_data/artist_list.json")
    
    trackID_to_artistName_table = build_lookup_table_from_dict(trackID_to_artistName, "trackID_to_artistName")
    artistName_to_artistNumber_table = build_lookup_table_from_dict(artistName_to_artistNumber, "artistName_to_artistNumber")
    # artistName_to_songCount_table = build_lookup_table_from_dict(artistName_to_songCount, "artistName_to_songCount")
    # artistNumber_to_artistsName_table = build_lookup_table_from_list(artist_list, "artist_list")

    def extract_audio_and_label(item):
        audio = item['audio']
        tid = item['tid'][0]
        artist_name = trackID_to_artistName_table.lookup(tid)
        label = artistName_to_artistNumber_table.lookup(artist_name)
        return audio, label

    def filter_classes(num_classes=1000):
        def fun(_, label):
            return label < num_classes
        return fun

    def setup_dataset_for_training(audio, label):
        return ((audio, label), label)
    
    dataset = _get_MSD_raw_dataset(local=True)
    dataset = dataset.map(extract_audio_and_label)
    dataset = dataset.filter(filter_classes(num_classes=1000))
    dataset = dataset.map(setup_dataset_for_training)
    
    # TODO processing to make sure the songs are the same length
    # TODO data augmentation

    return dataset


