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

# TODO function to return test dataset