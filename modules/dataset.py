import tensorflow as tf
import numpy as np


def get_fashion_mnist_dataset(one_hot=True, batch_size=64, shuffle=True, buffer_size=1000):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize images to [0, 1]
    x_train = x_train/255.
    x_test = x_test/255.

    # Add channel axis
    x_train = x_train[..., np.newaxis]

    if one_hot:
        # Convert labels to one-hot
        y_test = tf.one_hot(indices=y_test, depth=10).numpy()
        y_train = tf.one_hot(indices=y_train, depth=10).numpy()

    ## CREATE DATASET OBJECTS

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return (x_train, y_train), (x_test, y_test), train_dataset, test_dataset