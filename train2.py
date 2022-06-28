import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss


# TODO Load json instead
config = {
    'epochs' : 10,
    'learning_rate' : 5e-3,
    'num_classes' : 10,
    'embd_shape': 2,
    'backbone_type' : 'Custom',
    'head_type' : 'NormHead'
}


model = ArcFaceModel(input_size=28,
                    backbone_type=config['backbone_type'],
                    num_classes=config['num_classes'],
                    head_type=config['head_type'],
                    embd_shape=config['embd_shape'],
                    training=True)
model.summary(line_length=80)


# TODO cleanup into modules/dataset.py
batch_size=64
shuffle=True
buffer_size=1000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train/255.
# x_test = x_test/255.

x_train = x_train[..., np.newaxis]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
# test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


optimizer = Adam(learning_rate=config['learning_rate'])
loss_fn = SoftmaxLoss()


model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(train_dataset,
            epochs=config['epochs'])


print("[*] training done!")


# TODO add ckpt saving
# TODO re-load the model, this time in test mode
# TODO examine the embeddings
# TODO Re-do it all with Arcface head!
