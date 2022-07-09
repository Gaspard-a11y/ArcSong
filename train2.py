import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from modules.utils import load_json_dict
from modules.models import ArcFaceModel
from modules.losses import SoftmaxLoss


config = load_json_dict("configs/test_run.json")


model = ArcFaceModel(input_size=28,
                    channels=1, 
                    name='Backbone_test',
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
x_test = x_test/255.

x_train = x_train[..., np.newaxis]

y_train = tf.convert_to_tensor(y_train, tf.float32)
y_train = tf.expand_dims(y_train, axis=1)

train_dataset = tf.data.Dataset.from_tensor_slices(((x_train, y_train), y_train))
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


optimizer = Adam(learning_rate=config['learning_rate'])
loss_fn = SoftmaxLoss()


model.compile(optimizer=optimizer, loss=loss_fn)

checkpoint_callback = ModelCheckpoint(
    'checkpoints/' + config['ckpt_name'] + '/e_{epoch}.ckpt', 
    save_freq='epoch', 
    verbose=1, 
    save_weights_only=True)

callbacks = [checkpoint_callback]

model.fit(train_dataset,
            epochs=config['epochs'],
            callbacks=callbacks)


# TODO re-load the model, this time in test mode -> test.py
# TODO examine the embeddings
# TODO Re-do it all with Arcface head!
