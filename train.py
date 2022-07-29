import os
import shutil

import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from modules.utils import load_json_dict
from modules.models import ArcModel
from modules.losses import SoftmaxLoss


def main(config="configs/test_norm.json", debug=True):

    print(f"Loading config {config}")
    config = load_json_dict(config)

    model = ArcModel(input_size=config['input_size'],
                        channels=1, 
                        name='Backbone_test',
                        backbone_type=config['backbone_type'],
                        num_classes=config['num_classes'],
                        head_type=config['head_type'],
                        embd_shape=config['embd_shape'],
                        training=True)
    model.summary(line_length=80)

    ckpt_path = 'checkpoints/' + config['ckpt_name']
    if os.path.isdir(ckpt_path):
        # Previous checkpoints found, cleanup
        # TODO allow training from a previous checkpoint
        shutil.rmtree(ckpt_path)

    # TODO cleanup into modules/dataset.py
    batch_size=config['batch_size']
    shuffle=True
    buffer_size=1000

    (x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train/255.

    x_train = x_train[..., np.newaxis]

    y_train = tf.convert_to_tensor(y_train, tf.float32)
    y_train = tf.expand_dims(y_train, axis=1)

    train_dataset = tf.data.Dataset.from_tensor_slices(((x_train, y_train), y_train))
    # train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    learning_rate = tf.constant(config['learning_rate'])
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = SoftmaxLoss()

    if debug==False:
        # Automatic fitting
        model.compile(optimizer=optimizer, loss=loss_fn)

        checkpoint_callback = ModelCheckpoint(
            ckpt_path + '/e_{epoch}.ckpt', 
            save_freq='epoch', 
            verbose=1,
            save_weights_only=True)

        callbacks = [checkpoint_callback]

        model.fit(train_dataset,
                    epochs=config['epochs'],
                    callbacks=callbacks)
    else:
        # Manual loop
        for epoch in range(config['epochs']):
            print(f"====== Begin epoch {epoch} / {config['epochs']} ======")
            
            for step, (inputs, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logist = model(inputs, training=True)
                    loss = loss_fn(labels, logist)
                
                #TODO delete this debug
                if step%100==0:
                    print(f"Step: {step}, loss: {loss}")  
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
            print(f"End of epoch {epoch}, saving weights...")
            model.save_weights(ckpt_path+f"/e_{epoch}.ckpt")
        
        print("============ Training done! ============")


if __name__=='__main__':
    fire.Fire(main)