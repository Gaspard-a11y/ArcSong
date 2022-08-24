import os
import shutil
from pathlib import Path

import fire
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from modules.utils import load_json
from modules.models import ArcModel
from modules.dataset import get_fashion_mnist_train_dataset, get_MSD_train_dataset
from modules.losses import SoftmaxLoss


def main(network_config=None, dataset_config=None, from_scratch=True, debug=False):
    # Parse inputs
    from_scratch = (from_scratch==1)
    debug = (debug==1)

    print("Loading configs ...")
    config = load_json(network_config)
    
    if dataset_config is not None:
        dataset_config = load_json(dataset_config)

    # Sanity check
    assert config["input_size"] == dataset_config["input_size"], "Make sure input_size matches between configs"
    assert config["num_classes"] == dataset_config["num_classes"], "Make sure num_classes matches between configs"

    ### Load full model
    model = ArcModel(config=config, training=True)
    model.summary(line_length=80)

    ckpt_path = Path('checkpoints/') / config['ckpt_name']
    if os.path.isdir(ckpt_path):
        # Found a previous checkpoint
        if from_scratch:
            # Cleanup
            shutil.rmtree(ckpt_path)
        else:
            # Load previous weigths
            previous_weights = tf.train.latest_checkpoint(ckpt_path)
            model.load_weights(previous_weights)


    ### Load dataset
    batch_size=config['batch_size']
    if config['data_dim']==2:
        train_dataset = get_fashion_mnist_train_dataset(shuffle=True, buffer_size=1000)
    elif config['data_dim']==1:
        train_dataset = get_MSD_train_dataset(dataset_config)
    else:
        raise TypeError('Only images (data_dim=2) and audio (data_dim=1) are supported')
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    ### Train
    learning_rate = tf.constant(config['learning_rate'])
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = SoftmaxLoss()

    if debug==False:
        # Automatic fitting
        model.compile(optimizer=optimizer, loss=loss_fn)

        checkpoint_callback = ModelCheckpoint(
            ckpt_path / 'e_{epoch}.ckpt', 
            save_freq='epoch', 
            verbose=1,
            save_weights_only=True)

        # TODO add other callbacks?
        callbacks = [checkpoint_callback]

        model.fit(train_dataset,
                    epochs=config['epochs'],
                    callbacks=callbacks)
    
    else:
        # Manual loop
        for epoch in range(1, config['epochs']+1):
            print(f"====== Begin epoch {epoch} / {config['epochs']} ======")
            
            for step, (inputs, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logist = model(inputs, training=True)
                    loss = loss_fn(labels, logist)
                
                #TODO delete this debug
                if step%5==0:
                    print(f"Epoch: {epoch}, step: {step}, loss: {loss}")  
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
            print(f"End of epoch {epoch}, saving weights...")
            model.save_weights(ckpt_path / f"e_{epoch}.ckpt")
        
        print("============ Training done! ============")


if __name__=='__main__':
    fire.Fire(main)