import fire
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from modules.utils import load_json_dict
from modules.models import ArcModel
from modules.losses import SoftmaxLoss


def main(config="configs/test_norm.json", debug=False):

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

    # TODO cleanup into modules/dataset.py
    batch_size=config['batch_size']
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

    dataset_length = len(x_train) # FIXME unclean as there's won't always be an x_train 
    learning_rate = tf.constant(config['learning_rate'])
    optimizer = Adam(learning_rate=learning_rate)
    loss_fn = SoftmaxLoss()

    if debug==False:
        # Automatic fitting
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
    else:
        # Manual loop
        epoch, step = 1, 1
        train_dataset = iter(train_dataset)

        while epoch <= config['epochs']:
            inputs, labels = next(train_dataset)

            with tf.GradientTape() as tape:
                logist = model(inputs, training=True)
                reg_loss = tf.reduce_sum(model.losses)
                pred_loss = loss_fn(labels, logist)
                total_loss = pred_loss + reg_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            
            # TODO delete this debug 
            print(f"Step: {step}, Loss: {total_loss}")
            
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

#            if steps % 5 == 0:
#                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
#                print(verb_str.format(epoch, cfg['epochs'],
#                                      steps % steps_per_epoch,
#                                      steps_per_epoch,
#                                      total_loss.numpy(),
#                                      learning_rate.numpy()))
#
#                with summary_writer.as_default():
#                    tf.summary.scalar(
#                        'loss/total loss', total_loss, step=steps)
#                    tf.summary.scalar(
#                        'loss/pred loss', pred_loss, step=steps)
#                    tf.summary.scalar(
#                        'loss/reg loss', reg_loss, step=steps)
#                    tf.summary.scalar(
#                        'learning rate', optimizer.lr, step=steps)
#
#            if steps % cfg['save_steps'] == 0:
#                print('[*] save ckpt file!')
#                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
#                    cfg['sub_name'], epoch, steps % steps_per_epoch))

            step += 1
            epoch = (step*batch_size)//dataset_length + 1

        


if __name__=='__main__':
    fire.Fire(main)