from pathlib import Path

import fire
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm 

from modules.utils import load_json
from modules.models import ArcModel


def main(out_dir="media", img=True):
    """
    Load the two trained models and save plots.
    """

    # Load test dataset
    _, (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test/255.

    # Load configs
    config_norm = load_json("configs/test_norm.json")
    config_arc = load_json("configs/test_arc.json")


    # Load models
    print("Loading models and weights...")
    model_norm = ArcModel(input_size=config_norm['input_size'],
                            data_dim=config_norm['data_dim'],
                            channels=1, 
                            name='mnist_norm',
                            num_classes=config_norm['num_classes'],
                            head_type=config_norm['head_type'],
                            embd_shape=config_norm['embd_shape'],
                            training=False)

    model_arc = ArcModel(input_size=config_arc['input_size'],
                            data_dim=config_arc['data_dim'],
                            channels=1, 
                            name='mnist_arc',
                            num_classes=config_arc['num_classes'],
                            head_type=config_arc['head_type'],
                            embd_shape=config_arc['embd_shape'],
                            training=False)

    # Load model weights
    ckpt_path_norm = tf.train.latest_checkpoint('./checkpoints/' + config_norm['ckpt_name'])
    if ckpt_path_norm is not None:
        model_norm.load_weights(ckpt_path_norm)
        
    ckpt_path_arc = tf.train.latest_checkpoint('./checkpoints/' + config_arc['ckpt_name'])
    if ckpt_path_arc is not None:
        model_arc.load_weights(ckpt_path_arc)

    # Compute embeddings
    embeddings_norm = model_norm(x_test).numpy()
    embeddings_arc = model_arc(x_test).numpy()

    print("Plotting embeddings...")
    # Plot untrained and trained encodings
    class_names = np.array(['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress',
                            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle boot'])

    plt.figure(figsize=(14, 7))
    cmap = cm.get_cmap('jet', 10)

    plt.subplot(1, 2, 1)
    for i, class_label in enumerate(class_names):
        inx = np.where(y_test == i)[0]
        plt.scatter(embeddings_norm[inx, 0], embeddings_norm[inx, 1],
                    color=cmap(i), label=class_label, alpha=0.7)
    plt.xlabel('$z_1$', fontsize=16) 
    plt.ylabel('$z_2$', fontsize=16)
    plt.title('Image encodings with Norm head')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, class_label in enumerate(class_names):
        inx = np.where(y_test == i)[0]
        plt.scatter(embeddings_arc[inx, 0], embeddings_arc[inx, 1],
                    color=cmap(i), label=class_label, alpha=0.7)
    plt.xlabel('$z_1$', fontsize=16) 
    plt.ylabel('$z_2$', fontsize=16)
    plt.title('Image encodings with Arc head')
    plt.legend()

    if img:
        out_path = Path(out_dir) / "mnist_encodings.png"
    else:
        out_path = Path(out_dir) / "mnist_encodings.pdf"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    return

if __name__=='__main__':
    fire.Fire(main)