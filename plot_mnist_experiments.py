from pathlib import Path

import fire
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm 

from modules.utils import load_json
from modules.models import ArcModel


def main(out_dir="media", img=True, 
        config1="configs/test_arc.json", 
        config2="configs/test_arc_2.json", 
        config3="configs/test_arc_3.json",
        config4="configs/test_arc_4.json",
        fig_name="mnist_experiments"):
    """
    Load the four trained models and save plotted embeddings.
    """

    # Load test dataset
    _, (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test/255.

    # Load configs
    configs = [config1, config2, config3, config4]
    configs = [load_json(config) for config in configs]
    
    # Load models
    print("Loading models and weights...")
    models = [ArcModel(config=config, training=False) for config in configs]

    # Load model weights
    for k, config in enumerate(configs):
        weights = tf.train.latest_checkpoint('./checkpoints/' + config['ckpt_name'])
        models[k].load_weights(weights)
        
    print("Computing and plotting embeddings...")
    # Compute embeddings
    embeddings = [model(x_test).numpy() for model in models]

    # Plot untrained and trained encodings
    class_names = np.array(['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress',
                            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle boot'])

    plt.figure(figsize=(14, 14))
    cmap = cm.get_cmap('jet', 10)

    for k in range(1,5): 
        plt.subplot(2, 2, k)
        embds = embeddings[k-1]
        margin = configs[k-1]["margin"]
        logist_scale = configs[k-1]["logist_scale"]
        for i, class_label in enumerate(class_names):
            inx = np.where(y_test == i)[0]
            plt.scatter(embds[inx, 0], embds[inx, 1],
                        color=cmap(i), label=class_label, alpha=0.7)
        plt.xlabel('$z_1$', fontsize=16) 
        plt.ylabel('$z_2$', fontsize=16)
        plt.title(f"m = {margin}, s = {logist_scale}")
        plt.legend()

    if img:
        out_path = Path(out_dir) / (fig_name+".png")
    else:
        out_path = Path(out_dir) / (fig_name+".pdf")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    return


if __name__=='__main__':
    fire.Fire(main)