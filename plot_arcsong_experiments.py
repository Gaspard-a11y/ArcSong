import os
from pathlib import Path

import fire
import numpy as np
import scipy as sp
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from modules.utils import load_json
from modules.models import ArcModel
from modules.dataset import get_MSD_test_data
from modules.math import euclidean_distance, spherical_distance


def get_confusion_matrix(y_pred, y_test, num_classes):
    """
    Return the confusion matrix C.
    C(i,j) = number of observations known to be 
    in class i and predicted to be in class j.
    """
    # Sanity check
    y_pred, y_test = np.array(y_pred), np.array(y_test)
    
    n = len(y_pred) # = len(y_test)
    C = np.zeros((num_classes,num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            C[i][j] = np.sum(np.logical_and(y_test==i, y_pred==j))
    return C


def main(out_dir="media", network_config = None, dataset_config = None):
    print("Loading config...")
    ### Load config
    config = load_json(network_config)
    dataset_config = load_json(dataset_config)
    
    # Define output dir
    out_dir = Path(out_dir) / config["ckpt_name"]

    # Make directory is not existing
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Sanity check
    assert config["num_classes"] == dataset_config["num_classes"]
    
    # Get artist names
    if dataset_config["order_by_count"]==1:
        artist_names = load_json("data_tfrecord_x_echonest/artist_list_by_count.json") 
    else :
        artist_names = load_json("data_tfrecord_x_echonest/artist_list_by_length.json") 

    ### Load test dataset
    print("Loading test data...")
    x_test, y_test = get_MSD_test_data(dataset_config)
    
    # Dummy dataset with 15 examples
    # x_test = np.random.normal(0,1, (15, 59049))
    # y_test = np.random.randint(7,10, (15))+1

    labels = artist_names[:config["num_classes"]]

    ### Classical supervised classification

    # Load model
    print("Loading full model...")
    model = ArcModel(config=config, training=True)
    ckpt_path = Path('checkpoints/') / config['ckpt_name']
    previous_weights = tf.train.latest_checkpoint(ckpt_path)
    model.load_weights(previous_weights)

    # Get class vectors for later
    fc_matrix = model.layers[-1].weights[0]
    class_vectors = fc_matrix.numpy().transpose()

    # Graph mode
    model = tf.function(model)

    # Compute y_pred
    print("Computing predictions...")
    unscaled_logits = model((x_test, y_test))
    logits = sp.special.softmax(unscaled_logits, axis=1)
    y_pred = np.argmax(logits, axis=1)

    # Plot and save confusion matrix
    print("Saving confusion matrix...")
    confusion_matrix = get_confusion_matrix(y_pred, y_test, num_classes=config["num_classes"])
    
    plt.figure(figsize=(16,14))
    sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=30) 
    plt.title("Predictions on the test set (standard classification)")
    plt.xticks(rotation=30)
    plt.grid()

    out_path = Path(out_dir) / "arcsong_confusion.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    ### Classification using latent distances

    # Load model
    print("Loading partial model...")
    model = ArcModel(config=config, training=False)
    ckpt_path = Path('checkpoints/') / config['ckpt_name']
    previous_weights = tf.train.latest_checkpoint(ckpt_path)
    model.load_weights(previous_weights)

    # Graph mode
    model = tf.function(model)

    # Compute y_pred
    print("Computing predictions...")
    embds = model(x_test).numpy()
    test_set_length = len(y_test)
    num_classes = config["num_classes"]

    euclidean_distance_matrix = np.zeros((test_set_length, num_classes)) 
    spherical_distance_matrix = np.zeros((test_set_length, num_classes)) 

    for i in range(test_set_length): # 15
        for j in range(num_classes): # 10
            euclidean_distance_matrix[i][j] = euclidean_distance(embds[i], class_vectors[j])
            spherical_distance_matrix[i][j] = spherical_distance(embds[i], class_vectors[j])

    y_pred_euclidean = np.argmax(euclidean_distance_matrix, axis=1)
    y_pred_spherical = np.argmax(spherical_distance_matrix, axis=1)

    # Plot and save confusion matrices
    print("Saving confusion matrices...")
    confusion_matrix_euclidean = get_confusion_matrix(y_pred_euclidean, y_test, num_classes=config["num_classes"])
    confusion_matrix_spherical = get_confusion_matrix(y_pred_spherical, y_test, num_classes=config["num_classes"])

    plt.figure(figsize=(16,14))
    sns.heatmap(confusion_matrix_euclidean, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=30) 
    plt.title("Predictions on the test set (euclidean distance in the latent space)")
    plt.xticks(rotation=30)
    plt.grid()

    out_path = Path(out_dir) / "arcsong_confusion_euclidean.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)


    plt.figure(figsize=(16,14))
    sns.heatmap(confusion_matrix_spherical, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=30) 
    plt.title("Predictions on the test set (spherical distance in the latent space)")
    plt.xticks(rotation=30)
    plt.grid()

    out_path = Path(out_dir) / "arcsong_confusion_spherical.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    
    return


if __name__=='__main__':
    fire.Fire(main)

