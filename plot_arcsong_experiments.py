import os
from pathlib import Path

import fire
import numpy as np
import scipy as sp
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import auc

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
    
    C = np.zeros((num_classes, num_classes))
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
    
    plt.figure(figsize=(9,7))
    sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=30) 
    plt.title("Predictions on the test set (standard classification)")
    plt.xticks(rotation=30)
    plt.grid()

    out_path = Path(out_dir) / "confusion_matrix.png"
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

    plt.figure(figsize=(9,7))
    sns.heatmap(confusion_matrix_euclidean, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=30) 
    plt.title("Predictions on the test set (euclidean distance in the latent space)")
    plt.xticks(rotation=30)
    plt.grid()

    out_path = Path(out_dir) / "confusion_matrix_euclidean.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)


    plt.figure(figsize=(9,7))
    sns.heatmap(confusion_matrix_spherical, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=30) 
    plt.title("Predictions on the test set (spherical distance in the latent space)")
    plt.xticks(rotation=30)
    plt.grid()

    out_path = Path(out_dir) / "confusion_matrix_spherical.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    
    ### 2-PCA of ID vectors
    print("Saving 2-PCA of ID vectors...")

    def normalise(vectors):
        _, dim = vectors.shape
        vectors = vectors - np.mean(vectors, axis=0)
        vectors = np.divide(vectors, np.repeat(np.linalg.norm(vectors, ord=2, axis=1)[...,np.newaxis], dim, axis=1))
        return vectors    

    class_vectors_normalised = normalise(class_vectors)
    pca = PCA(n_components=2)
    vectors = pca.fit_transform(class_vectors_normalised)

    vectors = class_vectors_normalised
    plt.figure(figsize=(9,7))
    for k in range(len(vectors)):
        plt.plot([0, vectors[k][0]], [0, vectors[k][1]], label=labels[k])
        plt.plot(vectors[k][0], vectors[k][1],'o', c='black')
    plt.xlabel("2-PCA axis 1")
    plt.ylabel("2-PCA axis 2")
    plt.title("2-PCA of normalised identity vectors")
    plt.legend()

    out_path = Path(out_dir) / "2pca_identities.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    
    ### Binary classification
    
    binary_y_test_matrix = np.array(
        [[y_test[i]==y_test[j] for j in range(test_set_length)] for i in range(test_set_length)],
        dtype=np.int32)

    euclidean_embds_distance_matrix = np.zeros((test_set_length, test_set_length))
    spherical_embds_distance_matrix = np.zeros((test_set_length, test_set_length))

    for i in range(test_set_length):
        for j in range(i):
            euclidean_distance_value = euclidean_distance(embds[i], embds[j])
            spherical_distance_value = spherical_distance(embds[i], embds[j])
            euclidean_embds_distance_matrix[i][j] = euclidean_embds_distance_matrix[j][i] = euclidean_distance_value
            spherical_embds_distance_matrix[i][j] = spherical_embds_distance_matrix[j][i] = spherical_distance_value

    def return_stats(embds_distance_matrix, binary_y_test_matrix, threshold):
        binary_y_pred_matrix = (embds_distance_matrix < threshold).astype(np.int32)
        tn, fp, fn, tp = sklearn_confusion_matrix(
            np.ndarray.flatten(binary_y_test_matrix), 
            np.ndarray.flatten(binary_y_pred_matrix)).ravel()

        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)
        f1 = 2*tp/(2*tp+fp+fn)
        
        return [fpr, tpr, f1]

    ## Euclidean
    print("Saving euclidean analysis...")

    max_distance_euclidean = np.amax(euclidean_embds_distance_matrix)
    threshold_range_euclidean = np.linspace(start=0., stop=max_distance_euclidean, num=100)
    euclidean_stats = np.array([return_stats(euclidean_embds_distance_matrix, binary_y_test_matrix, threshold) for threshold in threshold_range_euclidean])
    euclidean_auroc_score = auc(euclidean_stats[:,0], euclidean_stats[:,1])

    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.plot(euclidean_stats[:,0], euclidean_stats[:,1], label=f"ROC curve, area={euclidean_auroc_score:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid()
    plt.legend()
    plt.title("ROC curve of the binary classification")

    plt.subplot(1, 2, 2)
    plt.plot(threshold_range_euclidean, euclidean_stats[:,2])
    plt.plot([threshold_range_euclidean[0], threshold_range_euclidean[-1]], 
            [np.amax(euclidean_stats[:,2]), np.amax(euclidean_stats[:,2])], 
            linestyle='--', label=f"Maximal F1-score: {np.amax(euclidean_stats[:,2]):.2f}")
    plt.xlabel("Distance threshold")
    plt.ylabel("F1-score")
    plt.grid()
    plt.legend()
    plt.title("F1 score as a function of the distance threshold")

    out_path = Path(out_dir) / "binary_classification_euclidean.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    ## Spherical
    print("Saving spherical analysis...")

    max_distance_spherical = np.amax(spherical_embds_distance_matrix)
    threshold_range_spherical = np.linspace(start=0., stop=max_distance_spherical, num=100)
    spherical_stats = np.array([return_stats(spherical_embds_distance_matrix, binary_y_test_matrix, threshold) for threshold in threshold_range_spherical])
    spherical_auroc_score = auc(spherical_stats[:,0], spherical_stats[:,1])

    plt.figure(figsize=(16, 7))

    plt.subplot(1, 2, 1)
    plt.plot(spherical_stats[:,0], spherical_stats[:,1], label=f"ROC curve, area={spherical_auroc_score:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid()
    plt.legend()
    plt.title("ROC curve of the binary classification")

    plt.subplot(1, 2, 2)
    plt.plot(threshold_range_spherical, spherical_stats[:,2])
    plt.plot([threshold_range_spherical[0], threshold_range_spherical[-1]], 
            [np.amax(spherical_stats[:,2]), np.amax(spherical_stats[:,2])], 
            linestyle='--', label=f"Maximal F1-score: {np.amax(spherical_stats[:,2]):.2f}")
    plt.xlabel("Distance threshold")
    plt.ylabel("F1-score")
    plt.grid()
    plt.legend()
    plt.title("F1 score as a function of the distance threshold")

    out_path = Path(out_dir) / "binary_classification_spherical.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    return


if __name__=='__main__':
    fire.Fire(main)

