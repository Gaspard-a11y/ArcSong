from pathlib import Path

import numpy as np
import tensorflow as tf
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import Model

from modules.math import euclidean_distance, great_circle_distance
from modules.dataset import get_MSD_test_dataset, get_MSD_train_dataset
from modules.models import ArcModel
from modules.utils import load_json


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


out_dir="media"
### Load config

network_config = "configs/test_msd.json"
dataset_config = "configs/msd_config_local.json"

config = load_json(network_config)
dataset_config = load_json(dataset_config)

# Get artist names
if dataset_config["order_by_count"]==1:
    artist_names = load_json("data_tfrecord_x_echonest/artist_list_by_count.json") 
else :
    artist_names = load_json("data_tfrecord_x_echonest/artist_list_by_length.json") 

### Load test dataset

# Replace with test set eventually
# dataset = get_MSD_test_dataset(dataset_config)
# Dummy dataset with 15 examples
x_test = np.random.normal(0,1, (15, 59049))
y_test = np.random.randint(7,10, (15))+1

# Temporary fix
config["num_classes"] = 10
labels = artist_names[:config["num_classes"]]

### Classical supervised classification

# Load model
model = ArcModel(config=config, training=True)
ckpt_path = Path('checkpoints/') / config['ckpt_name']
previous_weights = tf.train.latest_checkpoint(ckpt_path)
model.load_weights(previous_weights)

# Compute y_pred
unscaled_logits = model((x_test, y_test))
logits = sp.special.softmax(unscaled_logits, axis=1)
y_pred = np.argmax(logits, axis=1)+1

# Plot and save confusion matrix
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

# Get class vectors
fc_matrix = model.layers[-1].weights[0]
class_vectors = fc_matrix.numpy().transpose()

# Load model
model = ArcModel(config=config, training=False)
ckpt_path = Path('checkpoints/') / config['ckpt_name']
previous_weights = tf.train.latest_checkpoint(ckpt_path)
model.load_weights(previous_weights)
embds = model(x_test).numpy()

# Compute y_pred
# TODO Cleanup
test_set_length = embds.shape[0]
num_classes = class_vectors.shape[0]

euclidean_distance_matrix = np.zeros((test_set_length, num_classes)) 
great_circle_distance_matrix = np.zeros((test_set_length, num_classes)) 

for i in range(test_set_length): # 15
    for j in range(num_classes): # 10
        euclidean_distance_matrix[i][j] = euclidean_distance(embds[i], class_vectors[j])
        great_circle_distance_matrix[i][j] = great_circle_distance(embds[i], class_vectors[j])

y_pred_euclidean = np.argmax(euclidean_distance_matrix, axis=1)
y_pred_great_circle = np.argmax(great_circle_distance_matrix, axis=1)

# Plot and save confusion matrices
confusion_matrix_euclidean = get_confusion_matrix(y_pred_euclidean, y_test, num_classes=config["num_classes"])
confusion_matrix_great_circle = get_confusion_matrix(y_pred_great_circle, y_test, num_classes=config["num_classes"])

plt.figure(figsize=(16,14))
sns.heatmap(confusion_matrix_euclidean, xticklabels=labels, yticklabels=labels)
plt.yticks(rotation=30) 
plt.title("Predictions on the test set (euclidean distance in the latent space)")
plt.xticks(rotation=30)
plt.grid()

out_path = Path(out_dir) / "arcsong_confusion_euclidean.png"
plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

plt.figure(figsize=(16,14))
sns.heatmap(confusion_matrix_great_circle, xticklabels=labels, yticklabels=labels)
plt.yticks(rotation=30) 
plt.title("Predictions on the test set (spherical distance in the latent space)")
plt.xticks(rotation=30)
plt.grid()

out_path = Path(out_dir) / "arcsong_confusion_spherical.png"
plt.savefig(out_path, bbox_inches='tight', pad_inches=0)


