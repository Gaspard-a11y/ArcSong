from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # For display

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

## LOAD DATASET

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

# Add channel axis
x_train = x_train[..., np.newaxis]

# Convert labels to one-hot
y_test = tf.one_hot(indices=y_test, depth=10).numpy()
y_train = tf.one_hot(indices=y_train, depth=10).numpy()

## CREATE DATASET OBJECTS

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(1000)
train_dataset = train_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

## MODEL

def get_model(encoded_dim=2, classes_dim=10):
    encoder = Sequential([
        Conv2D(16, 5, activation='relu', input_shape=(28,28,1)),
        MaxPool2D(2),
        Conv2D(8, 5, activation='relu'),
        Flatten(),
        # Dense(64, activation=None),
        Dense(encoded_dim, activation=None)
    ])

    final_layer = Sequential([
        Dense(classes_dim, activation='softmax', input_shape=(encoded_dim,))
    ])

    model = Model(inputs=encoder.input, outputs=final_layer(encoder.output))
    return encoder, model

encoder, model = get_model(encoded_dim=2, classes_dim=10)

## COMPILE AND FIT THE MODEL
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=5e-3))
model.fit(train_dataset, epochs=10)

## DISPLAY EMBEDDINGS

class_names = np.array(['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress',
                        'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle boot'])

# Compute encodings after training
inx = np.random.choice(x_test.shape[0], 1000, replace=False)
trained_encodings = encoder(x_test[inx]).numpy()
trained_encoding_labels = y_test[inx]

# Un-one-hot labels
trained_encoding_labels_restacked = np.argmax(trained_encoding_labels, axis=1)

plt.figure(figsize=(8, 8))
cmap = cm.get_cmap('jet', 10)

for i, class_label in enumerate(class_names):
    inx = np.where(trained_encoding_labels_restacked == i)[0]
    plt.scatter(trained_encodings[inx, 0], trained_encodings[inx, 1],
                color=cmap(i), label=class_label, alpha=0.7)
plt.xlabel('$z_1$', fontsize=16) 
plt.ylabel('$z_2$', fontsize=16)
plt.title('Encodings after training')
plt.legend()
plt.savefig(Path("./media/embeddings.png"))
