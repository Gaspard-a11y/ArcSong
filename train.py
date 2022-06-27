from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # For display

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from modules.dataset import get_fashion_mnist_data, get_fashion_mnist_dataset

## DATASET

train_dataset, test_dataset = get_fashion_mnist_dataset(one_hot=True,
                                                        batch_size=64, 
                                                        shuffle=True, 
                                                        buffer_size=1000)


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

(x_train, y_train), (x_test, y_test) = get_fashion_mnist_data(one_hot=True)

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
