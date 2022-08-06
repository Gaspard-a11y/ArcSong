from keras import Model
from keras.layers import (Conv1D, MaxPool1D, BatchNormalization,
                          Dense, Dropout, Activation, Flatten, Reshape, Input)


def SampleCNN(n_outputs=256, activation='relu', kernel_initializer='he_uniform', dropout_rate=0.5, name='SampleCNN'):
    def sample_cnn(x_in):
        # 59049
        x = inputs = Input(x_in.shape[1:])
        x = Reshape([-1, 1])(x)
        # 59049 X 1
        x = Conv1D(128, 3, strides=3, padding='valid',
                 kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        # 19683 X 128
        x = Conv1D(128, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 6561 X 128
        x = Conv1D(128, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 2187 X 128
        x = Conv1D(256, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 729 X 256
        x = Conv1D(256, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 243 X 256
        x = Conv1D(256, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 81 X 256
        x = Conv1D(256, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 27 X 256
        x = Conv1D(256, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 9 X 256
        x = Conv1D(256, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 3 X 256
        x = Conv1D(512, 3, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPool1D(3)(x)
        # 1 X 512
        x = Conv1D(512, 1, padding='same',
                    kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        # 1 X 512
        x = Dropout(dropout_rate)(x)
        x = Flatten()(x)

        x = Dense(units=n_outputs, activation='sigmoid')(x)
        return Model(inputs, x, name=name)(x_in)
    
    return sample_cnn

        
