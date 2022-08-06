from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
    Conv2D, 
    MaxPool2D
)

from tensorflow.keras.models import Sequential

from keras.layers import (Conv1D, MaxPool1D, BatchNormalization,
                          Dense, Dropout, Activation, Flatten, Reshape, Input)


from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists
)

from .samplecnn import SampleCNN


def Backbone(backbone_type='Toy'):
    """Backbone Model"""
    def backbone(x_in):
        if backbone_type == 'Toy':
            model=Sequential([
                Conv2D(128, kernel_size=(5, 5), activation='relu', input_shape=x_in.shape[1:]),
                MaxPool2D((2, 2)),
                Conv2D(128, kernel_size=(5, 5), activation='relu'),
                MaxPool2D(pool_size=(2, 2)),
                Flatten(),
                Dense(256, activation='relu')
            ])
            return model(x_in)
        elif backbone_type == 'SampleCNN':
            return SampleCNN(n_outputs=256, 
                            activation='relu', 
                            kernel_initializer='he_uniform', 
                            dropout_rate=0.5, 
                            name='SampleCNN')(x_in)
        else:
            raise TypeError('Invalid backbone_type')
    return backbone


def OutputLayer(embd_shape, name='OutputLayer'):
    """Output Layer"""
    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(embd_shape)(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)
    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""
    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))
    return arc_head


def NormHead(num_classes, name='NormHead'):
    """Norm Head"""
    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes)(x)
        return Model(inputs, x, name=name)(x_in)
    return norm_head


def ArcModel(input_size=None, channels=1, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=2,
                 head_type='NormHead', backbone_type='Toy',
                 training=True):
    """Arc Face Model"""
    x = inputs = Input([input_size, input_size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type)(x)

    embds = OutputLayer(embd_shape)(x)

    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        elif head_type == 'NormHead':
            logist = NormHead(num_classes=num_classes)(embds)
        else:
            raise TypeError('Invalid head_type')
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)
