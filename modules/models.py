from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input, Dense, Flatten, Dropout, Reshape,
    Activation, BatchNormalization, 
    Conv1D, Conv2D, MaxPool1D, MaxPool2D
)
                          

from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists
)


def Backbone(backbone_type='ImageCNN'):
    """Backbone Model"""
    def backbone(x_in):
        if backbone_type == 'ImageCNN':
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
            # From https://github.com/tae-jun/sample-cnn/blob/master/sample_cnn/model.py
            model = Sequential([# 59049 input
                Reshape([-1, 1], input_shape=x_in.shape[1:]), # 59049 X 1
                Conv1D(128, 3, strides=3, padding='valid', kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'), # 19683 X 128
                Conv1D(128, 3, padding='same', kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 6561 X 128
                Conv1D(128, 3, padding='same', kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 2187 X 128
                Conv1D(256, 3, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 729 X 256
                Conv1D(256, 3, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 243 X 256
                Conv1D(256, 3, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 81 X 256
                Conv1D(256, 3, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 27 X 256
                Conv1D(256, 3, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 9 X 256
                Conv1D(256, 3, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 3 X 256
                Conv1D(512, 3, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=3), # 1 X 512
                Conv1D(512, 1, padding='same',
                            kernel_initializer='he_uniform'),
                BatchNormalization(),
                Activation('relu'), # 1 X 512
                Dropout(rate=0.5),
                Flatten(),
                Dense(256, activation='sigmoid')
            ])
            return model(x_in)
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


def ArcModel(config=None,
            input_size=None, 
            data_dim=None, 
            channels=None, 
            num_classes=None, 
            name=None,
            margin=0.5, 
            logist_scale=64, 
            embd_shape=None,
            head_type='NormHead',
            training=True):

    # Parse config
    if config is not None:
        input_size=config['input_size']
        data_dim=config['data_dim']
        channels=config['channels']
        name=config['ckpt_name']
        num_classes=config['num_classes']
        head_type=config['head_type']
        embd_shape=config['embd_shape']
        margin = config['margin']
        logist_scale = config['logist_scale']

    # Arc Face Model
    if data_dim==2:
        backbone_type='ImageCNN'
        x = inputs = Input([input_size, input_size, channels], name='input_image')
    elif data_dim==1:
        backbone_type='SampleCNN'
        x = inputs = Input([input_size], name='input_song')
    else:
        raise TypeError('Only images (data_dim=2) and audio (data_dim=1) are supported')
    
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
