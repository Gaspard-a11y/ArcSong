import tensorflow as tf
import math


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, epsilon=0.01, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale
        self.epsilon = epsilon

    def build(self, input_shape):
        self.w = self.add_weight(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        # Necessary for stability
        # cos_t = tf.clip_by_value(cos_t, clip_value_min=-1, clip_value_max=1)
        cos_t = cos_t/(1.+self.epsilon)

        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        # = cos(theta+m)
        cos_mt = tf.subtract(cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')
        logists = tf.where(mask == 1., cos_mt, cos_t)
        # = cos(theta+m) on label index, cos(theta) elsewhere
        
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists
