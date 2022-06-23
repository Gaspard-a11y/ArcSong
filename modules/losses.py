import tensorflow as tf

def SoftmaxLoss():
    """
    Returns callable loss function softmax_loss.
    """
    def softmax_loss(y_true, y_pred):
        """
        Compute soft_max loss of logit y_pred against label y_true.
        Warning: Logits must be unscaled as a softmax is performed internally.

        :param y_true: unscaler logits
        :param y_pred: shape [batch_size, num_classes] 
        :type y_pred: labels, shape [batch_size]
        :return: Categorical-Cross-Entropy-Loss(y_true, softmax(y_pred))
        """
        # y_true: sparse target
        # y_pred: logits
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)
    return softmax_loss
