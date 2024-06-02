import tensorflow as tf
import tensorflow_transform as tft

import constants

_NUMERICAL_FEATURES = constants.NUMERICAL_FEATURES
_LABEL_KEY = constants.LABEL_KEY


def fill_in_missing(x):
    default_value = '' if x.dtype == tf.string or False else 0
    if type(x) == tf.SparseTensor:
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
        default_value)
    return tf.squeeze(x, axis=1)
    
def preprocessing_fn(inputs):
    outputs={}
    for key in _NUMERICAL_FEATURES:
        scaled = tft.scale_to_0_1(fill_in_missing(inputs[key]))
        outputs[constants.t_name(key)] = tf.reshape(scaled, [-1])

    outputs[constants.t_name(_LABEL_KEY)] = fill_in_missing(inputs[_LABEL_KEY])
    return outputs
