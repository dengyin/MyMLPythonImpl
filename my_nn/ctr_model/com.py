from copy import copy

import tensorflow as tf
from tensorflow import keras


def fc_build(use_bn, fc_layers, drop_p, use_drop_out, activation, regularizer):
    layer_list = [keras.layers.BatchNormalization()] if use_bn else []

    for h, units in enumerate(fc_layers):
        layer_list.append(
            tf.keras.layers.Dense(units=units, name=f'dense_h{h + 1}', kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer, activation=activation))
        if use_bn:
            layer_list.append(keras.layers.BatchNormalization(name=f'bn_h{h + 1}'))
        if use_drop_out:
            layer_list.append(keras.layers.Dropout(drop_p, seed=42, name=f'dropout_h{h + 1}'))

    return layer_list

def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r