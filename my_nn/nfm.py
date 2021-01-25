from copy import copy

import tensorflow as tf
from tensorflow import keras

from my_nn import LRRegModel
from my_nn.base import BaseModel


class BiInteraction(BaseModel):

    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_list_features: dict, fc_layers=(128,),
                 activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, output_dim=1, regularizer=None, **kwargs):
        super(BiInteraction, self).__init__(None, conti_embd_features, cate_features, cate_list_features, 'fm',
                                            regularizer=regularizer, **kwargs)

        self.output_dim = output_dim
        self.layer_list = [keras.layers.BatchNormalization()] if use_bn else []

        for h, units in enumerate(fc_layers):
            self.layer_list.append(
                tf.keras.layers.Dense(units=units, name=f'dense_h{h + 1}', kernel_regularizer=self.regularizer,
                                      bias_regularizer=self.regularizer))
            if use_bn:
                self.layer_list.append(keras.layers.BatchNormalization(name=f'bn_h{h + 1}'))
            self.layer_list.append(keras.layers.Activation(activation, name=f'acti_h{h + 1}'))
            if use_drop_out:
                self.layer_list.append(keras.layers.Dropout(drop_p, seed=42, name=f'dropout_h{h + 1}'))

        self.fl = tf.keras.layers.Flatten()

        self.output_layer = tf.keras.layers.Dense(units=self.output_dim)

    def call(self, inputs: dict, **kwargs):
        embd_vecs = super(BiInteraction, self).call(inputs)  # batch_size * n * embd_size
        embd_vecs_sum_square = tf.reduce_sum(embd_vecs, axis=1) ** 2
        embd_vecs_square_sum = tf.reduce_sum(embd_vecs ** 2, axis=1)
        outputs = 0.5 * (embd_vecs_sum_square - embd_vecs_square_sum)
        outputs = self.fl(outputs)
        for layer in self.layer_list:
            outputs = layer(outputs)
        return self.output_layer(outputs)  # batch_size * k


class NFMRegModel(tf.keras.Model):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True, drop_p=0.5, output_dim=1,
                 regularizer=None, **kwargs):
        super(NFMRegModel, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.conti_features = conti_features
        self.conti_embd_features = conti_embd_features
        self.cate_features = cate_features
        self.cate_list_features = cate_list_features
        self.regularizer = regularizer

        self.bi_interaction = BiInteraction(self.conti_embd_features, self.cate_features, self.cate_list_features,
                                            fc_layers, activation, use_bn, use_drop_out, drop_p, output_dim,
                                            regularizer=regularizer)

        self.lr = LRRegModel(self.conti_features, self.conti_embd_features, self.cate_features, self.cate_list_features,
                             self.output_dim, regularizer=regularizer)

    def call(self, inputs: dict):
        first_order = self.lr(inputs)
        second_order = self.bi_interaction(inputs)  # batch_size * k
        return first_order + second_order


class NFMClfModel(NFMRegModel):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True, drop_p=0.5, output_dim=1,
                 regularizer=None, **kwargs):
        super(NFMClfModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_list_features,
                                          fc_layers, activation, use_bn, use_drop_out, drop_p,
                                          output_dim, regularizer=regularizer, **kwargs)
        self.output_func = tf.nn.softmax if self.output_dim >= 2 else tf.nn.sigmoid

    def call(self, inputs: dict):
        return self.output_func(super(NFMClfModel, self).call(inputs))


def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r
