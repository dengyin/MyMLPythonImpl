from copy import copy

import tensorflow as tf
from tensorflow import keras

from my_nn.attention import MultiHeadAttention
from my_nn.base import BaseModel


class InteractingLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim: int, n_attention_head=2, n_layers=1, regularizer=tf.keras.regularizers.L2(0.01)):
        super(InteractingLayer, self).__init__()
        self.output_dim = output_dim
        self.n_attention_head = n_attention_head
        self.n_layers = n_layers
        self.regularizer = regularizer

        assert self.output_dim % self.n_attention_head == 0

        self.multi_head_attention = [
            MultiHeadAttention(self.output_dim, self.n_attention_head, regularizer=self.regularizer) for _ in
            range(self.n_layers)]
        self.fl = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self.n_feature = input_shape[1]
        self.W_res = [tf.keras.layers.Dense(units=self.n_feature * self.output_dim, use_bias=False,
                                            name=f'w_res_layer{l + 1}') for l in
                      range(self.n_layers)]

    def call(self, x0):
        xl = x0
        result = []
        for l in range(self.n_layers):
            x_res = self.W_res[l](self.fl(xl))
            xl = self.multi_head_attention[l](xl, xl, xl, None, False)
            x_res = tf.reshape(x_res, shape=[-1, self.n_feature, self.output_dim])
            xl = tf.nn.relu(xl + x_res)
            result.append(xl)

        return self.fl(tf.concat(result, axis=-1))  # ?, n_feats * output_dim * n_layers


class AutoIntRegModel(BaseModel):
    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_list_features: dict, output_dim=1,
                 interacting_output_dim=6, n_attention_head=2, interacting_n_layers=1,
                 fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, cate_list_concat_way='mean', regularizer=tf.keras.regularizers.L2(0.01), **kwargs):
        super(AutoIntRegModel, self).__init__(None, conti_embd_features, cate_features, cate_list_features,
                                              cate_list_concat_way, regularizer=regularizer, **kwargs)

        self.output_dim = output_dim
        self.interacting_output_dim = interacting_output_dim
        self.n_attention_head = n_attention_head
        self.interacting_n_layers = interacting_n_layers
        self.cate_list_concat_way = cate_list_concat_way
        self.features = {}
        if self.conti_embd_features:
            self.features.update(self.conti_embd_features)
        if self.cate_features:
            self.features.update(self.cate_features)
        if self.cate_list_features:
            self.features.update(self.cate_list_features)

        self.InteractingLayer = InteractingLayer(self.interacting_output_dim, self.n_attention_head,
                                                 self.interacting_n_layers, regularizer=self.regularizer)
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

        self.embd_dim = 0
        for key in self.features.keys():
            self.embd_dim = self.features.get(key).get('output_dim') if self.features.get(key).get(
                'output_dim') else 0
            self.embd_dim = self.features.get(key).get('units') if self.features.get(key).get(
                'units') else self.embd_dim
            if self.embd_dim > 0:
                break

        self.output_layer = tf.keras.layers.Dense(units=self.output_dim)

    def call(self, inputs: dict):
        embd_vecs = super(AutoIntRegModel, self).call(inputs)
        result = embd_vecs if self.cate_list_concat_way == 'fm' else tf.reshape(embd_vecs, (
            tf.shape(embd_vecs)[0], len(self.features), self.embd_dim))
        result = self.InteractingLayer(result)
        for layer in self.layer_list:
            result = layer(result)
        return self.output_layer(result)


class AutoIntClfModel(AutoIntRegModel):
    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_list_features: dict, output_dim=1,
                 interacting_output_dim=6, n_attention_head=2, interacting_n_layers=1,
                 fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True, drop_p=0.5,
                 cate_list_concat_way='mean', regularizer=tf.keras.regularizers.L2(0.01), **kwargs):
        super(AutoIntClfModel, self).__init__(conti_embd_features, cate_features, cate_list_features, output_dim,
                                              interacting_output_dim, n_attention_head,
                                              interacting_n_layers, fc_layers, activation, use_bn, use_drop_out,
                                              drop_p, cate_list_concat_way, regularizer, **kwargs)
        self.output_func = tf.nn.softmax if self.output_dim >= 2 else tf.nn.sigmoid

    def call(self, inputs: dict):
        return self.output_func(super(AutoIntClfModel, self).call(inputs))


def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r
