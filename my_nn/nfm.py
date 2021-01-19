from copy import copy

import tensorflow as tf
from tensorflow import keras

from my_nn.base import BaseModel


class BiInteraction(BaseModel):

    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_list_features: dict, fc_layers=(128,),
                 activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, output_dim=1, **kwargs):
        super(BiInteraction, self).__init__(None, conti_embd_features, cate_features, cate_list_features, 'fm',
                                            **kwargs)

        self.output_dim = output_dim
        self.layer_list = [keras.layers.BatchNormalization()] if use_bn else []

        for units in fc_layers:
            self.layer_list.append(tf.keras.layers.Dense(units=units))
            if use_bn:
                self.layer_list.append(keras.layers.BatchNormalization())
            self.layer_list.append(keras.layers.Activation(activation))
            if use_drop_out:
                self.layer_list.append(keras.layers.Dropout(drop_p, seed=42))

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
                 **kwargs):
        super(NFMRegModel, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.conti_features = conti_features
        self.conti_embd_features = conti_embd_features
        self.cate_features = cate_features
        self.cate_list_features = cate_list_features

        self.bi_interaction = BiInteraction(self.conti_embd_features, self.cate_features, self.cate_list_features,
                                            fc_layers, activation, use_bn, use_drop_out, drop_p, output_dim)

        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                setattr(self, name + '_first_order',
                        tf.keras.layers.Embedding(
                            **remove_key(self.cate_list_features[name], 'output_dim'),
                            output_dim=self.output_dim,
                            name=name + '_first_order'))

        if self.cate_features:
            for name in self.cate_features.keys():
                setattr(self, name + '_first_order',
                        tf.keras.layers.Embedding(
                            **remove_key(self.cate_features[name], 'output_dim'),
                            output_dim=self.output_dim,
                            name=name + '_first_order'
                        ))

        if self.conti_features:
            self.bn1 = keras.layers.BatchNormalization()
            self.dense1 = tf.keras.layers.Dense(units=self.output_dim, name='dense_first_order')
        if self.conti_embd_features:
            self.bn2 = keras.layers.BatchNormalization()
            self.dense2 = tf.keras.layers.Dense(units=self.output_dim, name='dense_first_order')
        self.fl = tf.keras.layers.Flatten()

    def call(self, inputs: dict):
        cats_feature = [name for name in
                        self.__dict__.keys() if name.endswith('_first_order')]
        first_order = self.fl(
            tf.reduce_sum(
                tf.concat(
                    [self.__dict__[name](inputs[name[:-len('_first_order')]]) for name in cats_feature],
                    axis=1
                )
                , axis=1)
        ) if cats_feature else 0

        if self.conti_features:
            first_order += self.fl(
                self.dense1(
                    self.bn1(
                        tf.reshape(
                            tf.stack(
                                [inputs[name] for name in self.conti_features.keys()],
                                axis=-1
                            ), (-1, len(self.conti_features.keys()))
                        )  # batch_size * n
                    )
                )
            )  # batch_size * k

        if self.conti_embd_features:
            first_order += self.fl(
                self.dense2(
                    self.bn2(
                        tf.reshape(
                            tf.stack(
                                [inputs[name] for name in self.conti_embd_features.keys()],
                                axis=-1
                            ), (-1, len(self.conti_embd_features.keys()))
                        )  # batch_size * n
                    )
                )
            )  # batch_size * k

        second_order = self.bi_interaction(inputs)  # batch_size * k

        return first_order + second_order


class NFMClfModel(NFMRegModel):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True, drop_p=0.5, output_dim=1,
                 **kwargs):
        super(NFMClfModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_list_features,
                                          fc_layers, activation, use_bn, use_drop_out, drop_p, output_dim, **kwargs)

    def call(self, inputs: dict):
        return tf.nn.softmax(super(NFMClfModel, self).call(inputs))


def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r
