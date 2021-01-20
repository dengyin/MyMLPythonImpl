from copy import copy

import tensorflow as tf
from tensorflow import keras

from my_nn.base import BaseModel


class LRRegModel(tf.keras.Model):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 output_dim=1, **kwargs):
        super(LRRegModel, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.conti_features = conti_features
        self.conti_embd_features = conti_embd_features
        self.cate_features = cate_features
        self.cate_list_features = cate_list_features

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
        result = self.fl(
            tf.reduce_sum(
                tf.concat(
                    [self.__dict__[name](inputs[name[:-len('_first_order')]]) for name in cats_feature],
                    axis=1
                )
                , axis=1)
        ) if cats_feature else 0

        if self.conti_features:
            result += self.fl(
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
            result += self.fl(
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

        return result


class LRClfModel(LRRegModel):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 output_dim=1, **kwargs):
        super(LRClfModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_list_features,
                                         output_dim, **kwargs)
        self.output_func = tf.nn.softmax if self.output_dim >= 2 else tf.nn.sigmoid

    def call(self, inputs: dict):
        return self.output_func(super(LRClfModel, self).call(inputs))


def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r
