from copy import copy

import tensorflow as tf
from tensorflow import keras

from my_nn.base import BaseModel


class FMLayer(BaseModel):

    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_list_features: dict, **kwargs):
        super(FMLayer, self).__init__(None, conti_embd_features, cate_features, cate_list_features, 'fm', **kwargs)
        self.fl = tf.keras.layers.Flatten()

    def call(self, inputs: dict, **kwargs):
        embd_vecs = super(FMLayer, self).call(inputs)  # batch_size * n * embd_size
        embd_vecs_square = tf.matmul(embd_vecs, embd_vecs, transpose_b=True)
        outputs = 0.5 * (tf.reduce_sum(tf.reshape(embd_vecs_square, (-1, embd_vecs.shape[1] ** 2)),
                                       axis=1) - tf.linalg.trace(embd_vecs_square))
        return self.fl(outputs)


class FMRegModel(tf.keras.Model):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 **kwargs):
        super(FMRegModel, self).__init__(**kwargs)

        self.conti_features = conti_features
        self.conti_embd_features = conti_embd_features
        self.cate_features = cate_features
        self.cate_list_features = cate_list_features

        self.fm_layer = FMLayer(self.conti_embd_features, self.cate_features, self.cate_list_features)

        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                setattr(self, name + '_first_order',
                        tf.keras.layers.Embedding(
                            **remove_key(self.cate_list_features[name], 'output_dim'),
                            output_dim=1,
                            name=name + '_first_order'))

        if self.cate_features:
            for name in self.cate_features.keys():
                setattr(self, name + '_first_order',
                        tf.keras.layers.Embedding(
                            **remove_key(self.cate_features[name], 'output_dim'),
                            output_dim=1,
                            name=name + '_first_order'
                        ))

        if self.conti_features:
            self.bn1 = keras.layers.BatchNormalization()
            self.dense1 = tf.keras.layers.Dense(units=1, name='dense_first_order')
        if self.conti_embd_features:
            self.bn2 = keras.layers.BatchNormalization()
            self.dense2 = tf.keras.layers.Dense(units=1, name='dense_first_order')
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
            )

        if self.conti_embd_features:
            first_order += self.fl(
                self.dense2(
                    self.bn2(
                        tf.reshape(
                            tf.stack(
                                [inputs[name] for name in self.conti_embd_features.keys()],
                                axis=-1
                            ), (-1, len(self.conti_features.keys()))
                        )  # batch_size * n
                    )
                )
            )

        second_order = self.fm_layer(inputs)  # batch_size * 1

        return first_order + second_order


class FMClfModel(FMRegModel):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 **kwargs):
        super(FMClfModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_list_features,
                                         **kwargs)

    def call(self, inputs: dict):
        return tf.nn.sigmoid(super(FMClfModel, self).call(inputs))


def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r
