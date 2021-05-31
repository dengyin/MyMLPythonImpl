import tensorflow as tf
from tensorflow import keras

from .com import fc_build
from .lr import LRRegModel
from my_nn.input_laysers import InputLayer


class BiInteraction(tf.keras.Model):

    def __init__(self, input_layer: InputLayer, fc_layers=(128,),
                 activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, output_dim=1, regularizer=None, **kwargs):
        super(BiInteraction, self).__init__(**kwargs)

        self.input_layer = input_layer
        self.regularizer = regularizer

        self.output_dim = output_dim
        self.layer_list = fc_build(use_bn, fc_layers, drop_p, use_drop_out, activation, regularizer)

        self.fl = tf.keras.layers.Flatten()

        self.output_layer = tf.keras.layers.Dense(units=self.output_dim)

    def call(self, inputs: dict, **kwargs):
        embd_vecs = self.input_layer(inputs, 'stack')  # batch_size * n * embd_size
        embd_vecs_sum_square = tf.reduce_sum(embd_vecs, axis=1) ** 2
        embd_vecs_square_sum = tf.reduce_sum(embd_vecs ** 2, axis=1)
        outputs = 0.5 * (embd_vecs_sum_square - embd_vecs_square_sum)
        outputs = self.fl(outputs)
        for layer in self.layer_list:
            outputs = layer(outputs)
        return self.output_layer(outputs)  # batch_size * k


class NFMRegModel(tf.keras.Model):
    def __init__(self, input_layer: InputLayer, fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, output_dim=1, regularizer=None, **kwargs):
        super(NFMRegModel, self).__init__(**kwargs)

        self.input_layer = input_layer
        self.output_dim = output_dim
        self.regularizer = regularizer

        self.bi_interaction = BiInteraction(self.input_layer, fc_layers, activation, use_bn, use_drop_out,
                                            drop_p, output_dim, regularizer=regularizer)

        conti_features_ = {}
        if self.input_layer.conti_features:
            conti_features_.update(self.input_layer.conti_features)
        if self.input_layer.conti_embd_features:
            conti_features_.update(self.input_layer.conti_embd_features)
        self.lr = LRRegModel(conti_features_, self.input_layer.cate_features, output_dim, regularizer=self.regularizer)

    def call(self, inputs: dict):
        first_order = self.lr(inputs)
        second_order = self.bi_interaction(inputs)  # batch_size * k
        return first_order + second_order


class NFMClfModel(NFMRegModel):
    def __init__(self, input_layer: InputLayer, fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, output_dim=1, regularizer=None, **kwargs):
        super(NFMClfModel, self).__init__(input_layer, fc_layers, activation, use_bn, use_drop_out,
                                          drop_p, output_dim, regularizer, **kwargs)
        self.output_func = tf.nn.softmax if self.output_dim >= 2 else tf.nn.sigmoid

    def call(self, inputs: dict):
        return self.output_func(super(NFMClfModel, self).call(inputs))
