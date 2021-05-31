import tensorflow as tf

from .lr import LRRegModel
from my_nn.input_laysers import InputLayer


class FMLayer(tf.keras.Model):

    def __init__(self, input_layer: InputLayer, **kwargs):
        super(FMLayer, self).__init__(**kwargs)
        self.input_layer = input_layer
        self.fl = tf.keras.layers.Flatten()

    def call(self, inputs: dict, **kwargs):
        embd_vecs = self.input_layer(inputs, 'stack')  # batch_size * n * embd_size
        embd_vecs_square = tf.matmul(embd_vecs, embd_vecs, transpose_b=True)
        outputs = 0.5 * (tf.reduce_sum(tf.reshape(embd_vecs_square, (-1, embd_vecs.shape[1] ** 2)),
                                       axis=1) - tf.linalg.trace(embd_vecs_square))
        return self.fl(outputs)


class FMRegModel(tf.keras.Model):
    def __init__(self, input_layer: InputLayer, regularizer=None, **kwargs):
        super(FMRegModel, self).__init__(**kwargs)

        self.input_layer = input_layer
        self.regularizer = regularizer

        self.fm_layer = FMLayer(self.input_layer)

        conti_features_ = {}
        if self.input_layer.conti_features:
            conti_features_.update(self.input_layer.conti_features)
        if self.input_layer.conti_embd_features:
            conti_features_.update(self.input_layer.conti_embd_features)
        self.lr = LRRegModel(conti_features_, self.input_layer.cate_features, 1, regularizer=self.regularizer)

    def call(self, inputs: dict):
        first_order = self.lr(inputs)
        second_order = self.fm_layer(inputs)  # batch_size * 1
        return first_order + second_order


class FMClfModel(FMRegModel):
    def __init__(self, input_layer: InputLayer, regularizer=None, **kwargs):
        super(FMClfModel, self).__init__(input_layer, regularizer, **kwargs)

    def call(self, inputs: dict):
        return tf.nn.sigmoid(super(FMClfModel, self).call(inputs))
