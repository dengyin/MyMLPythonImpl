import tensorflow as tf

from .lr import LRRegModel
from my_nn.input_laysers import InputLayer


class FMLayer(InputLayer):

    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, **kwargs):
        super(FMLayer, self).__init__(None, conti_embd_features, cate_features, cate_seq_features,
                                      conti_embd_seq_features, 'stack', **kwargs)
        self.fl = tf.keras.layers.Flatten()

    def call(self, inputs: dict, **kwargs):
        embd_vecs = super(FMLayer, self).call(inputs)  # batch_size * n * embd_size
        embd_vecs_square = tf.matmul(embd_vecs, embd_vecs, transpose_b=True)
        outputs = 0.5 * (tf.reduce_sum(tf.reshape(embd_vecs_square, (-1, embd_vecs.shape[1] ** 2)),
                                       axis=1) - tf.linalg.trace(embd_vecs_square))
        return self.fl(outputs)


class FMRegModel(tf.keras.Model):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, regularizer=None, **kwargs):
        super(FMRegModel, self).__init__(**kwargs)

        self.conti_features = conti_features
        self.conti_embd_features = conti_embd_features
        self.cate_features = cate_features
        self.cate_seq_features = cate_seq_features
        self.conti_embd_seq_features = conti_embd_seq_features
        self.regularizer = regularizer

        self.fm_layer = FMLayer(self.conti_embd_features, self.cate_features, self.cate_seq_features,
                                self.conti_embd_seq_features)

        conti_features_ = {}
        if self.conti_features:
            conti_features_.update(self.conti_features)
        if self.conti_embd_features:
            conti_features_.update(self.conti_embd_features)
        self.lr = LRRegModel(conti_features_, self.cate_features, 1, regularizer=self.regularizer)

    def call(self, inputs: dict):
        first_order = self.lr(inputs)
        second_order = self.fm_layer(inputs)  # batch_size * 1
        return first_order + second_order


class FMClfModel(FMRegModel):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, regularizer=None, **kwargs):
        super(FMClfModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_seq_features,
                                         conti_embd_seq_features, regularizer, **kwargs)

    def call(self, inputs: dict):
        return tf.nn.sigmoid(super(FMClfModel, self).call(inputs))
