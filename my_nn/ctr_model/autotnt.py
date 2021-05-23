import tensorflow as tf
from tensorflow import keras

from my_nn.Transformer import MultiHeadAttention
from my_nn.ctr_model.com import fc_build
from my_nn.input_laysers import InputLayer


class InteractingLayer(tf.keras.layers.Layer):

    def __init__(self, output_dim: int, n_attention_head=2, n_layers=1, regularizer=None):
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


class AutoIntRegModel(InputLayer):
    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, output_dim=1, interacting_output_dim=6, n_attention_head=2,
                 interacting_n_layers=1, fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, seq_features_concat_way='mean', regularizer=None, **kwargs):
        super(AutoIntRegModel, self).__init__(None, conti_embd_features, cate_features, cate_seq_features,
                                              conti_embd_seq_features, seq_features_concat_way, **kwargs)

        self.output_dim = output_dim
        self.interacting_output_dim = interacting_output_dim
        self.n_attention_head = n_attention_head
        self.interacting_n_layers = interacting_n_layers
        self.seq_features_concat_way = seq_features_concat_way
        self.regularizer = regularizer
        self.features = {}
        self.n_features = 0
        if self.conti_embd_features:
            self.features.update(self.conti_embd_features)
            self.n_features += len(self.conti_embd_features)
        if self.cate_features:
            self.features.update(self.cate_features)
            self.n_features += len(self.cate_features)
        if self.cate_seq_features:
            self.features.update(self.cate_seq_features)
        if self.conti_embd_seq_features:
            self.features.update(self.conti_embd_seq_features)
        if self.cate_seq_features or self.conti_embd_seq_features:
            self.n_features += 1

        self.InteractingLayer = InteractingLayer(self.interacting_output_dim, self.n_attention_head,
                                                 self.interacting_n_layers, regularizer=self.regularizer)
        self.layer_list = fc_build(use_bn, fc_layers, drop_p, use_drop_out, activation, regularizer)

        self.embd_dim = 0
        for key in self.features.keys():
            embd_dim = self.features.get(key).get('output_dim') if self.features.get(key).get(
                'output_dim') else 0
            embd_dim = self.features.get(key).get('units') if self.features.get(key).get(
                'units') else embd_dim
            if self.embd_dim < embd_dim:
                self.embd_dim = embd_dim

        self.output_layer = tf.keras.layers.Dense(units=self.output_dim)

    def call(self, inputs: dict):
        embd_vecs = super(AutoIntRegModel, self).call(inputs)
        result = embd_vecs if self.seq_features_concat_way == 'stack' else tf.reshape(embd_vecs, (
            tf.shape(embd_vecs)[0], self.n_features, self.embd_dim))
        result = self.InteractingLayer(result)
        for layer in self.layer_list:
            result = layer(result)
        return self.output_layer(result)


class AutoIntClfModel(AutoIntRegModel):
    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, output_dim=1, interacting_output_dim=6, n_attention_head=2,
                 interacting_n_layers=1, fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, seq_features_concat_way='mean', regularizer=None, **kwargs):
        super(AutoIntClfModel, self).__init__(conti_embd_features, cate_features, cate_seq_features,
                                              conti_embd_seq_features, output_dim, interacting_output_dim,
                                              n_attention_head,
                                              interacting_n_layers, fc_layers, activation, use_bn, use_drop_out,
                                              drop_p, seq_features_concat_way, regularizer, **kwargs)
        self.output_func = tf.nn.softmax if self.output_dim >= 2 else tf.nn.sigmoid

    def call(self, inputs: dict):
        return self.output_func(super(AutoIntClfModel, self).call(inputs))
