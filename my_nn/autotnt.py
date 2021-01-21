from copy import copy

import tensorflow as tf
from tensorflow import keras

from my_nn import LRRegModel
from my_nn.base import BaseModel


class InteractingLayer(tf.keras.Model):

    def __init__(self, input_dim: int, output_dim: int, n_attention_head=2, n_layers=1):
        super(InteractingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_attention_head = n_attention_head
        self.n_layers = n_layers
        self.W_Q = [[tf.Variable(tf.random.truncated_normal(
            shape=(self.input_dim if l == 0 else self.output_dim * self.n_attention_head, self.output_dim)),
            name=f"query_layer{l + 1}_head{h + 1}", trainable=True) for h in
            range(self.n_attention_head)] for l in range(self.n_layers)]
        self.W_K = [[tf.Variable(tf.random.truncated_normal(
            shape=(self.input_dim if l == 0 else self.output_dim * self.n_attention_head, self.output_dim)),
            name=f"key_layer{l + 1}_head{h + 1}", trainable=True) for h in
            range(self.n_attention_head)] for l in range(self.n_layers)]
        self.W_V = [[tf.Variable(tf.random.truncated_normal(
            shape=(self.input_dim if l == 0 else self.output_dim * self.n_attention_head, self.output_dim)),
            name=f"value_layer{l + 1}_head{h + 1}", trainable=True) for h in
            range(self.n_attention_head)] for l in range(self.n_layers)]
        self.W_res = [tf.Variable(
            tf.random.truncated_normal(shape=(self.input_dim if l == 0 else self.output_dim * self.n_attention_head,
                                              self.output_dim * self.n_attention_head)),
            name=f"w_res_layer{l + 1}", trainable=True) for l in range(self.n_layers)]  # k, d*n_attention_head
        self.fl = tf.keras.layers.Flatten()

    def auto_interacting(self, embed_map, n_layer):
        """
        实现单层 AutoInt Interacting Layer
        @param embed_map: 输入的embedding feature map, (?, n_feats, n_dim)
        @param output_dim: Q,K,V映射后的维度
        @param n_attention_head: multi-head attention的个数
        """

        # 存储多个self-attention的结果
        attention_heads = []

        for i in range(self.n_attention_head):
            # 映射到d维空间
            embed_q = tf.matmul(embed_map, self.W_Q[n_layer][i])  # ?, n_feats, output_dim
            embed_k = tf.matmul(embed_map, self.W_K[n_layer][i])  # ?, n_feats, output_dim
            embed_v = tf.matmul(embed_map, self.W_V[n_layer][i])  # ?, n_feats, output_dim

            # 计算attention
            energy = tf.matmul(embed_q, tf.transpose(embed_k, [0, 2, 1]))  # ?, n_feats, n_feats
            attention = tf.nn.softmax(energy)  # ?, n_feats, n_feats

            attention_output = tf.matmul(attention, embed_v)  # ?, n_feats, output_dim
            attention_heads.append(attention_output)

        # 2.concat multi head
        multi_attention_output = tf.concat(attention_heads, axis=-1)  # ?, n_feats, n_attention_head*d

        # 3.ResNet
        output = tf.nn.relu(
            multi_attention_output + tf.matmul(embed_map,
                                               self.W_res[n_layer]))  # ?, n_feats, output_dim * n_attention_head

        return output

    def call(self, x0):
        xl = x0
        result = []
        for l in range(self.n_layers):
            xl = self.auto_interacting(xl, l)
            result.append(xl)

        return self.fl(tf.concat(result, axis=-1))  # ?, n_feats * output_dim * n_attention_head * n_layers


class AutoIntRegModel(BaseModel):
    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_list_features: dict, output_dim=1,
                 interacting_input_dim=8, interacting_output_dim=6, n_attention_head=2, interacting_n_layers=1,
                 fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, **kwargs):
        super(AutoIntRegModel, self).__init__(None, conti_embd_features, cate_features, cate_list_features,
                                              cate_list_concat_way='mean', **kwargs)

        self.output_dim = output_dim
        self.n_features = 0
        self.n_features += len(self.conti_embd_features) if self.conti_embd_features else 0
        self.n_features += len(self.cate_features) if self.cate_features else 0
        self.n_features += len(self.cate_list_features) if self.cate_list_features else 0
        self.interacting_output_dim = interacting_output_dim
        self.n_attention_head = n_attention_head
        self.interacting_n_layers = interacting_n_layers
        self.interacting_input_dim = interacting_input_dim  # embedding dim

        self.InteractingLayer = InteractingLayer(self.interacting_input_dim, self.interacting_output_dim,
                                                 self.n_attention_head, self.interacting_n_layers)
        self.layer_list = [keras.layers.BatchNormalization()] if use_bn else []

        for units in fc_layers:
            self.layer_list.append(tf.keras.layers.Dense(units=units))
            if use_bn:
                self.layer_list.append(keras.layers.BatchNormalization())
            self.layer_list.append(keras.layers.Activation(activation))
            if use_drop_out:
                self.layer_list.append(keras.layers.Dropout(drop_p, seed=42))

        self.output_layer = tf.keras.layers.Dense(units=self.output_dim)

    def call(self, inputs: dict):
        embd_vecs = super(AutoIntRegModel, self).call(inputs)
        result = tf.reshape(embd_vecs, (tf.shape(embd_vecs)[0], self.n_features, self.interacting_input_dim))
        result = self.InteractingLayer(result)
        for layer in self.layer_list:
            result = layer(result)
        return self.output_layer(result)


class AutoIntClfModel(AutoIntRegModel):
    def __init__(self, conti_embd_features: dict, cate_features: dict, cate_list_features: dict, output_dim=1,
                 interacting_input_dim=8, interacting_output_dim=6, n_attention_head=2, interacting_n_layers=1,
                 fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, **kwargs):
        super(AutoIntClfModel, self).__init__(conti_embd_features, cate_features, cate_list_features, output_dim,
                                              interacting_input_dim, interacting_output_dim, n_attention_head,
                                              interacting_n_layers, fc_layers, activation, use_bn, use_drop_out,
                                              drop_p, **kwargs)
        self.output_func = tf.nn.softmax if self.output_dim >= 2 else tf.nn.sigmoid

    def call(self, inputs: dict):
        return self.output_func(super(AutoIntClfModel, self).call(inputs))


def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r
