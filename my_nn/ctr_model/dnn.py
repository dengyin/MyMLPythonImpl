import tensorflow as tf
from tensorflow import keras

from my_nn.ctr_model.com import fc_build
from my_nn.input_laysers import InputLayer


class Dnn(tf.keras.Model):
    def __init__(self, input_layer: InputLayer, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True,
                 drop_p=0.5, regularizer=None, **kwargs):
        super(Dnn, self).__init__(**kwargs)
        self.input_layer = input_layer
        self.seq_features_concat_way = seq_features_concat_way

        self.regularizer = regularizer

        self.layer_list = fc_build(use_bn, fc_layers, drop_p, use_drop_out, activation, regularizer)

        if self.input_layer.conti_features:
            self.fl = tf.keras.layers.Flatten()

    def call(self, inputs: dict, **kwargs):
        x = self.input_layer(inputs, self.seq_features_concat_way)
        if self.input_layer.conti_features:
            conti_x = self.fl(
                tf.reshape(
                    tf.stack(
                        [inputs[name] for name in self.input_layer.conti_features.keys()],
                        axis=-1
                    ), (-1, len(self.input_layer.conti_features.keys()))
                )  # batch_size * n
            )
            x = tf.concat([x, conti_x], axis=1)

        for layer in self.layer_list:
            x = layer(x)

        output = x
        return output


class DnnClfModel(Dnn):
    def __init__(self, input_layer: InputLayer, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True, drop_p=0.5, n_class=2, regularizer=None, **kwargs):
        super(DnnClfModel, self).__init__(input_layer, seq_features_concat_way, fc_layers, activation,
                                          use_bn, use_drop_out, drop_p, regularizer, **kwargs)

        self.output_func = keras.layers.Dense(units=1 if n_class == 2 else n_class,
                                              activation='sigmoid' if n_class == 2 else 'softmax')

    def call(self, inputs: dict, **kwargs):
        return self.output_func(
            tf.nn.relu(super(DnnClfModel, self).call(inputs))
        )


class DnnRegModel(Dnn):
    def __init__(self, input_layer: InputLayer, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True, drop_p=0.5, regularizer=None, **kwargs):
        super(DnnRegModel, self).__init__(input_layer, seq_features_concat_way, fc_layers, activation,
                                          use_bn, use_drop_out, drop_p, regularizer=regularizer, **kwargs)

        self.output_func = keras.layers.Dense(units=1)

    def call(self, inputs: dict, **kwargs):
        return self.output_func(
            tf.nn.relu(super(DnnRegModel, self).call(inputs))
        )
