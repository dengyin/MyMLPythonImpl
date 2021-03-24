import tensorflow as tf
from tensorflow import keras

from my_nn.input_laysers import InputLayer


class Dnn(InputLayer):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True,
                 drop_p=0.5, regularizer=None, **kwargs):
        super(Dnn, self).__init__(conti_features, conti_embd_features, cate_features, cate_seq_features,
                                  conti_embd_seq_features,
                                  seq_features_concat_way, **kwargs)
        self.regularizer = regularizer

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

        if self.conti_features:
            self.fl = tf.keras.layers.Flatten()

    def call(self, inputs: dict, **kwargs):
        x = super(Dnn, self).call(inputs)
        if self.conti_features:
            conti_x = self.fl(
                tf.reshape(
                    tf.stack(
                        [inputs[name] for name in self.conti_features.keys()],
                        axis=-1
                    ), (-1, len(self.conti_features.keys()))
                )  # batch_size * n
            )
            x = tf.concat([x, conti_x], axis=1)

        for layer in self.layer_list:
            x = layer(x)

        output = x
        return output


class DnnClfModel(Dnn):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True, drop_p=0.5, n_class=2, regularizer=None, **kwargs):
        super(DnnClfModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_seq_features,
                                          conti_embd_seq_features, seq_features_concat_way, fc_layers, activation,
                                          use_bn, use_drop_out, drop_p, regularizer, **kwargs)

        self.output_func = keras.layers.Dense(units=1 if n_class == 2 else n_class,
                                              activation='sigmoid' if n_class == 2 else 'softmax')

    def call(self, inputs: dict, **kwargs):
        return self.output_func(
            tf.nn.relu(super(DnnClfModel, self).call(inputs))
        )


class DnnRegModel(Dnn):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True, drop_p=0.5, regularizer=None, **kwargs):
        super(DnnRegModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_seq_features,
                                          conti_embd_seq_features, seq_features_concat_way, fc_layers, activation,
                                          use_bn, use_drop_out, drop_p, regularizer=regularizer, **kwargs)

        self.output_func = keras.layers.Dense(units=1)

    def call(self, inputs: dict, **kwargs):
        return self.output_func(
            tf.nn.relu(super(DnnRegModel, self).call(inputs))
        )
