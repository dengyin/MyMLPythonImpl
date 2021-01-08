import tensorflow as tf
from tensorflow import keras

from my_nn.base import BaseModel


class Dnn(BaseModel):
    def __init__(self, conti_features: dict, cate_features: dict, cate_list_features: dict, **kwargs):
        super(Dnn, self).__init__(conti_features, cate_features, cate_list_features, **kwargs)

        self.bn1 = keras.layers.BatchNormalization()
        self.dense1 = tf.keras.layers.Dense(units=128, activation='relu')

        self.bn2 = keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(units=128)

        self.fl = tf.keras.layers.Flatten()

    def call(self, inputs: dict, **kwargs):
        x = self.fl(super(Dnn, self).call(inputs))
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

        output = self.dense1(self.bn1(x))
        output = self.dense2(self.bn2(output))
        return output


class DnnClfModel(Dnn):
    def __init__(self, conti_features: dict, cate_features: dict, cate_list_features: dict, n_class, **kwargs):
        super(DnnClfModel, self).__init__(conti_features, cate_features, cate_list_features, **kwargs)

        self.output_func = keras.layers.Dense(units=1 if n_class == 2 else n_class,
                                              activation='sigmoid' if n_class == 2 else 'softmax')

    def call(self, inputs: dict, **kwargs):
        return self.output_func(
           tf.nn.relu(super(DnnClfModel, self).call(inputs))
        )


class DnnRegModel(Dnn):
    def __init__(self, conti_features: dict, cate_features: dict, cate_list_features: dict, **kwargs):
        super(DnnRegModel, self).__init__(conti_features, cate_features, cate_list_features, **kwargs)

        self.output_func = keras.layers.Dense(units=1)

    def call(self, inputs: dict, **kwargs):
        return self.output_func(
            tf.nn.relu(super(DnnRegModel, self).call(inputs))
        )
