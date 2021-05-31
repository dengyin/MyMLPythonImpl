import tensorflow as tf
from tensorflow import keras

from .fm import FMRegModel
from .dnn import DnnRegModel
from ..input_laysers import InputLayer


class DeepFmRegModel(keras.Model):
    def __init__(self, input_layer: InputLayer, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True, drop_p=0.5, regularizer=None, **kwargs):
        super(DeepFmRegModel, self).__init__(**kwargs)
        self.input_layer = input_layer
        self.fm = FMRegModel(self.input_layer, regularizer)
        self.dnn = DnnRegModel(self.input_layer, seq_features_concat_way, fc_layers, activation,
                               use_bn, use_drop_out, drop_p, regularizer)

    def call(self, inputs: dict):
        return self.fm(inputs) + self.dnn(inputs)


class DeepFmClfModel(DeepFmRegModel):
    def __init__(self, input_layer: InputLayer, seq_features_concat_way='mean', fc_layers=(128,), activation='relu',
                 use_bn=True, use_drop_out=True, drop_p=0.5, regularizer=None, **kwargs):
        super(DeepFmClfModel, self).__init__(input_layer, seq_features_concat_way, fc_layers, activation,
                                             use_bn, use_drop_out, drop_p, regularizer, **kwargs)

    def call(self, inputs: dict):
        return tf.nn.sigmoid(super(DeepFmClfModel, self).call(inputs))
