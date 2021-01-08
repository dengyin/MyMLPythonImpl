import tensorflow as tf
from tensorflow import keras

from my_nn import FMRegModel, DnnRegModel


class DeepFmRegModel(keras.Model):
    def __init__(self, conti_features: dict, cate_features: dict, cate_list_features: dict, **kwargs):
        super(DeepFmRegModel, self).__init__(**kwargs)
        self.fm = FMRegModel(conti_features, cate_features, cate_list_features)
        self.dnn = DnnRegModel(conti_features, cate_features, cate_list_features)

    def call(self, inputs: dict):
        return self.fm(inputs) + self.dnn(inputs)


class DeepFmClfModel(DeepFmRegModel):
    def __init__(self, conti_features: dict, cate_features: dict, cate_list_features: dict, **kwargs):
        super(DeepFmClfModel, self).__init__(conti_features, cate_features, cate_list_features)

    def call(self, inputs: dict):
        return tf.nn.sigmoid(super(DeepFmClfModel, self).call(inputs))
