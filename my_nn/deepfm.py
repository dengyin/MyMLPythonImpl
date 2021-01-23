import tensorflow as tf
from tensorflow import keras

from my_nn import FMRegModel, DnnRegModel


class DeepFmRegModel(keras.Model):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 dnn_cate_list_concat_way='mean', fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, regularizer=tf.keras.regularizers.L2(0.01), **kwargs):
        super(DeepFmRegModel, self).__init__(**kwargs)
        self.fm = FMRegModel(conti_features, conti_embd_features, cate_features, cate_list_features,
                             regularizer=regularizer)
        self.dnn = DnnRegModel(conti_features, conti_embd_features, cate_features, cate_list_features,
                               dnn_cate_list_concat_way,
                               fc_layers, activation, use_bn, use_drop_out,
                               drop_p, regularizer=regularizer, **kwargs)

    def call(self, inputs: dict):
        return self.fm(inputs) + self.dnn(inputs)


class DeepFmClfModel(DeepFmRegModel):
    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 dnn_cate_list_concat_way='mean', fc_layers=(128,), activation='relu', use_bn=True, use_drop_out=True,
                 drop_p=0.5, regularizer=tf.keras.regularizers.L2(0.01), **kwargs):
        super(DeepFmClfModel, self).__init__(conti_features, conti_embd_features, cate_features, cate_list_features,
                                             dnn_cate_list_concat_way, fc_layers, activation, use_bn,
                                             use_drop_out, drop_p, regularizer=regularizer, **kwargs)

    def call(self, inputs: dict):
        return tf.nn.sigmoid(super(DeepFmClfModel, self).call(inputs))
