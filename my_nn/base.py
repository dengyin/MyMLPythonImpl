import tensorflow as tf
from tensorflow import keras


class BaseModel(keras.Model):

    def __init__(self, conti_features: dict, cate_features: dict, cate_list_features: dict, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.conti_features = conti_features
        self.cate_features = cate_features
        self.cate_list_features = cate_list_features

        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                setattr(self, name + '_embedding',
                        tf.keras.layers.Embedding(**self.cate_list_features[name], name=name + '_embedding'))
                self.__dict__[name + '_embedding'].build((None, self.cate_list_features[name]['input_length']))

        if self.cate_features:
            for name in self.cate_features.keys():
                setattr(self, name + '_embedding',
                        tf.keras.layers.Embedding(**self.cate_features[name], name=name + '_embedding'))
                self.__dict__[name + '_embedding'].build((None, self.cate_features[name]['input_length']))

    def call(self, inputs: dict, **kwargs):
        return tf.concat(
            [self.__dict__[name](inputs[name[:-len('_embedding')]]) for name in self.__dict__.keys() if
             name.endswith('_embedding')],
            axis=1
        )  # batch_size * n_cate_features * embd_size

    def create_input(self, data):
        result = {}
        if self.conti_features:
            for name in self.conti_features.keys():
                result[name] = data[name].astype('float32')
        if self.cate_features:
            for name in self.cate_features.keys():
                result[name] = data[name]
        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                result[name] = tf.keras.preprocessing.sequence.pad_sequences(
                    data[name].apply(lambda x: [a for a in x if isinstance(a, int)]),
                    maxlen=self.cate_list_features[name]['input_length'],
                    dtype='int32',
                    padding='pre',
                    truncating='pre', value=0.0
                )
        return result
