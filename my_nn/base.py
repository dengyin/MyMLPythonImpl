import tensorflow as tf
from tensorflow import keras


class BaseModel(keras.Model):

    def __init__(self, conti_features: dict, cate_features: dict, cate_list_features: dict,
                 cate_list_concat_way='concate',
                 **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.conti_features = conti_features
        self.cate_features = cate_features
        self.cate_list_features = cate_list_features
        self.cate_list_concat_way = cate_list_concat_way

        self.cate_list_embd_suf = '_cate_list_embd'
        self.cate_embd_suf = '_cate_embd'

        self.fl = keras.layers.Flatten()

        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                setattr(self, name + self.cate_list_embd_suf,
                        tf.keras.layers.Embedding(**self.cate_list_features[name], name=name + self.cate_list_embd_suf))
                self.__dict__[name + self.cate_list_embd_suf].build(
                    (None, self.cate_list_features[name]['input_length']))

        if self.cate_features:
            for name in self.cate_features.keys():
                setattr(self, name + self.cate_embd_suf,
                        tf.keras.layers.Embedding(**self.cate_features[name], name=name + self.cate_embd_suf))
                self.__dict__[name + self.cate_embd_suf].build((None, self.cate_features[name]['input_length']))

        if self.cate_list_concat_way == 'fm':
            self.cate_list_concat_func = lambda x: x
        elif self.cate_list_concat_way == 'concate':
            self.cate_list_concat_func = self.fl
        elif self.cate_list_concat_way == 'mean':
            self.cate_list_concat_func = lambda x: tf.reduce_mean(x, axis=1)
        elif self.cate_list_concat_way == 'sum':
            self.cate_list_concat_func = lambda x: tf.reduce_sum(x, axis=1)

    def call(self, inputs: dict, **kwargs):
        if self.cate_list_features:
            cate_list_embd = tf.concat(
                [self.cate_list_concat_func(self.__dict__[name](inputs[name[:-len(self.cate_list_embd_suf)]]))
                 for name in self.__dict__.keys() if name.endswith(self.cate_list_embd_suf)],
                axis=1
            )

        if self.cate_features:
            if self.cate_list_concat_way == 'fm':
                cate_embd = tf.concat(
                    [self.__dict__[name](inputs[name[:-len(self.cate_embd_suf)]])
                     for name in self.__dict__.keys() if name.endswith(self.cate_embd_suf)],
                    axis=1
                )  # batch_size * n_cate_features * embd_size
            else:
                cate_embd = tf.concat(
                    [self.fl(self.__dict__[name](inputs[name[:-len(self.cate_embd_suf)]]))
                     for name in self.__dict__.keys() if name.endswith(self.cate_embd_suf)],
                    axis=1
                )  # batch_size * (n_cate_features * embd_size)

        if self.cate_features and self.cate_list_features:
            return tf.concat([cate_list_embd, cate_embd], axis=1)
        elif self.cate_features:
            return cate_embd
        else:
            return cate_list_embd

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
