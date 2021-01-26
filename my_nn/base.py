from copy import copy

import tensorflow as tf
from tensorflow import keras


class BaseModel(keras.Model):
    """
    cate{_list}_features = { embdding parameter
        'feature_name':
            {
                'input_dim': len(vocabulary) + 1,
                'output_dim': 16,
                'mask_zero': True,
                'input_length': 20
            }
        , .....
    }

    conti_embd_features = { dense parameter
        'feature_name':
            {
                'units': 16,
                'use_bias': False
            }
        , ....
    }

    conti_features = {
        'feature_name':'feature_name'
        , ....
    }

    """

    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_list_features: dict,
                 cate_list_concat_way='concate', regularizer=None, **kwargs):
        super(BaseModel, self).__init__(**kwargs)
        self.conti_features = conti_features
        self.conti_embd_features = conti_embd_features
        self.cate_features = cate_features
        self.cate_list_features = cate_list_features
        self.cate_list_concat_way = cate_list_concat_way

        self.regularizer = regularizer

        self.conti_embd_suf = '_conti_embd'
        self.cate_list_embd_suf = '_cate_list_embd'
        self.cate_embd_suf = '_cate_embd'

        self.fl = keras.layers.Flatten()

        if self.conti_embd_features:
            for name in self.conti_embd_features.keys():
                seq = tf.keras.Sequential([
                    tf.keras.layers.BatchNormalization(name=name + '_bn'),
                    tf.keras.layers.Dense(**self.conti_embd_features[name], name=name + self.conti_embd_suf)
                ], name=name)
                setattr(self, name + self.conti_embd_suf, seq)

        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                setattr(self, name + self.cate_list_embd_suf,
                        tf.keras.layers.Embedding(**self.cate_list_features[name], name=name + self.cate_list_embd_suf))

        if self.cate_features:
            for name in self.cate_features.keys():
                setattr(self, name + self.cate_embd_suf,
                        tf.keras.layers.Embedding(**self.cate_features[name], name=name + self.cate_embd_suf))

        if self.cate_list_concat_way == 'fm':
            self.cate_list_concat_func = lambda x: x
        elif self.cate_list_concat_way == 'concate':
            self.cate_list_concat_func = self.fl
        elif self.cate_list_concat_way == 'mean':
            self.cate_list_concat_func = lambda x: tf.reduce_mean(x, axis=1)
        elif self.cate_list_concat_way == 'sum':
            self.cate_list_concat_func = lambda x: tf.reduce_sum(x, axis=1)

    def call(self, inputs: dict, **kwargs):
        result = []
        if self.cate_list_features:
            result.append(tf.concat(
                [self.cate_list_concat_func(self.__dict__[name](inputs[name[:-len(self.cate_list_embd_suf)]]))
                 for name in self.__dict__.keys() if name.endswith(self.cate_list_embd_suf)],
                axis=1
            ))

        if self.cate_features:
            if self.cate_list_concat_way == 'fm':
                result.append(tf.concat(
                    [self.__dict__[name](inputs[name[:-len(self.cate_embd_suf)]])
                     for name in self.__dict__.keys() if name.endswith(self.cate_embd_suf)],
                    axis=1
                ))  # batch_size * n_cate_features * embd_size
            else:
                result.append(tf.concat(
                    [self.fl(self.__dict__[name](inputs[name[:-len(self.cate_embd_suf)]]))
                     for name in self.__dict__.keys() if name.endswith(self.cate_embd_suf)],
                    axis=1
                ))  # batch_size * (n_cate_features * embd_size)

        if self.conti_embd_features:
            if self.cate_list_concat_way == 'fm':
                result.append(tf.concat(
                    [tf.expand_dims(self.__dict__[name](inputs[name[:-len(self.conti_embd_suf)]]), axis=1)
                     for name in self.__dict__.keys() if name.endswith(self.conti_embd_suf)],
                    axis=1
                ))  # batch_size * n_cate_features * embd_size
            else:
                result.append(tf.concat(
                    [self.fl(self.__dict__[name](inputs[name[:-len(self.conti_embd_suf)]]))
                     for name in self.__dict__.keys() if name.endswith(self.conti_embd_suf)],
                    axis=1
                ))  # batch_size * (n_cate_features * embd_size)

        if len(result) > 1:
            return tf.concat(result, axis=1)
        else:
            return result[0]

    def create_input(self, data, return_input_shape=False):
        result = {}
        input_shape = {}
        if self.conti_features:
            for name in self.conti_features.keys():
                result[name] = tf.convert_to_tensor(data[name].values.reshape(-1, 1), dtype=tf.float32)
                input_shape[name] = (None, 1)
        if self.conti_embd_features:
            for name in self.conti_embd_features.keys():
                result[name] = tf.convert_to_tensor(data[name].values.reshape(-1, 1), dtype=tf.float32)
                input_shape[name] = (None, 1)
        if self.cate_features:
            for name in self.cate_features.keys():
                result[name] = tf.convert_to_tensor(data[name].values.reshape(-1, 1), dtype=tf.int32)
                input_shape[name] = (None, 1)
        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                result[name] = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
                    data[name].apply(lambda x: [a for a in x if isinstance(a, int)]),
                    maxlen=self.cate_list_features[name]['input_length'],
                    dtype='int32',
                    padding='pre',
                    truncating='pre', value=0.0
                ))
                input_shape[name] = (None, self.cate_list_features[name]['input_length'])
        if return_input_shape:
            return result, input_shape
        else:
            return result


class BaseSeqModel(keras.Model):
    """
    cate_list_features = { embdding parameter
        'feature_name':
            {
                'input_dim': len(vocabulary) + 1,
                'output_dim': 16,
                'mask_zero': True,
                'input_length': 20
            }
        , .....
    }

    conti_embd_list_features = { dense parameter
        'feature_name':
            {
                'units': 16,
                'use_bias': False,
                'input_length': 20
            }
        , ....
    }


    """

    def __init__(self, conti_list_embd_features: dict, cate_list_features: dict, **kwargs):
        super(BaseSeqModel, self).__init__(**kwargs)
        self.conti_list_embd_features = conti_list_embd_features
        self.cate_list_features = cate_list_features

        self.conti_list_embd_suf = '_conti_list_embd'
        self.cate_list_embd_suf = '_cate_list_embd'

        if self.conti_list_embd_features:
            for name in self.conti_list_embd_features.keys():
                seq = tf.keras.Sequential([
                    tf.keras.layers.Reshape([self.conti_list_embd_features[name].get('input_length'), 1]),
                    tf.keras.layers.BatchNormalization(name=name + '_bn'),
                    tf.keras.layers.Dense(**remove_key(self.conti_list_embd_features[name], 'input_length'),
                                          name=name + self.conti_list_embd_suf)
                ], name=name)
                setattr(self, name + self.conti_list_embd_suf, seq)

        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                setattr(self, name + self.cate_list_embd_suf,
                        tf.keras.layers.Embedding(**self.cate_list_features[name], name=name + self.cate_list_embd_suf))

    def call(self, inputs: dict, **kwargs):
        result = []
        if self.cate_list_features:
            result += [self.__dict__[name](inputs[name[:-len(self.cate_list_embd_suf)]])
                       for name in self.__dict__.keys() if
                       name.endswith(self.cate_list_embd_suf)]  # batch_size * input_length * (embd_size * n)

        if self.conti_list_embd_features:
            result += [tf.reshape(self.__dict__[name](inputs[name[:-len(self.conti_list_embd_suf)]]),
                                  shape=[-1,
                                         self.conti_list_embd_features.get(name[:-len(self.conti_list_embd_suf)]).get(
                                             'input_length'),
                                         self.conti_list_embd_features.get(name[:-len(self.conti_list_embd_suf)]).get(
                                             'units'),
                                         ])
                       for name in self.__dict__.keys() if
                       name.endswith(self.conti_list_embd_suf)]  # batch_size * input_length * (units * n)

        if len(result) > 1:
            return tf.concat(result, axis=-1)
        else:
            return result[0]

    def create_input(self, data, return_input_shape=False):
        result = {}
        input_shape = {}
        if self.conti_list_embd_features:
            for name in self.conti_list_embd_features.keys():
                result[name] = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
                    data[name].apply(lambda x: [a for a in x if isinstance(a, float)]),
                    maxlen=self.conti_list_embd_features[name]['input_length'],
                    dtype='float32',
                    padding='pre',
                    truncating='pre', value=0.0
                ))
                input_shape[name] = (None, self.conti_list_embd_features[name]['input_length'])
        if self.cate_list_features:
            for name in self.cate_list_features.keys():
                result[name] = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
                    data[name].apply(lambda x: [a for a in x if isinstance(a, int)]),
                    maxlen=self.cate_list_features[name]['input_length'],
                    dtype='int32',
                    padding='pre',
                    truncating='pre', value=0.0
                ))
                input_shape[name] = (None, self.cate_list_features[name]['input_length'])
        if return_input_shape:
            return result, input_shape
        else:
            return result


def remove_key(d: dict, key):
    r = copy(d)
    del r[key]
    return r
