import tensorflow as tf
from tensorflow import keras


class ContiFeaLayer(keras.Model):
    def __init__(self, conti_feature, **kwargs):
        super(ContiFeaLayer, self).__init__(name=conti_feature, **kwargs)
        self.conti_feature = conti_feature
        self.bn = tf.keras.layers.BatchNormalization(name=f'{self.conti_feature}_bn')

    def call(self, input):
        return self.bn(input)


class ContiEmbdFeaLayer(keras.Model):
    def __init__(self, conti_embd_feature, paras, **kwargs):
        super(ContiEmbdFeaLayer, self).__init__(name=conti_embd_feature, **kwargs)
        self.conti_embd_feature = conti_embd_feature
        self.model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(name=f'{self.conti_embd_feature}_bn'),
            tf.keras.layers.Dense(**paras, name=f'{self.conti_embd_feature}_embd')
        ], name=f'{self.conti_embd_feature}_layer')

    def call(self, input):
        return self.model(input)


class ContiEmbdSeqFeaLayer(keras.Model):
    def __init__(self, conti_embd_seq_feature, input_length, paras, **kwargs):
        super(ContiEmbdSeqFeaLayer, self).__init__(name=conti_embd_seq_feature, **kwargs)
        self.conti_embd_seq_feature = conti_embd_seq_feature
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape([input_length, 1]),
            tf.keras.layers.BatchNormalization(name=f'{self.conti_embd_seq_feature}_bn'),
            tf.keras.layers.Dense(**paras, name=f'{self.conti_embd_seq_feature}_embd')
        ], name=f'{self.conti_embd_seq_feature}_layer')

    def call(self, input):
        return self.model(input)


class CateFeaLayer(keras.Model):
    def __init__(self, cate_feature, paras, **kwargs):
        super(CateFeaLayer, self).__init__(name=cate_feature, **kwargs)
        self.cate_feature = cate_feature
        self.model = tf.keras.layers.Embedding(**paras, name=f'{self.cate_feature}_embd')

    def call(self, input):
        return tf.squeeze(self.model(input), axis=1)


class CateSeqFeaLayer(keras.Model):
    def __init__(self, cate_seq_feature, paras, **kwargs):
        super(CateSeqFeaLayer, self).__init__(name=cate_seq_feature, **kwargs)
        self.cate_seq_feature = cate_seq_feature
        self.model = tf.keras.layers.Embedding(**paras, name=f'{self.cate_seq_feature}_embd')

    def call(self, input):
        return self.model(input)


class InputLayer(keras.Model):
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

    conti_embd_list_features = { dense parameter
        'feature_name':
            {
                'input_length': 20,
                'paras': {
                    'units': 16,
                    'use_bias': False
                }
            }
        , ....
    }

    """

    def __init__(self, conti_features: dict, conti_embd_features: dict, cate_features: dict, cate_seq_features: dict,
                 conti_embd_seq_features: dict, **kwargs):
        super(InputLayer, self).__init__(**kwargs)

        self.conti_features = conti_features
        self.conti_embd_features = conti_embd_features
        self.cate_features = cate_features
        self.cate_seq_features = cate_seq_features
        self.conti_embd_seq_features = conti_embd_seq_features
        self.input_layers = {}
        self.seq_input_layers = {}

        if conti_features:
            for k in conti_features.keys():
                self.input_layers[k] = ContiFeaLayer(k)

        if conti_embd_features:
            for k in conti_embd_features.keys():
                self.input_layers[k] = ContiEmbdFeaLayer(k, conti_embd_features.get(k))

        if conti_embd_seq_features:
            for k in conti_embd_seq_features.keys():
                self.seq_input_layers[k] = ContiEmbdSeqFeaLayer(k,
                                                                conti_embd_seq_features.get(k)['input_length'],
                                                                conti_embd_seq_features.get(k)['paras'])
        if cate_features:
            for k in cate_features.keys():
                self.input_layers[k] = CateFeaLayer(k, cate_features.get(k))

        if cate_seq_features:
            for k in cate_seq_features.keys():
                self.seq_input_layers[k] = CateSeqFeaLayer(k, cate_seq_features.get(k))

    def call(self, input, seq_features_concat_way):
        if seq_features_concat_way == 'stack':
            assert self.conti_features is None, 'stack way 不允许输入连续变量'
        assert seq_features_concat_way in ['stack', 'flatten', 'mean', 'sum']

        if self.cate_seq_features or self.conti_embd_seq_features:
            if seq_features_concat_way == 'stack':
                seq_fea_concat_func = lambda x: x
            elif seq_features_concat_way == 'flatten':
                seq_fea_concat_func = keras.layers.Flatten()
            elif seq_features_concat_way == 'mean':
                seq_fea_concat_func = lambda x: tf.reduce_mean(x, axis=1)
            elif seq_features_concat_way == 'sum':
                seq_fea_concat_func = lambda x: tf.reduce_sum(x, axis=1)

        outputs = []
        for k in self.input_layers.keys():
            outputs.append(self.input_layers.get(k)(input.get(k)))
        seq_outputs = []
        for k in self.seq_input_layers.keys():
            seq_outputs.append(self.seq_input_layers.get(k)(input.get(k)))

        result = []

        if outputs:
            if seq_features_concat_way == 'stack':
                outputs = tf.stack(outputs, axis=1)  # batch * n_feature * dim
            else:
                outputs = tf.concat(outputs, axis=-1)  # batch * (n_feature * dim)
            result.append(outputs)

        if seq_outputs:
            seq_outputs = tf.concat(seq_outputs, axis=-1)  # batch * length * dim
            seq_outputs = seq_fea_concat_func(seq_outputs)
            result.append(seq_outputs)

        if len(result) > 1:
            return tf.concat(result, axis=1)
        else:
            return result[0]

    def create_input(self, data):
        result = {}
        if self.conti_features:
            for name in self.conti_features.keys():
                result[name] = tf.convert_to_tensor(data[name].values.reshape(-1, 1), dtype=tf.float32)
        if self.conti_embd_features:
            for name in self.conti_embd_features.keys():
                result[name] = tf.convert_to_tensor(data[name].values.reshape(-1, 1), dtype=tf.float32)
        if self.cate_features:
            for name in self.cate_features.keys():
                result[name] = tf.convert_to_tensor(data[name].values.reshape(-1, 1), dtype=tf.int32)
        if self.cate_seq_features:
            for name in self.cate_seq_features.keys():
                result[name] = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
                    data[name].apply(lambda x: [a for a in x if isinstance(a, int)]),
                    maxlen=self.cate_seq_features[name]['input_length'],
                    dtype='int32',
                    padding='pre',
                    truncating='pre', value=0.0
                ))
        if self.conti_embd_seq_features:
            for name in self.conti_embd_seq_features.keys():
                result[name] = tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences(
                    data[name].apply(lambda x: [a for a in x if isinstance(a, float)]),
                    maxlen=self.conti_embd_seq_features[name]['input_length'],
                    dtype='float32',
                    padding='pre',
                    truncating='pre', value=0.0
                ))

        return result
