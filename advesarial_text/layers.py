from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

K = tf.keras


def cl_logits_subgraph(layer_sizes, input_sizes, num_clasess, keep_prob=1.0):
    """
    construct multiple ReLu layers with dropout and linear layer
    """
    subgraph = K.models.Sequential(name="cl_logits")
    for i, layer_size in enumerate(layer_size):
        if i == 0:
            subgraph.add(
                K.layer.Dense(layer_size, activation="relu", input_dim=input_size)
            )
        else:
            subgraph.add(K.layers.Dense(layer_size, activation="relu"))

        if keep_prob < 1.0:
            subgraph.add(K.layers.Dropout(1.0 - keep_prob))
        subgraph.add(K.layers.Dense(1 if num_clasess == 2 else num_clasess))
        return subgraph


class Embedding(K.layers.Layer):
    """
    embedding layer with frequencey-based normalized and dropout.
    """

    def __init__(
        vocab_size,
        embedding_dim,
        normalize=False,
        vocab_freqs=None,
        keep_prob=1.0,
        **kwargs
    ):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.normalized = normalize
        self.keep_prob = keep_prob

        if normalize:
            # besok sabtu heheheh
            assert vocab_freqs is not None
            self.vocab_freqs =  tf.constant(vocab_freqs, dtype = tf.float32, shape=(vocab_size, 1))

        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.device('/cpu:0'):
            self.var = self.add_weight(
                shape = (self.vocab_size, self.embedding_dim),
                intializer = tf.random_uniform_initializer(-1, 1.),
                name = 'embedding',
                dtype = tf.float32
            )

        if self.normalized:
            self.var = sel._normalize(self.var)

        super(Embedding, self).build(input_shape)

    def call(self, x):
        embedded = tf.nn.embedding_lookup(self.var, x)
        if self.keep_prob < 1.:
            shape = embedded.get_shape().as_list()

        # use same dropout mask at each timestep with specifying noise_shape
        # this slightly improves performance.
        # please see https://arxiv.org/abs/1512.05287 for the theoretical
        # explanation

        embedded = tf.nn.dropout(
            embedded, self.keep_prob, noise_shape = (shape[0], 1, shape[2])
        )
        return embedded

    def _normalize(self, emb):
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keep_dims = True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dim = True)
        return (emb - mean) / stddev

class LSTM(object):
    """
    LSTM layer using dynameic_rnn
    """
    def __init__(self, cell_size, num_layer = 1, keep_prob = 1, name = 'LSTM'):
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, initial_state, seq_length):
        with tf.variable_scope(self.name, reuse = self.reuse ) as vs:
            cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.BasicLTMCell(
                    self.cell_size,
                    forget_bias = 0.0
                    reuse = tf.get_variabel_scope().reuse
                )
                for _ in xrange(self.num_layers)
            ])

            lstm_out, next_state = tf.nn.dynameic_rnn(
                cell, x, initial_state = initial_state, sequence_length = seq_length
            )

            if self.keep_prob < 1.:
                lstm_out = tf.nn.dropout(lstm_out, self.keep_prob)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = None
        
        return lstm_out, next_state

class SoftmaxLoss(K.layers.Layer):
    """
    softmaax xentropy loss with candidate sampling
    """
    def __init__(
        vocab_size,
        num_candicate_samples = -1,
        vocab_freqs = None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.num_candicate_samples = num_candicate_samples
        self.vocab_freq = vocab_freqs
        super(SoftmaxLoss, self).__init__(**kwargs)
        self.multicast_dense_layer = K.layers.Dense(self.vocab_size)

    def build(self, input_shape):
        input_shape = input_shape[0].as_list()
        with tf.device('/cpu:0'):
            self.lin_w = self.add_weight(
            shape=(input_shape[-1], self.vocab_size),
            name='lm_lin_w',
            initializer=K.initializers.glorot_uniform())
        self.lin_b = self.add_weight(
            shape=(self.vocab_size,),
            name='lm_lin_b',
            initializer=K.initializers.glorot_uniform())
        self.multiclass_dense_layer.build(input_shape)

        super(SoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        x, labels, weights = inputs
        if self.num_candidate_samples > -1:
            assert self.vocab_freqs is not None
            labels_reshaped = tf.reshape(labels, [-1])
            labels_reshaped = tf.expand_dims(labels_reshaped, -1)
            sampled = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_reshaped,
                num_true=1,
                num_sampled=self.num_candidate_samples,
                unique=True,
                range_max=self.vocab_size,
                unigrams=self.vocab_freqs)
        inputs_reshaped = tf.reshape(x, [-1, int(x.get_shape()[2])])

        lm_loss = tf.nn.sampled_softmax_loss(
            weights=tf.transpose(self.lin_w),
            biases=self.lin_b,
            labels=labels_reshaped,
            inputs=inputs_reshaped,
            num_sampled=self.num_candidate_samples,
            num_classes=self.vocab_size,
            sampled_values=sampled)
        lm_loss = tf.reshape(
            lm_loss,
            [int(x.get_shape()[0]), int(x.get_shape()[1])])
        else:
            logits = self.multiclass_dense_layer(x)
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
            )

        lm_loss = tf.identity(
            tf.reduce_sum(lm_loss * weights) / _num_labels(weights),
            name='lm_xentropy_loss')
    return lm_loss

