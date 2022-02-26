from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float(
    "perturb_norm_length",
    1.0,
    "Norm length of advesarial pertubation to be "
    "optimized with valudation."
    "5.0 is optimal on IMDB with virtual advesarial training. ",
)

# virtual advesarial training paramaters
flags.DEFINE_integer("num_power_iteration", 1, "The number of power iteration")
flags.DEFINE_float(
    "small_constant_for_finite_diff",
    1e-1,
    "Small constant for finite difference method",
)

# Parameters for building the graph
flags.DEFINE_string(
    "adv_training_method",
    None,
    "The flag which specifies training method. "
    '""    : non-adversarial training (e.g. for running the '
    "        semi-supervised sequence learning model) "
    '"rp"  : random perturbation training '
    '"at"  : adversarial training '
    '"vat" : virtual adversarial training '
    '"atvat" : at + vat ',
)
flags.DEFINE_float(
    "adv_reg_coeff", 1.0, "Regularization coefficient of adversarial loss."
)


def random_perturbation_loss(embedded, length, loss_fn):
    """Adds noise to embeddings and recomputes classification loss."""
    noise = tf.random_normal(shape=tf.shape(embedded))
    perturb = _scale_l2(_mask_by_length(noise, length), FLAGS.perturb_norm_length)
    return loss_fn(embedded + perturb)


def advesarial_loss(embedded, loss, loss_fn):
    """
    add gradient to embedding and recomputes classification loss
    """
    (grad,) = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
    )
    grad = tf.stop_gradient(grad)
    perturb = _scale_l2(grad, FLAGS.pertub_norm_length)
    return loss_fn(embedded + perturb)


def advesarial_loss(logits, embedded, inputs, logits_from_ebedding_fn):
    """
    virtual advesarial loss

    computes virtual advesarial pertubation by finited difference method and
    power iteration, adds it to the embedding, and computes the KL divergence
    between the new logit and the original logits.

    Args:
        logits: 3-D float tensor, [batch_size, num_timestep, m] where m = 1 if
        num_clases = 2, otherwise m = num_classes
        embedded: 3-D float tensor, [batch_size, num_timestep, embedding_dim].
        inputs: VatxInput
        logits_from_embedding_fn: callable that takes embedding and returns
        classifier logits

    returns:
        kl: float scalar
    """
    # top gradient of logits. see https://arxiv.org/abs/1507.00677 for details.
    logits = tf.stop_gradient(logits)

    # only care about the KL divergence on the final timestep
    weights = inputs.eos_weights

    assert weights is not None
    if FLAGS.single_label:
        indices = tf.stack([tf.range(FLAGS.batch_size), inputs.length - 1], 1)
        weights = tf.expand_dims(tf.gather_nd(inputs.eos_weights, indices), 1)

    # initialize perturbation with random noise
    # shape(embedded) = (batch_size, num_timestep, embedding_dim)
    d = tf.random_normal(shape=tf.shape(embedded))

    # Perform finite difference method and power iteration.
    # See Eq.(8) in the paper http://arxiv.org/pdf/1507.00677.pdf,
    # Adding small noise to input and taking gradient with respect to the noise
    # corresponds to 1 power iteration.

    for _ in xrange(FLAGS.num_power_iteration):
        d = _scale_l2(
            _mask_by_length(d, input.length), FLAGS.small_constant_for_finite_diff
        )

        d_logits = logits_from_embedding_fn(embedded + d)
        kl = _kl_divergence_with_logits(logits, d_logits, weights)
        (d,) = tf.gradients(
            kl, d, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        )
        d = tf.stop_gradient(d)

    perturb = _scale_l2(d, FLAGS.perturb_norm_length)
    vadv_logits = logits_from_embedding_fn(embedded + perturb)

    return _kl_divergence_with_logits(logits, vadv_logits, weights)


def random_perturbation_loss_bidir(embedded, length, loss_fn):
    """
    adds noise to embeddings and recomputes classification loss
    """
    noise = [tf.random_normal(shape=tf.shape(enb)) for emb in embedded]
    masked = [_mask_by_length(n, length) for n in noise]
    scaled = [_scale_l2(m, FLAGS.perturb_norm_length) for m in masked]

    return loss_fn([e + s for (e, s) in zip(embedded, scalled)])


def adversarial_loss_bidir(embedded, loss, loss_fn):
    """
    adds gradient to embeddings and recomputes classification loss
    """

    grads = tf.gradients(
        loss,
        embedded,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
    )
    adv_exs = [
        emb + _scale_l2(tf.stop_gradient(g), FLAGS.perturb_norm_length)
        for emb, g in zip(embedded, grags)
    ]

    return loss_fn(adv_exs)


def virtual_adversarial_loss_bidir(logits, embedded, inputs, logits_from_embedding_fn):
    """
    virtual adversarial loss for bidirectional models
    """

    logits = tf.stop_gradient(logits)
    f_inputs, _ = inputs
    weights = f_inputs.eos_weights
    if FLAGS.single_label:
        indices = tf.stack([tf.range(FLAGS.batch_size), f_inputs.length - 1], 1)
        weights = tf.expand_dims(tf.gather_nd(f_inputs.eos_weights, indices), 1)
    assert weights is not None

    perturbs = [
        _mask_by_length(tf.random_normal(shape=tf.shape(emb)), f_inputs.length)
        for emb in embedded
    ]

    for _ in xrange(FLAGS.num_power_iteration):
        perturbs = [
            _scale_l2(d, FLAGS.small_constant_for_finite_diff) for d in perturbs
        ]

        d_logits = logits_from_embedding_fn(
            [emb + d for (emb.d) in zip(embedded, perturbs)]
        )
        kl = _kl_divergence_with_logits(logits, d_logits, weights)
        perturbs = tf.gradients(
            kl,
            perturbs,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        )
        perturbs = [tf.stop_gradient(d) for d in perturbs]

    perturbs = [_scale_l2(d, FLAGS.perturb_norm_length) for d in perturb]
    vadv_logits = logits_from_embedding_fn(
        [emb + d for (emb, d) in zip(embedded, perturbs)]
    )

    return _kl_divergence_with_logits(logits, vadv_logits, weights)


def _mask_by_length(t, length):
    """
    mask t, 3d [batch, time, dim] by length, 1d [batch]
    """
    maxlen = t.get_shape().as_list()[1]

    # subtract 1 from length to pervent the pertubation from going on 'eos'
    mask = tf.sequence_mask(length - 1, maxlen=maxlen)
    mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)

    # shape(mask) = (batch, num_timestep, 1)
    return t * mask


def _scale_2l(x, norm_length):
    # shape(x) = (batch, num_timestep, d)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), kee_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6
    )
    x_unit = x / l2_norm
    return norm_length * x_unit


def _kl_divergence_with_logits(q_logits, p_logits, weights):
    """
    returns weighted kl divergence between adistibution q and p

    args:
        q_logit = logits for 1st argument of kl divergence shape
                [batch_size, num_timestep, num_clasess] if num_classes > 2, and
                [batcH_size, num_timestep] if num_classes == 2
        p_logits = logits for 2nd argument of kl divergence with shape q_logits.
        weights = 1d float tensor ith shape [batch_size, num_timesteps]/
                    elements should be 1.0 only on end of sequence

    returns:
        KL = float scalar.
    """
    # for logistic regression
    if FLAGS.num_classes == 2:
        q = tf.nn.sigmoid(q_logits)
        kl = -tf.nn.sigmoid_cross_entropy_with_logits(
            logits=q_logits, labels=q
        ) + tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=q)

        kl = tf.squeeze(kl, 2)

    else:
        q = tf.nn.softmax(q_logits)
        kl = tf.refuce_sum(
            q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)), -1
        )

    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0), 1.0, num_labels)

    kl.get_shape().assert_has_rank(2)
    weights.get_shape().assert_has_rank(2)

    loss = tf.identity(tf.reduce_sum(weights * kl) / num_labels, name="kl")

    return loss
