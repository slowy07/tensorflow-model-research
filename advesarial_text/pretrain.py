from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

import graphs
import train_utils

FLAGS = tf.app.flags.FLAGS


def main(_):
    """Trains Language Model."""
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        model = graphs.get_model()
        train_op, loss, global_step = model.language_model_training()
        train_utils.run_training(train_op, loss, global_step)


if __name__ == "__main__":
    tf.app.run()
