import os
import json


import numpy as np
import tensorflow as tf

from os.path import join
from data_util import load_glove_embeddings, load_dataset, load_snli_dataset
from model import InferModel
import logging
from ptpython.repl import embed

logging.basicConfig(level=logging.INFO)


tf.app.flags.DEFINE_string("data_dir", "./data/squad", "SQUAD data directory")
tf.app.flags.DEFINE_string("data_size", "tiny", "tiny/full")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate ")
tf.app.flags.DEFINE_integer("num_epochs_per_decay", 6, "Epochs before reducing learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.1, "Decay factor")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Norm for clipping gradients ")
tf.app.flags.DEFINE_float("dropout", 0.15, "Dropout")
tf.app.flags.DEFINE_integer("num_epochs",10, "Number of epochs")
tf.app.flags.DEFINE_integer("state_size",100, "State Size")
tf.app.flags.DEFINE_integer("embedding_size",100, "Embedding Size")
tf.app.flags.DEFINE_integer("max_question_length", 60, "Maximum Question Length")
tf.app.flags.DEFINE_integer("max_context_length", 300, "Maximum Context Length")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size")
tf.app.flags.DEFINE_integer("gpu_id", 0, "gpu id")
tf.app.flags.DEFINE_float("gpu_fraction", 0.5, " % of GPU memory used.")
tf.app.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.app.flags.DEFINE_string("dataset","squad","squad/snli")
FLAGS = tf.app.flags.FLAGS


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def main(_):
    dataset, max_q_len, max_c_len = load_dataset(FLAGS.data_dir,
                                                 FLAGS.data_size,
                                                 FLAGS.max_question_length,
                                                 FLAGS.max_context_length)
    embed_path = join("data", "squad", "glove.trimmed.100.npz")
    vocab_path = join(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    embeddings = load_glove_embeddings(embed_path)

    model = InferModel(FLAGS, embeddings, vocab)
    with tf.device("/gpu:{}".format(FLAGS.gpu_id)):
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
        with tf.Session(config=config) as sess:
            logging.info("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
            logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
            model.train(sess,dataset)

if __name__ == "__main__":
    tf.app.run()
