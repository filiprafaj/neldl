import pickle
import sys

import numpy as np
import tensorflow as tf
from evaluation.metrics import tp_fp_fn

import model.config as config


def load_embeddings(args):
  # word embeddings
    word_embeddings_npy = np.load(config.base_folder/"data/experiments"/args.experiment_name/
                                  "embeddings/word_embeddings.npy")

  # entity embeddings
    entity_embeddings_npy = np.load(config.base_folder/"data/experiments"/args.experiment_name/
                                    "embeddings/entity_embeddings.npy")
    entity_embeddings_npy[0] = 0

    return word_embeddings_npy, entity_embeddings_npy


def load_train_args(output_folder, running_mode):
    """
    running_mode: train, train_continue, evaluate
    """
    with open(output_folder/"train_args.pickle", "rb") as handle:
        train_args = pickle.load(handle)
    train_args.running_mode = running_mode

    return train_args


########################################################################
# DATASETS READER
def parse_sequence_example(serialized):
    context_features={
            "chunk_id": tf.FixedLenFeature([], dtype=tf.string),
            "words_len": tf.FixedLenFeature([], dtype=tf.int64),
            "spans_len": tf.FixedLenFeature([], dtype=tf.int64),
            "ground_truth_len": tf.FixedLenFeature([], dtype=tf.int64)
        }
    sequence_features={
            "words": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "chars": tf.VarLenFeature(tf.int64),
            "chars_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "begin_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "end_span": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "cand_entities": tf.VarLenFeature(tf.int64),
            "cand_entities_scores": tf.VarLenFeature(tf.float32),
            "cand_entities_labels": tf.VarLenFeature(tf.int64),
            "cand_entities_len": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "ground_truth": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "chunk_embeddings": tf.VarLenFeature(tf.float32),
    }
    if True:
        sequence_features["begin_gm"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)
        sequence_features["end_gm"] = tf.FixedLenSequenceFeature([], dtype=tf.int64)

    context, sequence = tf.parse_single_sequence_example(serialized, context_features, sequence_features)

    return (
            context["chunk_id"],
            sequence["words"],
            context["words_len"],
            tf.sparse_tensor_to_dense(sequence["chars"]),
            sequence["chars_len"],
            sequence["begin_span"],
            sequence["end_span"],
            context["spans_len"],
            tf.sparse_tensor_to_dense(sequence["cand_entities"]),
            tf.sparse_tensor_to_dense(sequence["cand_entities_scores"]),
            tf.sparse_tensor_to_dense(sequence["cand_entities_labels"]),
            sequence["cand_entities_len"],
            sequence["ground_truth"],
            context["ground_truth_len"],
            sequence["begin_gm"],
            sequence["end_gm"],
            tf.sparse_tensor_to_dense(sequence["chunk_embeddings"])
           )

def train_input_pipeline(filenames, buffer_size, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size, dataset.output_shapes)
    dataset = dataset.prefetch(1)
    return dataset

def eval_input_pipeline(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_sequence_example)
    dataset = dataset.padded_batch(batch_size, dataset.output_shapes)
    dataset = dataset.prefetch(1)
    return dataset


########################################################################
# OPTIMAL THRESHOLD
def optimal_thr_calc(model, handles, iterators, val_datasets, el_mode):
    tp_fp_scores_labels = [] # filled with (score, TP or FP label)
    fn_scores = []
    for val_dataset in val_datasets:
        dataset_handle = handles[val_dataset]
        iterator = iterators[val_dataset]
        model.sess.run(iterator.initializer)
        while True:
            try:
                retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                              model.begin_span, model.end_span, model.spans_len,
                              model.begin_gm, model.end_gm,
                              model.ground_truth, model.ground_truth_len,
                              model.words_len, model.chunk_id]
                result_l = model.sess.run(retrieve_l,
                                          feed_dict={model.input_handle_ph: dataset_handle,
                                                     model.dropout_keep_prob: 1})

                tp_fp_batch, fn_batch = tp_fp_fn(*result_l, weak_matching=el_mode, allspans=el_mode)
                tp_fp_scores_labels.extend(tp_fp_batch)
                fn_scores.extend(fn_batch)
            except tf.errors.OutOfRangeError:
                break

    return optimal_thr_calc_aux(tp_fp_scores_labels, fn_scores)


def optimal_thr_calc_aux(tp_fp_scores_labels, fn_scores):
    """
    Find optimal threshold based scores.
    Threshold is set to the score which separates the possitives
    and negatives s.t. TP,FP and FN give best F1 score
    """
    tp_fp_scores_labels = sorted(tp_fp_scores_labels)   # low --> high
    fn_scores = sorted(fn_scores) # contains all gm
    fn_idx = len(fn_scores)
    # fn_scores[0: fn_idx-1] ~ FN, fn_scores[fn_idx:] ~ TP/FP
    # Start with such a threshold we reject everything
    # -> set thr to highest score + 1
    best_thr = tp_fp_scores_labels[-1][0]+1
    best_f1 = -1
    # That is why fn_idx = len(fn_scores), we would label all gm as FN.
    # -> tp_count, fp_count = 0
    tp_count, fp_count = 0, 0

    tp_fp_idx = len(tp_fp_scores_labels)
    # this position and right from it is included in the tp, fp
    # left from it remains to be processed
    while tp_fp_idx > 0: # process from right to left
        tp_fp_idx -= 1
        new_thr, label = tp_fp_scores_labels[tp_fp_idx]
        tp_count += label
        fp_count += (1 - label)
      # SKIP equal scores
        while tp_fp_idx > 0 and tp_fp_scores_labels[tp_fp_idx-1][0] == new_thr:
            tp_fp_idx -= 1
            new_thr, label = tp_fp_scores_labels[tp_fp_idx]
            tp_count += label
            fp_count += (1 - label)
      # SHIFT fn_index so it separates fn_scores on new_thr
        while fn_idx > 0 and fn_scores[fn_idx-1] >= new_thr:
            fn_idx -= 1

        assert( 0 <= tp_count <= len(tp_fp_scores_labels) and
                0 <= fp_count <= len(tp_fp_scores_labels) and
                0 <= fn_idx <= len(fn_scores))
      # COUNT F1
        precision = 100 * tp_count / (tp_count + fp_count + 1e-6)
        recall = 100 * tp_count / (tp_count + fn_idx + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        assert(0 <= precision <= 100 and 0 <= recall <= 100 and 0 <= f1 <= 100)
      # BEST F1
        if f1 > best_f1:
            best_f1 = f1
            best_thr = new_thr

    print("Best validation threshold = %.3f with F1=%.1f " % (best_thr, best_f1))

    return best_thr, best_f1


########################################################################
# FFNN
def shape(x, dim):
    return x.get_shape()[dim].value or tf.shape(x)[dim]


def variable_summaries(var):
    """Attach summaries to a Tensor (for TensorBoard visualization)."""
    name = "_" + var.name.split("/")[-1].split(":")[0]
    with tf.name_scope("summaries"+name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))


def ffnn(inputs, num_hidden_layers, hidden_size, output_size, dropout,
         output_weights_initializer=None, model=None):

    if len(inputs.get_shape()) > 2:
        # from [batch, max_sentence_length, emb]
        # to   [batch*max_sentence_length, emb]
        flattened = tf.reshape(inputs, [-1, shape(inputs, -1)])
    else:
        flattened = inputs

  # HIDDEN LAYERS
    for i in range(num_hidden_layers):
        hidden_weights = tf.get_variable("hidden_weights_{}".format(i),
                                         [shape(flattened, 1), hidden_size])
        hidden_bias = tf.get_variable("hidden_bias_{}".format(i), [hidden_size])
#        variable_summaries(hidden_weights)
#        variable_summaries(hidden_bias)

        flattened = tf.nn.relu(tf.matmul(flattened, hidden_weights) + hidden_bias)

        if dropout is not None:
            flattened = tf.nn.dropout(flattened, dropout)

  # OUTPUT LAYER
    output_weights = tf.get_variable("output_weights",
                                     [shape(flattened, 1), output_size],
                                     initializer=output_weights_initializer)
    output_bias = tf.get_variable("output_bias", [output_size])
#    variable_summaries(output_weights)
#    variable_summaries(output_bias)

    flattened = tf.matmul(flattened, output_weights) + output_bias
    #print("model/util variable name = ", output_weights.name, output_bias.name)

    if len(inputs.get_shape()) == 3:
        outputs = tf.reshape(flattened, [shape(inputs, 0), shape(inputs, 1), output_size])
    elif len(inputs.get_shape()) == 4:
        outputs = tf.reshape(flattened, [shape(inputs, 0), shape(inputs, 1), shape(inputs, 2), output_size])
    elif len(inputs.get_shape()) > 4:
        raise ValueError("inputs with rank {} not supported".format(len(inputs.get_shape())))

    return outputs


def projection(inputs, output_size, initializer=None, model=None):
    return ffnn(inputs, 0, -1, output_size, dropout=None,
                output_weights_initializer=initializer, model=model)


########################################################################
# bi-LSTM
def bi_LSTM(input_data, lengths, units, reuse=False):
    cell_fw = tf.nn.rnn_cell.LSTMCell(units, state_is_tuple=True, reuse=reuse)
    cell_bw = tf.nn.rnn_cell.LSTMCell(units, state_is_tuple=True, reuse=reuse)
    output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                     input_data,
                                                     sequence_length=lengths,
                                                     dtype=tf.float32)
    return output, states


########################################################################
# TEE
# https://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python
# http://web.archive.org/web/20141016185743/https://mail.python.org/pipermail/python-list/2007-May/460639.html
class Tee(object):

    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()
