import argparse
import os
import pickle
import time

import numpy as np
import tensorflow as tf
from evaluation.metrics import Evaluator

import model.config as config
import model.util as util
from model.model import Model
from model.util import Tee


def tensorboard_writers(summaries_folder, graph):
    tf_writers = dict()

    tf_writers["train"] = tf.summary.FileWriter(summaries_folder/"train", graph)

    tf_writers["strong_pr"] = tf.summary.FileWriter(summaries_folder/"strong_pr")
    tf_writers["strong_re"] = tf.summary.FileWriter(summaries_folder/"strong_re")
    tf_writers["strong_f1"] = tf.summary.FileWriter(summaries_folder/"strong_f1")

    tf_writers["weak_pr"] = tf.summary.FileWriter(summaries_folder/"weak_pr")
    tf_writers["weak_re"] = tf.summary.FileWriter(summaries_folder/"weak_re")
    tf_writers["weak_f1"] = tf.summary.FileWriter(summaries_folder/"weak_f1")

    return tf_writers


def create_training_pipelines(args):
    folder = config.base_folder/"data/experiments"/args.experiment_name/("tfrecords_allspans"
             if args.allspans else "tfrecords_goldspans")
    training_dataset = util.train_input_pipeline([str(folder/f) for f in args.train_datasets],
                                                 args.dataset_shuffle_buffer_size, args.batch_size)

    return training_dataset


def create_eval_pipelines(args, filenames):
    if filenames is None:
        return [], []

    folder = config.base_folder/"data/experiments"/args.experiment_name/("tfrecords_allspans"
             if args.allspans  else "tfrecords_goldspans")
    eval_datasets = []
    for file in filenames:
        eval_datasets.append(util.eval_input_pipeline([str(folder/file)], args.batch_size))

    return eval_datasets, filenames


def dataset_score(args, model, iterator, dataset_handle, opt_thr, weak_matching, allspans, name=""):
    # Run one pass over the validation dataset.
    # name is the name of the dataset e.g. aida_test.txt, aquaint.txt
    model.sess.run(iterator.initializer)
    evaluator = Evaluator(opt_thr, name=name)
    while True:
        try:
            retrieve_l = [model.final_scores, model.cand_entities_len, model.cand_entities,
                          model.begin_span, model.end_span, model.spans_len,
                          model.begin_gm, model.end_gm,
                          model.ground_truth, model.ground_truth_len,
                          model.words_len, model.chunk_id]

            result_l = model.sess.run(retrieve_l,
                                      feed_dict={model.input_handle_ph: dataset_handle, model.dropout_keep_prob: 1})

            evaluator.tp_fp_fn(*result_l, weak_matching, allspans)

        except tf.errors.OutOfRangeError:
            print(name)
            micro_f1, macro_f1 = evaluator.print_and_log_results(weak_matching, model.tf_writers, args.eval_cnt)
            break

    return micro_f1, macro_f1


def evaluate_datasets(args, model, handles, names, iterators, el_mode):
    if args.hardcoded_thr:
        opt_thr = args.hardcoded_thr
    else:
        # Compute the optimal threshold based on validation datasets.
        opt_thr, val_f1 = util.optimal_thr_calc(model, handles, iterators, args.eval_datasets_validation, el_mode)

    micro_results = []
    macro_results = []
    for test_handle, test_name, test_it in zip(handles, names, iterators):
        micro_f1, macro_f1 = dataset_score(args, model, test_it, test_handle, opt_thr, weak_matching=el_mode,
                                           allspans=el_mode, name=test_name)
        micro_results.append(micro_f1)
        macro_results.append(macro_f1)

  # just some test
    if (not args.hardcoded_thr and len(args.eval_datasets_validation) == 1
        and abs(micro_results[args.eval_datasets_validation[0]] - val_f1) > 0.1):
        print("ASSERTION ERROR: optimal threshold f1 calculalation differs from normal f1:",
              val_f1, "  and ", micro_results[args.eval_datasets_validation[0]])

    return micro_results, macro_results


def train(args):
  # DATASETS
    eval_datasets, eval_names = create_eval_pipelines(args, filenames=args.eval_datasets)
    training_dataset = create_training_pipelines(args)

  # FEEDABLE ITERATOR
    # from https://www.tensorflow.org/guide/datasets:
      # The `Iterator.string_handle()` method returns a tensor
      # that can be evaluated and used to feed the `handle` placeholder.
    input_handle_ph = tf.placeholder(tf.string, shape=[], name="input_handle_ph")
    iterator = tf.data.Iterator.from_string_handle(input_handle_ph, training_dataset.output_types,
                                                   training_dataset.output_shapes)
    next_element = iterator.get_next()

  # MODEL
    model = Model(args, next_element)
    model.build()
    model.input_handle_ph = input_handle_ph    # for access convenience

  # TensorBoard
    tf_writers = tensorboard_writers(args.summaries_folder, model.sess.graph)
    model.tf_writers = tf_writers   # for access convenience

    with model.sess as sess:
      # ITERATORS AND HANDLES
        def eval_dataset_handles(datasets):
            test_iterators = []
            test_handles = []
            for dataset in datasets:
                test_iterator = dataset.make_initializable_iterator()
                test_iterators.append(test_iterator)
                test_handles.append(sess.run(test_iterator.string_handle()))
            return test_iterators, test_handles

        training_iterator = training_dataset.make_one_shot_iterator()
        training_handle = sess.run(training_iterator.string_handle())

        eval_iterators, eval_handles = eval_dataset_handles(eval_datasets)


      # TRAINING and EVALUATION
        best_eval_score1 = 0
        best_eval_score2 = 0
        epochs_no_improve = 0  # early stopping
        while True:
          # TRAINING
            total_train_loss = 0
            #wall_start = time.time()
            for _ in range(args.steps_before_evaluation):
            #while ( (time.time() - wall_start) / 60 ) <= args.evaluation_minutes:
                _, loss = sess.run([model.train_op, model.loss],
                                   feed_dict={input_handle_ph: training_handle,
                                              model.dropout_keep_prob: args.dropout_keep_prob,
                                              model.lr: args.lr})
                total_train_loss += loss

          # LEARNING RATE DECAY
            if model.args.lr_decay > 0:
                model.args.lr *= model.args.lr_decay

          # EVALUATION
            args.eval_cnt += 1
            print("\nEVALUATION:", args.eval_cnt)

            summary = tf.Summary(value=[tf.Summary.Value(tag="total_train_loss", simple_value=total_train_loss)])
            tf_writers["train"].add_summary(summary, args.eval_cnt)

            wall_start = time.time()
            current_eval_score = -0.1
            print("Mode: {}".format("allspans" if args.allspans else "goldspans"))
            micro_f1, macro_f1 = evaluate_datasets(args, model, eval_handles, eval_names, eval_iterators,
                                       el_mode=args.allspans)
            current_eval_score1 = round(np.mean(np.array(micro_f1)[args.eval_datasets_validation]),1)
            current_eval_score2 = round(np.mean(np.array(macro_f1)[args.eval_datasets_validation]),1)
            print("Evaluation duration in minutes: ", (time.time() - wall_start) / 60)

          # IMPROVEMENT CHECK
            text = ""
            best_eval_flag = False
            if current_eval_score1 >= best_eval_score1 + args.improvement_threshold:
                if ((current_eval_score1 > best_eval_score1) or (current_eval_score2 >= best_eval_score2)):
                    text += ("New best score!\n"+"from micro: "+str(best_eval_score1)+', macro: '+str(best_eval_score2)+
                            "\nto micro: "+str(current_eval_score1)+', macro: '+str(current_eval_score2))
                    best_eval_flag = True
                    best_eval_score1 = current_eval_score1
                    best_eval_score2 = current_eval_score2

          # CHECKPOINT
            if best_eval_flag:
                print(text)
                if args.checkpoints:
                    model.save_session(args.eval_cnt)
          # EARLY STOPPING
                print("Improvement -> reset early stopping counter")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.epochs_no_improve_stop:
                    print("\nEARLY STOPPING - {} epochs without improvement".format(epochs_no_improve))
                    terminate(args)
                    break


def _parse_args():
    parser = argparse.ArgumentParser()
  # EXPERIMENT
    parser.add_argument("--experiment_name", default=None, help="under folder data/experiments/")
    parser.add_argument("--training_name", default=None, help="if None, time is used")
    parser.add_argument("--bert_size", default=config.bert_size_base_string, help="Bert model size")
    parser.add_argument("--comment", default='', help="any comment that describes the experiment (for log)")

  # TRAINING
    parser.add_argument("--steps_before_evaluation", type=int, default=1000)
#    parser.add_argument("--evaluation_minutes", type=int, default=10, help="every # minutes run an evaluation epoch")

    parser.add_argument("--allspans", dest="allspans", action="store_true")
    parser.add_argument("--no_allspans", dest="allspans", action="store_false")
    parser.set_defaults(allspans=True)

    parser.add_argument("--checkpoints", dest="checkpoints", action="store_true")
    parser.add_argument("--no_checkpoints", dest="checkpoints", action="store_false")
    parser.set_defaults(checkpoints=True)

    parser.add_argument("--dataset_shuffle_buffer_size", type=int, default=1000, help="used in util for dataset read")

    parser.add_argument("--max_checkpoints", type=int, default=3, help="maximum number of checkpoints to keep")

    parser.add_argument("--continue_training", dest="continue_training", action="store_true",
                        help="If true then restore the previous arguments and continue the training "
                             "(i.e. only the experiment_name and training_name used from here)"
                             "Retrieve values from latest checkpoint.")
    parser.add_argument("--no_continue_training", dest="continue_training", action="store_false")
    parser.set_defaults(continue_training=False)

   # early stopping
    parser.add_argument("--epochs_no_improve_stop", type=int, default=10, help="early stopping")
    parser.add_argument("--improvement_threshold", type=float, default=0, help="early stopping")

  # DATASETS
    parser.add_argument("--train_datasets", default="aida_train")
    parser.add_argument("--eval_datasets", default="aida_train-aida_dev-aida_test")
    parser.add_argument("--eval_datasets_validation", default="1", help="i-j-... ~ positions in --eval_datasets, used"
                                                         "to compute optimal thr and for early stopping ")

  # MODEL

    # HYPERPARAMETERS
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_cand_ent", type=int, default=None, help="max candidates for span (reducing memory")
    parser.add_argument("--max_span_width", type=int, default=10)
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5, help="keep probability (tf.nn.dropout used)")
    parser.add_argument("--clip", type=float, default=-1, help="gradient clipping, if negative then no clipping")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=-1,
                        help="if negative then no decay, else each epoch multiply lr by given number (0,1>")
    parser.add_argument("--lr_method", default="adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--gamma", type=float, default=0.2, help="\gamma, margin parameter (max-margin objective)")
    parser.add_argument("--hardcoded_thr", type=float, default=None,
                        help="if specified, we don't calculate optimal thr, but use this one.")
    parser.add_argument("--train_ent_vecs", dest='train_ent_vecs', action='store_true')
    parser.add_argument("--no_train_ent_vecs", dest='train_ent_vecs', action='store_false')
    parser.set_defaults(train_ent_vecs=False)

    # MODEL COMPONENTS
    parser.add_argument("--global_model", dest="global_model", action="store_true", help="use global information")
    parser.add_argument("--no_global_model", dest="global_model", action="store_false")
    parser.set_defaults(global_model=True)

    parser.add_argument("--local_score_components", default="pem_similarity_attention",
                         help="use any combination: "
                              "e.g pem_similarity_attention, pem_similarity, ...")

    parser.add_argument("--global_score_components", default="local",
                        help="use any combination, global is added automatically if args.global_model"
                             "pem_local, local,  pem")

    # char embeddings
    parser.add_argument("--use_chars", dest="use_chars", action="store_true", help="use character embeddings or not")
    parser.add_argument("--no_use_chars", dest="use_chars", action="store_false")
    parser.set_defaults(use_chars=True)
    parser.add_argument("--char_lstm_units", type=int, default=50, help="dimension of char embeddings")

    # entity embeddings
    parser.add_argument("--ent_vecs_dropout", dest="ent_vecs_dropout", action="store_true")
    parser.add_argument("--no_ent_vecs_dropout", dest="ent_vecs_dropout", action="store_false")
    parser.set_defaults(ent_vecs_dropout=True)

    # context embeddings - LSTM on all document
    parser.add_argument("--context_emb", dest="context_emb", action="store_true")
    parser.add_argument("--no_context_emb", dest="context_emb", action="store_false")
    parser.set_defaults(context_emb=True)

    # parser.add_argument("--context_lstm_units", type=int, default=150)

    # span embeddings
#    parser.add_argument("--span_lstm_units", type=int, default=2*(config.embeddings_size+2*128))
    parser.add_argument("--span_emb", default="head_boundaries",
                        help="boundaries for start+end and/or head for 'head' mechanism")

    parser.add_argument("--span_using_context_emb", dest="span_using_context_emb", action="store_true")
    parser.add_argument("--no_span_using_context_emb", dest="span_using_context_emb", action="store_false")
    parser.set_defaults(span_using_context_emb=True)

    # parser.add_argument("--entity_span_share_lstm", dest="entity_span_share_lstm", action="store_true")
    # parser.add_argument("--no_entity_span_share_lstm", dest="entity_span_share_lstm", action="store_false")
    # parser.set_defaults(entity_span_share_lstm=False)


    # local attention
    parser.add_argument("--attention_K", type=int, default=100, help="K from left and K from right, in total 2K")
    parser.add_argument("--attention_R", type=int, default=30, help="top R words from 2K window ")
    parser.add_argument("--attention_AB", dest="attention_AB", action="store_true")
    parser.add_argument("--no_attention_AB", dest="attention_AB", action="store_false")
    parser.set_defaults(attention_AB=True)

    parser.add_argument("--attention_using_context_emb", dest="attention_using_context_emb", action="store_true")
    parser.add_argument("--no_attention_using_context_emb", dest="attention_using_context_emb", action="store_false")
    parser.set_defaults(attention_using_context_emb=True)

    parser.add_argument("--attention_ent_vecs_no_dropout", dest="attention_ent_vecs_no_dropout", action="store_true")
    parser.add_argument("--no_attention_ent_vecs_no_dropout", dest="attention_ent_vecs_no_dropout", action="store_false")
    parser.set_defaults(attention_ent_vecs_no_dropout=True)

    parser.add_argument("--attention_max_cand_ent", type=int, default=None,
                        help="use only the top x number of entities for reducing noise and save memory")

    # global
    parser.add_argument("--global_thr", type=float, default=0, help="\gamma', candidates with local score >= thr "
                        "participate in global voting")
#    parser.add_argument("--global_norm_or_mean", default="norm")
    # global - mask
#    parser.add_argument("--global_mask_scale_each_span_voters_to_one",
#                        dest="global_mask_scale_each_span_voters_to_one", action="store_true")
#    parser.add_argument("--no_global_mask_scale_each_span_voters_to_one",
#                        dest="global_mask_scale_each_span_voters_to_one", action="store_false")
#    parser.set_defaults(global_mask_scale_each_span_voters_to_one=False)
#
#    parser.add_argument("--global_mask_based_on_local_score", dest="global_mask_based_on_local_score",
#                        action="store_true")
#    parser.add_argument("--no_global_mask_based_on_local_score", dest="global_mask_based_on_local_score",
#                        action="store_false")
#    parser.set_defaults(global_mask_based_on_local_score=False)

#    parser.add_argument("--global_mask_unambiguous", dest="global_mask_unambiguous", action="store_true")
#    parser.add_argument("--no_global_mask_unambiguous", dest="global_mask_unambiguous", action="store_false")
#    parser.set_defaults(global_mask_unambiguous=False)
#    # global - alternative selections of voters
#    parser.add_argument("--global_topk", type=int, default=None)
#    parser.add_argument("--global_topkthr", type=float, default=None)
#
#    parser.add_argument("--global_topkfromallspans", type=int, default=None)
#
#    parser.add_argument("--global_topkfromallspans_onlypositive", dest="global_topkfromallspans_onlypositive",
#                        action="store_true")
#    parser.add_argument("--no_global_topkfromallspans_onlypositive", dest="global_topkfromallspans_onlypositive",
#                        action="store_false")
#    parser.set_defaults(global_topkfromallspans_onlypositive=False)

    # FFNNs
    parser.add_argument("--attention_context_emb_ffnn", default="0_0",
                        help="int_int ~ hiddenlayers_hiddensize (0_0 ~ projecting without hidden layers)")
    parser.add_argument("--span_emb_ffnn", default="0_0",
                        help="int_int ~ hiddenlayers_hiddensize (0_0 ~ projecting without hidden layers)")
    parser.add_argument("--local_score_ffnn", default="0_0",
                        help="int_int, see span_emb_ffnn")
    parser.add_argument("--global_score_ffnn", default="0_0",
                        help="int_int, see span_emb_ffnn")

    parser.add_argument("--ffnn_dropout", dest="ffnn_dropout", action="store_true")
    parser.add_argument("--no_ffnn_dropout", dest="ffnn_dropout", action="store_false")
    parser.set_defaults(ffnn_dropout=True)

    # p_e_m
    parser.add_argument("--pem_log", dest="pem_log", action="store_true")
    parser.add_argument("--no_pem_log", dest="pem_log", action="store_false")
    parser.set_defaults(pem_log=True)

    parser.add_argument("--gpem_log", dest="gpem_log", action="store_true")
    parser.add_argument("--no_gpem_log", dest="gpem_log", action="store_false")
    parser.set_defaults(pem_log=True)

    args = parser.parse_args()

  # OUTPUT FOLDERS
    if args.training_name is None:
        from datetime import datetime
        args.training_name = "{:%d_%m_%Y____%H_%M}".format(datetime.now())

    args.output_folder = (config.base_folder/"data/experiments"/args.experiment_name/
                          "training_folder"/args.training_name)
    if not args.output_folder.exists():
        os.makedirs(args.output_folder)
    else:
        if args.continue_training:
            print("continue training...")
            train_args = util.load_train_args(args.output_folder, "train_continue")
            return train_args
        else:
            print("!!!!!\n Training folder: ", args.output_folder, "already exists and args.continue_training=False.")
            exit()
    args.checkpoints_folder = args.output_folder/"checkpoints"
    args.summaries_folder = args.output_folder/"summaries"
    if not args.summaries_folder.exists():
        os.makedirs(args.summaries_folder)

  # OTHERS
    args.eval_cnt = 0
    args.zero = 1e-6
    args.running_mode = "train"

  # PROCESS DATASET ARGS
    args.train_datasets = args.train_datasets.split("-") if args.train_datasets != "" else None
    args.eval_datasets = args.eval_datasets.split("-") if args.eval_datasets != "" else None
    args.eval_datasets_validation = [int(x) for x in args.eval_datasets_validation.split('-')]

  # PROCESS FFNN AND BOUNDARIES ARGS
    args.attention_context_emb_ffnn = [int(x) for x in args.attention_context_emb_ffnn.split('_')]
    args.span_emb_ffnn = [int(x) for x in args.span_emb_ffnn.split('_')]
    args.local_score_ffnn = [int(x) for x in args.local_score_ffnn.split('_')]
    args.global_score_ffnn = [int(x) for x in args.global_score_ffnn.split('_')]

    return args


def log_args(args):
    with open(args.output_folder/"train_args.txt", 'w') as fout:
        attrs = vars(args)
        fout.write('\n'.join("%s: %s" % item for item in attrs.items()))
    with open(args.output_folder/"train_args.pickle", "wb") as handle:
        pickle.dump(args, handle)


def terminate(args):
    tee.close()
    with open(args.output_folder/"train_args.pickle", "wb") as handle:
        pickle.dump(args, handle)


def main(args):
    print(args)
    log_args(args)
    tee = Tee(args.output_folder/"log.txt", 'a')
    try:
        train(args)
    except KeyboardInterrupt:
        terminate(args)

if __name__ == "__main__":
    main(_parse_args())
