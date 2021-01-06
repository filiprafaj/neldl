import argparse
import os
import pickle

import model.config as config
import model.train as train
import model.util as util
import tensorflow as tf
from model.model import Model
from model.util import Tee

from evaluation.metrics import Evaluator


def dataset_score(args, model, iterator, dataset_handle, opt_thr, weak_matching, allspans, printPredictions, name=""):
  # INITIALIZE
    if printPredictions:
        printPredictions.file_start(weak_matching, name, opt_thr)
    model.sess.run(iterator.initializer)
    evaluator = Evaluator(opt_thr, name=name)

  # EVALUATE DATASET
    while True:
        try:
            retrieve_l = [model.final_scores,
                          model.cand_entities_len, model.cand_entities,
                          model.begin_span, model.end_span, model.spans_len,
                          model.begin_gm, model.end_gm,
                          model.ground_truth, model.ground_truth_len,
                          model.words_len, model.chunk_id,
                          model.words, model.chars, model.chars_len]

          # SCORES TO GET
            scores_retrieve_l, scores_names_l = [], []
            if model.args.local_score_components.find("similarity") != -1:
                scores_retrieve_l.append(model.similarity_scores)
                scores_names_l.append("similarity")

            if model.args.local_score_components.find("pem") != -1:
                scores_retrieve_l.append(model.pem_scores)
                scores_names_l.append("pem")

            if model.args.local_score_components.find("attention") != -1:
                scores_retrieve_l.append(model.attention_scores)
                scores_names_l.append("attention")

            if model.args.global_model:
                scores_retrieve_l.append(model.final_local_scores)
                scores_names_l.append("final_local_scores")
            scores_retrieve_l.append(model.final_scores)
            scores_names_l.append("final_scores")

            global_pairwise_scores = []
            if args.print_global_voters:
                global_pairwise_scores.append(model.gmask)
                global_pairwise_scores.append(model.pure_entity_embeddings)

            retrieve_l.append(scores_retrieve_l)
            retrieve_l.append(global_pairwise_scores)

          # RUN
            result_l = model.sess.run(retrieve_l,
                                      feed_dict={model.input_handle_ph: dataset_handle,
                                                 model.dropout_keep_prob: 1})
          # METRICS AND PRINTING
            evaluator.tp_fp_fn_and_prediction_printing(*result_l, scores_names_l, weak_matching, allspans,
                                                       printPredictions=printPredictions)
        # END OF DATASET
        except tf.errors.OutOfRangeError:
            if args.print_predictions:
                printPredictions.file_ended()
            print(name)
            evaluator.print_and_log_results(weak_matching)
            break


def evaluate(args, train_args):
  # DATASETS
    eval_datasets, eval_names = train.create_eval_pipelines(args, filenames=args.eval_datasets)

  # FEEDABLE ITERATOR
    input_handle_ph = tf.placeholder(tf.string, shape=[], name="input_handle_ph")
    sample_dataset = eval_datasets[0]
    iterator = tf.data.Iterator.from_string_handle(input_handle_ph, sample_dataset.output_types,
                                                   sample_dataset.output_shapes)
    next_element = iterator.get_next()

  # MODEL
    model = Model(train_args, next_element)
    model.build()
    model.input_handle_ph = input_handle_ph # for access convenience
    checkpoint_model_num = model.restore_session()
    opt_thr = retrieve_optimal_threshold_from_logfile(model.args.output_folder, checkpoint_model_num)
    #print(tf.global_variables())

  # PRINTER
    printPredictions = None
    if args.print_predictions:
        from evaluation.print_predictions import PrintPredictions
        printPredictions = PrintPredictions(args.predictions_folder,
                                            args.experiment_name,
                                            args.gm_bucketing_pem_pos,
                                            args.print_global_voters)

        model_description = ("local: " + model.args.local_score_components +
                             "global: " + model.args.global_score_components if model.args.global_model else '')
        printPredictions.extra_info = "checkpoint={}, opt_thr={}, model={}".format(checkpoint_model_num,
                                                                                   opt_thr, model_description)

  # BASELINE MODEL
    if args.p_e_m_algorithm:
        model.final_scores = model.cand_entities_scores

  # EVALUATE DATASETS
    def eval_dataset_handles(sess, datasets):
        eval_iterators = []
        eval_handles = []
        for dataset in datasets:
            eval_iterator = dataset.make_initializable_iterator()
            eval_iterators.append(eval_iterator)
            eval_handles.append(sess.run(eval_iterator.string_handle()))
        return eval_iterators, eval_handles

    with model.sess as sess:
        print("\nEvaluating datasets - {} mode ".format("weak matching" if args.weak_matching
                                                        else "strong matching"))
        iterators, handles = eval_dataset_handles(sess, eval_datasets)
        for eval_handle, eval_name, eval_it in zip(handles, eval_names, iterators):
            dataset_score(args, model, eval_it, eval_handle, opt_thr, args.weak_matching,
                args.allspans, printPredictions, name=eval_name)


def retrieve_optimal_threshold_from_logfile(model_folder, checkpoint_model_num):
    with open(model_folder/"log.txt", "r") as fin:
        line = next(fin).rstrip()
        while line != "EVALUATION: {}".format(checkpoint_model_num):
            line = next(fin).rstrip()
        while not line.startswith("Best validation threshold = "):
            line = next(fin)
        line = line.split()
        assert line[3] == "=" and line[5] == "with", line
        return float(line[4])


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", help="under data/experiments/")
    parser.add_argument("--training_name")
    parser.add_argument("--checkpoint_model_num", default=None, help="e.g. give '7' if you want checkpoints/model-7")

    parser.add_argument("--allspans", dest="allspans", action="store_true")
    parser.add_argument("--no_allspans", dest="allspans", action="store_false")
    parser.set_defaults(allspans=True)

    parser.add_argument("--weak_matching", dest="weak_matching", action="store_true")
    parser.add_argument("--no_weak_matching", dest="weak_matching", action="store_false")
    parser.set_defaults(weak_matching=False)

    parser.add_argument("--gm_bucketing_pem_pos", default=None, help="0_1_2_7  create bins 0, 1, 2, [3,7], [8,inf)")

    parser.add_argument("--eval_datasets",
                        default="aida_train|aida_dev|aida_test|ace2004|aquaint|clueweb|msnbc|wikipedia")
    parser.add_argument("--eval_datasets_validation", default="1")

    parser.add_argument("--print_predictions", dest="print_predictions", action="store_true",
                        help="for each dataset print the predictions and compares with gt")
    parser.add_argument("--no_print_predictions", dest="print_predictions", action="store_false")
    parser.set_defaults(print_predictions=True)

    parser.add_argument("--print_global_voters", dest="print_global_voters", action="store_true")
    parser.add_argument("--no_print_global_voters", dest="print_global_voters", action="store_false")
    parser.set_defaults(print_global_voters=False)

    parser.add_argument("--p_e_m_algorithm", dest="p_e_m_algorithm", action="store_true",
                        help="Baseline. Doesn't use the NN but only the p_e_m dictionary for its predictions.")
    parser.add_argument("--no_p_e_m_algorithm", dest="p_e_m_algorith", action="store_false")
    parser.set_defaults(p_e_m_algorithm=False)

    args = parser.parse_args()

  # OUTPUT FOLDERS
    args.output_folder = (config.base_folder/"data/experiments"/args.experiment_name/
                          "training_folder"/args.training_name)
    args.checkpoints_folder = args.output_folder/"checkpoints"
    args.predictions_folder = args.output_folder/("predictions_weak_matching" if args.weak_matching
                                                  else "predictions_strong_matching")
    if args.p_e_m_algorithm:
        args.predictions_folder = args.output_folder + "p_e_m_predictions/"
    if args.print_predictions and not args.predictions_folder.exists():
        args.predictions_folder.mkdir(parents=True, exist_ok=True)

    train_args = util.load_train_args(args.output_folder, "evaluate")
    train_args.checkpoint_model_num = args.checkpoint_model_num
    args.batch_size = train_args.batch_size

  # PROCESS DATASET ARGS
    args.eval_datasets = args.eval_datasets.split("|") if args.eval_datasets != "" else None
    args.eval_datasets_validation = [int(x) for x in args.eval_datasets_validation.split('_')]
    args.gm_bucketing_pem_pos = ([int(x) for x in args.gm_bucketing_pem_pos.split('_')]
                                 if args.gm_bucketing_pem_pos else [])

    return args, train_args


def main():
    args, train_args = _parse_args()
    print("args:\n", args)
    print("train_args:\n", train_args)
    if args.weak_matching:
        tee = Tee(args.output_folder/"predictions_weak_log.txt", 'a')
    else:
        tee = Tee(args.output_folder/"predictions_strong_log.txt", 'a')
    evaluate(args, train_args)
    tee.close()

if __name__ == "__main__":
    main()
