import argparse
import os
import pickle
from collections import namedtuple

import model.config as config
import tensorflow as tf
from model.util import Tee

import preprocessing.util as util
from preprocessing.samples_generator import (AllspansSample, GoldspansSample,
                                             SamplesGenerator)

SampleEncoded = namedtuple("SampleEncoded",
                           ["chunk_id",
                            "words", "words_len", # list,  scalar
                            "chars", "chars_len", # list of lists,  list
                            "begin_spans", "end_spans",  "spans_len", # the first 2 are lists, last is scalar
                            "cand_entities", "cand_entities_scores", "cand_entities_labels",  # lists of lists
                            "cand_entities_len",  # list
                            "ground_truth", "ground_truth_len",
                            "begin_gm", "end_gm", # list
                            "chunk_embeddings"])

class EncoderGenerator(object):
    """receives samples Train or Test samples and encodes everything to numbers ready to
    be transformed to tfrecords. Also filters out candidate entities that are not in the
    entity universe."""
    def __init__(self, args):
        self._generator = SamplesGenerator(args)
        if (config.base_folder/"data/experiments"/args.experiment_name/"word_char_maps.pickle").is_file():
            self._word2id, self._char2id = util.restore_id_maps(args.experiment_name)
        else:
            self._word2id, self._char2id = util.create_id_maps_and_word_embeddigns(args)
        self._wikiid2nnid = util.load_wikiid2nnid(args.experiment_name)

    def set_goldspans_mode(self):
        self._generator.set_goldspans_mode()

    def set_allspans_mode(self):
        self._generator.set_allspans_mode()

    def is_goldspans_mode(self):
        return self._generator.is_goldspans_mode()

    def is_allspans_mode(self):
        return self._generator.is_allspans_mode()

    def process(self, filepath):
        ground_truth_errors_cnt = 0
        cand_entities_not_in_universe_cnt = 0
        samples_with_errors = 0
        for sample in self._generator.process(filepath):
            words = []
            chars = []
            for word in sample.chunk_words:
                words.append(self._word2id[word] if word in self._word2id else self._word2id["<wunk>"])
                chars.append([self._char2id[c] if c in self._char2id else self._char2id["<u>"]
                              for c in word])

            chars_len = [len(word) for word in chars]
            ground_truth_enc = [self._wikiid2nnid[gt] if gt in self._wikiid2nnid else self._wikiid2nnid["<u>"]
                                for gt in sample.ground_truth]
            ground_truth_errors_cnt += ground_truth_enc.count(self._wikiid2nnid["<u>"])

            if len(sample.begin_gm) != len(sample.end_gm) or len(sample.begin_gm) != len(ground_truth_enc):
                samples_with_errors += 1
                continue
          # GOLDSPANS
            if isinstance(sample, GoldspansSample):
                (
                cand_entities,
                cand_entities_scores,
                cand_entities_labels,
                not_in_universe_cnt
                ) = self._encode_cand_entities_and_labels(sample.cand_entities, sample.cand_entities_scores,
                                                          sample.ground_truth)

                yield SampleEncoded(chunk_id=sample.chunk_id,
                                    words=words, words_len=len(words),
                                    chars=chars, chars_len=chars_len,
                                    begin_spans=sample.begin_gm, end_spans=sample.end_gm,
                                    spans_len=len(sample.begin_gm),
                                    cand_entities=cand_entities, cand_entities_scores=cand_entities_scores,
                                    cand_entities_labels=cand_entities_labels,
                                    cand_entities_len=[len(t) for t in cand_entities],
                                    ground_truth=ground_truth_enc, ground_truth_len=len(sample.ground_truth),
                                    begin_gm=[], end_gm=[],
                                    chunk_embeddings=sample.chunk_embeddings)
          # ALLSAMPLES
            elif isinstance(sample, AllspansSample):
                if len(sample.begin_spans) != len(sample.end_spans):
                    samples_with_errors += 1
                    continue
                span_ground_truth = []
                # ~ for each span gt, or -1 if this span is not a gm
                gm_spans = list(zip(sample.begin_gm, sample.end_gm))
                for left, right in zip(sample.begin_spans, sample.end_spans):
                    if (left, right) in gm_spans:
                        span_ground_truth.append(sample.ground_truth[gm_spans.index((left, right))])
                    else:
                        span_ground_truth.append(-1) # span is not a gm

                (
                cand_entities,
                cand_entities_scores,
                cand_entities_labels,
                not_in_universe_cnt
                ) = self._encode_cand_entities_and_labels(sample.cand_entities, sample.cand_entities_scores,
                                                          span_ground_truth)

                yield SampleEncoded(chunk_id=sample.chunk_id,
                                    words=words, words_len=len(words),
                                    chars=chars, chars_len=chars_len,
                                    begin_spans=sample.begin_spans, end_spans=sample.end_spans,
                                    spans_len=len(sample.begin_spans),
                                    cand_entities=cand_entities, cand_entities_scores=cand_entities_scores,
                                    cand_entities_labels=cand_entities_labels,
                                    cand_entities_len=[len(t) for t in cand_entities],
                                    ground_truth=ground_truth_enc, ground_truth_len=len(sample.ground_truth),
                                    begin_gm=sample.begin_gm, end_gm=sample.end_gm,
                                    chunk_embeddings=sample.chunk_embeddings)

            cand_entities_not_in_universe_cnt += not_in_universe_cnt
        print(" ground_truth_errors_cnt =", ground_truth_errors_cnt)
        print(" cand_entities_not_in_universe_cnt =", cand_entities_not_in_universe_cnt)
        print(" encoder samples_with_errors =", samples_with_errors)



    def _encode_cand_entities_and_labels(self, cand_entities_p, cand_entities_scores_p, ground_truth_p):
        """
        With cand_entities (list of lists), scores (list of lists)
        and ground_truth (list) does the following:

          1) removes cand. ent. that are not in our universe
          2) creates a label 0, 1 denoting if candidate is correct
             (i.e. the span is a gold mention and the candidate is gt)

        Returns the filtered cand_entities, scores and the labels
        """
        cand_entities = []
        cand_entities_scores = []
        cand_entities_labels = []
        not_in_universe_cnt = 0
        for cand_ent_l, cand_scores_l, gt in zip(cand_entities_p, cand_entities_scores_p, ground_truth_p):
            ent_l = []
            score_l = []
            label_l = []
            for cand_ent, score in zip(cand_ent_l, cand_scores_l):
                if cand_ent in self._wikiid2nnid: # if in our universe
                    ent_l.append(self._wikiid2nnid[cand_ent])
                    score_l.append(score)
                    label_l.append(1 if cand_ent == gt else 0)
                else:
                    not_in_universe_cnt += 1
            cand_entities.append(ent_l)
            cand_entities_scores.append(score_l)
            cand_entities_labels.append(label_l)
        return cand_entities, cand_entities_scores, cand_entities_labels, not_in_universe_cnt


class TFRecordsGenerator(object):
    def __init__(self, args):
        self._generator = EncoderGenerator(args)
        self.args=args

    def set_goldspans_mode(self):
        self._generator.set_goldspans_mode()

    def set_allspans_mode(self):
        self._generator.set_allspans_mode()

    def is_goldspans_mode(self):
        return self._generator.is_goldspans_mode()

    def is_allspans_mode(self):
        return self._generator.is_allspans_mode()

    @staticmethod
    def serialize_example(sample):

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """value is a simple integer."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _int64_feature_list(values):
            """
            values is a list of integers like
            returns a feature list where each feature has one number
            """
            return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

        def _int64list_feature_list(values):
            """
            e.g. chars = [[1,2,3], [4,5], [6], [7,8], [9,10,11,12]]
            a feature list where each feature can have variable
            number of ements
            """
            def _int64list_feature(value):
                """value is a list of integers."""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            return tf.train.FeatureList(feature=[_int64list_feature(v) for v in values])

        def _floatlist_feature_list(values):
            """
            e.g. [[0.1,0.2,0.3], [0.4,0.5]]
            a feature list where each feature can have variable
            number of ements
            """
            def _floatlist_feature(value):
                """value is a list of integers."""
                return tf.train.Feature(float_list=tf.train.FloatList(value=value))
            return tf.train.FeatureList(feature=[_floatlist_feature(v) for v in values])

        context = tf.train.Features(feature={
                "chunk_id": _bytes_feature(sample.chunk_id.encode("utf-8")),
                "words_len": _int64_feature(sample.words_len),
                "spans_len": _int64_feature(sample.spans_len),
                "ground_truth_len": _int64_feature(sample.ground_truth_len)
        })
        feature_list = {
                "words": _int64_feature_list(sample.words),
                "chars": _int64list_feature_list(sample.chars),
                "chars_len": _int64_feature_list(sample.chars_len),
                "begin_span": _int64_feature_list(sample.begin_spans),
                "end_span": _int64_feature_list(sample.end_spans),
                "cand_entities": _int64list_feature_list(sample.cand_entities),
                "cand_entities_scores": _floatlist_feature_list(sample.cand_entities_scores),
                "cand_entities_labels": _int64list_feature_list(sample.cand_entities_labels),
                "cand_entities_len": _int64_feature_list(sample.cand_entities_len),
                "ground_truth": _int64_feature_list(sample.ground_truth),
                "chunk_embeddings": _floatlist_feature_list(sample.chunk_embeddings)
        }
        if True:
            feature_list["begin_gm"] = _int64_feature_list(sample.begin_gm)
            feature_list["end_gm"] = _int64_feature_list(sample.end_gm)
        feature_lists = tf.train.FeatureLists(feature_list=feature_list)

        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
        return sequence_example.SerializeToString()


    def process(self, filepath, output_folder):
        print("TFRecordsGenerator processing file: ", filepath)

        filename = filepath.name[:-4]
        # the name of the dataset, remove ".txt")

        output_folder = output_folder/("tfrecords_goldspans" if self.is_goldspans_mode() else "tfrecords_allspans")
        if not output_folder.exists():
            output_folder.mkdir(parents=True, exist_ok=True)

        writer = tf.python_io.TFRecordWriter(str(output_folder/filename))
        records_cnt = 0
        for sample in self._generator.process(filepath):
            serialized_example = self.serialize_example(sample)
            writer.write(serialized_example)
            records_cnt += 1
        writer.close()
        print(" records_cnt = ", records_cnt, "\n")
        print(" tfrecords saved to ", str(output_folder/filename), "\n")


def create_tfrecords(args, datasets, output_folder):
    print("\nDATASETS for TFRecordsGenerator: ", [d.name for d in datasets])

    tfrecords_generator = TFRecordsGenerator(args)
    print("\nMODE goldspans:")
    tfrecords_generator.set_goldspans_mode()
    for d in datasets:
        tfrecords_generator.process(d, output_folder)
    print("\nMODE allspans:")
    tfrecords_generator.set_allspans_mode()
    for d in datasets:
        tfrecords_generator.process(d, output_folder)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_name", help="under folder data/experiments/")
    parser.add_argument("--bert_casing", default=config.bert_uncased_string,
                        help="to use 'cased' or 'uncased' bert embeddings of files")
    parser.add_argument("--bert_size", default=config.bert_size_base_string, help="Bert model size")
    parser.add_argument("--max_cand_ent", type=int, default=config.max_cand_ent,
                        help="how many cand entities to keep for each span")
    parser.add_argument("--word_freq_thr", type=int, default=1, help="words with less freq not included in vocabulary")
    parser.add_argument("--char_freq_thr", type=int, default=1)
    parser.add_argument("--max_span_width", type=int, default=config.max_span_width,
                        help="in allspans are spans of len <= max considered candidate to be linked")


    parser.add_argument("--redirections", dest="redirections", action="store_true")
    parser.add_argument("--no_redirections", dest="redirections", action="store_false")
    parser.set_defaults(redirections=True)

    parser.add_argument("--lowercase_redirections", dest="lowercase_redirections", action="store_true")
    parser.add_argument("--no_lowercase_redirections", dest="lowercase_redirections", action="store_false")
    parser.set_defaults(lowercase_redirections=False)

    parser.add_argument("--lowercase_p_e_m", dest="lowercase_p_e_m", action="store_true")
    parser.add_argument("--no_lowercase_p_e_m", dest="lowercase_p_e_m", action="store_false")
    parser.set_defaults(lowercase_p_e_m=False)

    parser.add_argument("--lowercase_spans", dest="lowercase_spans", action="store_true")
    parser.add_argument("--no_lowercase_spans", dest="lowercase_spans", action="store_false")
    parser.set_defaults(lowercase_spans=False)

    parser.add_argument("--person_coreference", dest="person_coreference", action="store_true")
    parser.add_argument("--no_person_coreference", dest="person_coreference", action="store_false")
    parser.set_defaults(person_coreference=True)

    parser.add_argument("--person_coreference_merge", dest="person_coreference_merge", action="store_true")
    parser.add_argument("--no_person_coreference_merge", dest="person_coreference_merge", action="store_false")
    parser.set_defaults(person_coreference_merge=True)

    parser.add_argument("--calculate_stats", dest="calculate_stats", action="store_true")
    parser.add_argument("--no_calculate_stats", dest="calculate_stats", action="store_false")
    parser.set_defaults(calculate_stats=True)

    return parser.parse_args()


def log_args(args, folder):
    if not os.path.exists(folder):
        folder.mkdir(parents=True, exist_ok=True)
    with open(folder/("create_tfrecords_args.txt"), 'w') as fout:
        attrs = vars(args)
        fout.write('\n'.join("%s: %s" % item for item in attrs.items()))
    with open(folder/("create_tfrecords_args.pickle"), "wb") as handle:
        pickle.dump(args, handle)


def main(args):
    print(args)
    output_folder = config.base_folder/"data/experiments"/args.experiment_name
    args_and_logs_folder = output_folder/"args_and_logs"
    if not args_and_logs_folder.exists():
        args_and_logs_folder.mkdir(parents=True, exist_ok=True)

    log_args(args, args_and_logs_folder)

    datasets = [d for d in (config.base_folder/"data/base_data/new_datasets").glob("*.txt")]

    tee = Tee(args_and_logs_folder/"create_tfrecords_log.txt", 'a')
    create_tfrecords(args, datasets, output_folder)
    tee.close()

if __name__ == "__main__":
    main(_parse_args())
