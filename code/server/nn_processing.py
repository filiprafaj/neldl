from time import sleep
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

import model.config as config
from model.model import Model
from evaluation.metrics import _filtered_spans_and_gm_gt_list
import preprocessing.util as util

class StreamingSamples(object):
    def __init__(self):
        """
        self.words, self.words_len, self.chars, self.chars_len,
        self.begin_span, self.end_span, self.spans_len,
        self.cand_entities, self.cand_entities_scores,
        self.cand_entities_len,
        self.word_embeddings
        """
        # those are not used here
        # self.chunk_id, self.ground_truth, self.ground_truth_len, self.begin_gm, self.end_gm
        # self.cand_entities_labels,

        self.sample = None
        self.empty = True

    def new_sample(self, sample):
        self.sample = sample
        self.empty = False

    def gen(self):
        while True:
            if not self.empty:
                self.empty = True
                yield self.sample
            else:
                print("sleep")
                sleep(0.5)


class NNProcessing(object):
    def __init__(self, train_args, args, fetchCandidateEntities, bert):
        self.gm_idx_errors = 0
        self.args = args
        self.allspans_mode = True
        self.fetchCandidateEntities = fetchCandidateEntities
        self.bert = bert
        self.streaming_samples = StreamingSamples()

        self.wikiid2nnid = util.load_wikiid2nnid(train_args.experiment_name)
        self.nnid2wikiid = util.reverse_dict(self.wikiid2nnid, unique_values=True)


      # INPUT PIPELINE #################################################
        # words, words_len, chars, chars_len
        # begin_span, end_span, spans_len
        # cand_entities, cand_entities_scores, cand_entities_len
        # word_embeddings
        ds = tf.data.Dataset.from_generator(
                 self.streaming_samples.gen,
                 (tf.int64, tf.int64, tf.int64, tf.int64,
                  tf.int64, tf.int64, tf.int64,
                  tf.int64, tf.float32, tf.int64,
                  tf.float32),
                 (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None]),
                  tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]),
                  tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
                  tf.TensorShape([None,None])))

        next_element = ds.make_one_shot_iterator().get_next()

        cand_entities = next_element[7]
        # expand the dims to match the training that has batch dimension
        next_element = [tf.expand_dims(t, 0) for t in next_element]

        # None ~ chunk_id, cand_entities_labels,
        #        ground_truth, ground_truth_len, begin_gm, end_gm
        next_element = [None] + next_element[:-2] + [ None, next_element[-2], None, None, None, None,
                                                     next_element[-1]]

      # RESTORE MODEL
        model = Model(train_args, next_element)
        model.build()
        checkpoint_model_num = model.restore_session()
        self.model = model
        print("\nLOADED Model:", train_args.output_folder)

        # optimal threshold recovery from log file
        # based on the checkpoint selected
        self.thr = retrieve_optimal_threshold_from_logfile(train_args.output_folder, checkpoint_model_num)
        print("threshold from log file = ", self.thr)

        _, self.wiki_id_name_map = util.load_wiki_name_id_map(print_stats=False)

        with open(config.base_folder/"data/experiments"/args.experiment_name/"word_char_maps.pickle", 'rb') as handle:
            self.word2id, _, self.char2id, _, _, _ = pickle.load(handle)


    def process(self, text, given_spans, redirections, threshold, experiment_name=None):
        print("\nNNProcessing process:")
        if experiment_name:
            self.wikiid2nnid_tmp = util.load_wikiid2nnid(experiment_name)
            self.nnid2wikiid_tmp = util.reverse_dict(self.wikiid2nnid_tmp, unique_values=True)
#            print(self.wikiid2nnid_tmp)#DEBUG
        wikiid2nnid = self.wikiid2nnid_tmp if experiment_name else self.wikiid2nnid
        nnid2wikiid = self.nnid2wikiid_tmp if experiment_name else self.nnid2wikiid
        if not experiment_name: experiment_name=self.args.experiment_name

        if threshold: self.thr = threshold
        self.given_spans = given_spans
        idx = 0
      # PROCESS TEXT
        chunk_words = []
        words2charidx = []
        if self.given_spans:
            beginidx2wordnum = dict()
            endidx2wordnum = dict()

        original_words = word_tokenize(text)
        original_words = ['"' if w in ["``","''"] else w for w in original_words]

        for word_num, word in enumerate(original_words):
            chunk_words.append(word)
            begin_idx = text.find(word, idx)
            end_idx = begin_idx + len(word)
            idx = end_idx
            assert(len(words2charidx) == word_num)
            words2charidx.append((begin_idx, end_idx))  # [..)

            if self.given_spans:
                beginidx2wordnum[begin_idx] = word_num
                endidx2wordnum[end_idx] = word_num

      # TEXT EMBEDDINGS ################################################
        sentences = [word_tokenize(sent) for sent in sent_tokenize(text)]
        if len(max(sentences, key=len))>512:
            raise ValueError("too long sentence")
        word_embeddings = []
        for i, embed in enumerate(self.bert.bert_embeddings(sentences)):
            print("Processed {}/{} sentences.".format(i + 1, len(sentences)))
            word_embeddings.append(embed)
        word_embeddings = np.concatenate(word_embeddings)

      # CANDIDATE ENTITIES #############################################
      # GOLDSPANS MODE - use the spans provided
        if self.given_spans:
          # CONVERT given_spans (begin, length) in characters
          #         to given_spans (begin, end) in word_num
            self.given_spans = sorted(self.given_spans)
            begin_gm = []
            end_gm = []
            for span in self.given_spans:
                try:
                    begin_idx, length = span
                    end_idx = begin_idx+length
                    if begin_idx not in beginidx2wordnum:
                        begin_idx = self.nearest_idx(begin_idx, beginidx2wordnum.keys())
                    if end_idx not in endidx2wordnum:
                        end_idx = self.nearest_idx(end_idx, endidx2wordnum.keys())
                    if (begin_idx, end_idx-begin_idx) != span:
                        print("given span:", text[span[0]:span[0]+span[1]], " new span:",
                              text[begin_idx:end_idx])
                    begin_gm.append(beginidx2wordnum[begin_idx])
                    end_gm.append(endidx2wordnum[end_idx]+1)
                except KeyError:
                    print("Exception: KeyError")
                    print("original_words =", original_words)
                    print("chunk_words =", chunk_words)
                    print("begin={}, length={}, left={}, span={}, right={}".format(
                        begin_idx,
                        length,
                        text[begin_idx-30:begin_idx],
                        text[begin_idx:begin_idx+length],
                        text[begin_idx+length:begin_idx+length+30]))
                    print("text =", text)
                    print("begin= {}".format("in" if begin_idx in beginidx2wordnum else "out"))
                    print("end=   {}".format("in" if begin_idx + length in endidx2wordnum else "out"))

            self.fetchCandidateEntities.set_goldspans_mode()

            (
             begin_spans,
             end_spans,
             cand_entities,
             cand_entities_scores
            ) = self.fetchCandidateEntities.process(chunk_words, begin_gm=begin_gm, end_gm=end_gm,
                                                    redirections=redirections)

            if self.args.person_coreference:
                (
                 begin_spans,
                 end_spans,
                 cand_entities,
                 cand_entities_scores
                ) = self.fetchCandidateEntities.person_coreference(chunk_words,
                                                                   begin_spans,
                                                                   end_spans,
                                                                   cand_entities,
                                                                   cand_entities_scores,
                                                                   begin_gm=begin_gm,
                                                                   end_gm=end_gm)

#            print(cand_entities)#DEBUG
#            print(cand_entities_scores)#DEBUG
          # FILTER ENTITIES NOT IN UNIVERSE
            cand_entities_filtered = []
            cand_entities_scores_filtered = []
            for ce, sc in zip(cand_entities, cand_entities_scores):
                ce_filtered = []
                sc_filtered = []
                for e, s in zip(ce, sc):
                    if e in wikiid2nnid:
                        ce_filtered.append(wikiid2nnid[e])
                        sc_filtered.append(s)
                cand_entities_filtered.append(ce_filtered)
                cand_entities_scores_filtered.append(sc_filtered)

            cand_entities = cand_entities_filtered
            cand_entities_scores = cand_entities_scores_filtered
#            print(cand_entities) #DEBUG
#            print(cand_entities_scores)#DEBUG

            cand_entities_len = [len(t) for t in cand_entities]

          # PADDING
            cand_entities = list_of_lists_to_2darray(cand_entities, np.int64)
            cand_entities_scores = list_of_lists_to_2darray(cand_entities_scores, np.float32)

      # ALLSPANS MODE
        else:
            separation_indexes = np.cumsum([len(x) for x in sentences])
            self.fetchCandidateEntities.set_allspans_mode()
            (
             begin_spans,
             end_spans,
             cand_entities,
             cand_entities_scores
            ) = self.fetchCandidateEntities.process(chunk_words, separation_indexes=separation_indexes,
                                                    redirections=redirections)
            if self.args.person_coreference:
                (
                 begin_spans,
                 end_spans,
                 cand_entities,
                 cand_entities_scores
                ) = self.fetchCandidateEntities.person_coreference(chunk_words,
                                                                   begin_spans,
                                                                   end_spans,
                                                                   cand_entities,
                                                                   cand_entities_scores,
                                                                   separation_indexes=separation_indexes)

          # FILTER ENTITIES NOT IN UNIVERSE
            cand_entities_filtered = []
            cand_entities_scores_filtered = []
            for ce, sc in zip(cand_entities, cand_entities_scores):
                ce_filtered = []
                sc_filtered = []
                for e, s in zip(ce, sc):
                    if e in wikiid2nnid:
                        ce_filtered.append(wikiid2nnid[e])
                        sc_filtered.append(s)
                cand_entities_filtered.append(ce_filtered)
                cand_entities_scores_filtered.append(sc_filtered)

            cand_entities = cand_entities_filtered
            cand_entities_scores = cand_entities_scores_filtered

            cand_entities_len = [len(t) for t in cand_entities]

          # PADDING
            cand_entities = list_of_lists_to_2darray(cand_entities, np.int64)
            cand_entities_scores = list_of_lists_to_2darray(cand_entities_scores, np.float32)

        if not begin_spans:
            return []  # this document has no annotation

      # WORDS AND CHARS ################################################
        words = []
        chars = []
        for word in chunk_words:
            words.append(self.word2id[word] if word in self.word2id
                         else self.word2id["<wunk>"])
            chars.append([self.char2id[c] if c in self.char2id else self.char2id["<u>"]
                          for c in word])
        chars_len = [len(word) for word in chars]
        chars = list_of_lists_to_2darray(chars, np.int32)

      # COMPUTE RESULT, FILTER AND RETURN RESPONSE #####################
        new_sample = (words, len(words), chars, chars_len,
                      begin_spans, end_spans, len(begin_spans),
                      cand_entities, cand_entities_scores, cand_entities_len,
                      word_embeddings)
        self.streaming_samples.new_sample(new_sample)

        result_l = self.model.sess.run([self.model.final_scores,
                                        self.model.cand_entities_len, self.model.cand_entities,
                                        self.model.begin_span, self.model.end_span, self.model.spans_len],
                                       feed_dict={self.model.dropout_keep_prob: 1})
        # filter spans
        filtered_spans, _ = _filtered_spans_and_gm_gt_list(0, *result_l, None, None, None, [0], [len(words)])

        print("gm_idx_errors =", self.gm_idx_errors)

        response = []
        if self.args.each_entity_only_once:
            from operator import itemgetter
            filtered_spans = sorted(filtered_spans, key=itemgetter(1))
            used_entities = set()
            for span in filtered_spans:
                score, begin_idx, end_idx, nnid = span
                if score >= self.thr and nnid not in used_entities:
                    self._add_response_span(response, span, words2charidx, experiment_name)
                    used_entities.add(nnid)
        else:
            for span in filtered_spans:
                print(span)#DEBUG
                score, begin_idx, end_idx, nnid = span
                if score >= self.thr:
                    self._add_response_span(response, span, words2charidx, experiment_name)

        return response


    def nearest_idx(self, key, values):
        self.gm_idx_errors += 1
        # find the value in values that is nearest to key
        nearest_value = None
        min_distance = 1e+6
        for value in values:
            if abs(key - value) < min_distance:
                nearest_value = value
                min_distance = abs(key-value)
        return nearest_value


    def nearest_given_span(self, begin_idx, end_idx):
        """
        [begin_idx, end_idx)
        - end_idx points to character just after mention
        """
        min_distance = 1e+6
        nearest_idxes = (-1, -1)
        for (begin, length) in self.given_spans:
            distance = abs(begin_idx - begin) + abs(end_idx - (begin + length))
            if distance < min_distance:
                nearest_idxes = (begin, begin + length)
                min_distance = distance
        return nearest_idxes


    def _add_response_span(self, response, span, words2charidx, experiment_name):
        score, begin_idx, end_idx, nnid = span
        begin = words2charidx[begin_idx][0]
        # the word at begin_idx begins with this character
        end = words2charidx[end_idx-1][1]
        # the word at end_idx-1 (last inside) ends at this character
        wikiid = self.nnid2wikiid_tmp[nnid] if experiment_name!=self.args.experiment_name else self.nnid2wikiid[nnid]
        wikiname = self.wiki_id_name_map[wikiid].replace(' ', '_')

        if not self.allspans_mode: # try to match given span
            begin, end = self.nearest_given_span(begin, end)
        response.append((begin, end-begin, wikiname))



def list_of_lists_to_2darray(a, dtype):
    '''
    With padding zeros.
    '''
    b = np.zeros([len(a), len(max(a, key=len))], dtype=dtype)
    for i, j in enumerate(a):
        b[i][0:len(j)] = j
    return b


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


if __name__ == "__main__":
    pass

