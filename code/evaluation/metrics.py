from collections import defaultdict
from operator import itemgetter

import numpy as np
import tensorflow as tf


class Evaluator(object):
    def __init__(self, threshold, name):
        self.threshold = threshold
        self.name = name
        # TP, FP and FN are docid -> counter
        self.TP = defaultdict(int) # gm found and linked to gt
        self.FP = defaultdict(int) # something linked, but not gm-gt
        self.FN = defaultdict(int) # gm-gt missed
        self.docids = set()          # set with all the docid encountered
        self.gm_num = 0

    def gm_add(self, gm_in_batch):
        self.gm_num += gm_in_batch

    def check_tp(self, score, docid):
        if score >= self.threshold:
            self.docids.add(docid)
            self.TP[docid] += 1
            return True
        return False

    def check_fp(self, score, docid):
        if score >= self.threshold:
            self.docids.add(docid)
            self.FP[docid] += 1
            return True
        return False

    def check_fn(self, score, docid):
        if score < self.threshold:
            self.docids.add(docid)
            self.FN[docid] += 1
            return True
        return False

    def _score_computation(self, weak_matching):
        micro_tp, micro_fp, micro_fn = 0, 0, 0
        macro_pr, macro_re = 0, 0

        for docid in self.docids:
            tp, fp, fn = self.TP[docid], self.FP[docid], self.FN[docid]
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

            doc_precision = tp / (tp + fp + 1e-6)
            macro_pr += doc_precision

            doc_recall = tp / (tp + fn + 1e-6)
            macro_re += doc_recall

        if weak_matching is False:
            assert(self.gm_num == micro_tp + micro_fn)

        micro_pr = 100 * micro_tp / (micro_tp + micro_fp + 1e-6)
        micro_re = 100 * micro_tp / (micro_tp + micro_fn + 1e-6)
        micro_f1 = 2*micro_pr * micro_re / (micro_pr + micro_re + 1e-6)

        macro_pr = 100 * macro_pr / len(self.docids)
        macro_re = 100 * macro_re / len(self.docids)
        macro_f1 = 2*macro_pr*macro_re / (macro_pr + macro_re + 1e-6)

        return micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1

    def tp_fp_fn(self, final_scores, cand_entities_len, cand_entities,
                 begin_span, end_span, spans_len, begin_gm, end_gm,
                 ground_truth, ground_truth_len, words_len, chunk_id,
                 weak_matching, allspans):
        if not allspans:
            begin_gm = begin_span
            end_gm = end_span
      # FOR batch
        for b in range(final_scores.shape[0]):
            filtered_spans, gm_gt_list = _filtered_spans_and_gm_gt_list(b, final_scores,
                                                                        cand_entities_len, cand_entities,
                                                                        begin_span, end_span, spans_len,
                                                                        begin_gm, end_gm,
                                                                        ground_truth, ground_truth_len,
                                                                        words_len)
            self.gm_add(len(gm_gt_list)) # count gms
            matcher = WeakMatcher(gm_gt_list) if weak_matching else StrongMatcher(gm_gt_list)

          # TRUE/FALSE POSITIVES
            for t in filtered_spans:
                if matcher.check(t[1:]):
                    self.check_tp(t[0], chunk_id[b])
                else:
                    self.check_fp(t[0], chunk_id[b])

          # FALSE NEGATIVES
            matcher = FNWeakMatcher(filtered_spans) if weak_matching else FNStrongMatcher(filtered_spans)
            for t in gm_gt_list:
                score = matcher.check(t)
                self.check_fn(score, chunk_id[b])


    def tp_fp_fn_and_prediction_printing(self, final_scores, cand_entities_len, cand_entities,
                                                    begin_span, end_span, spans_len, begin_gm, end_gm,
                                                    ground_truth, ground_truth_len, words_len, chunk_id,
                                                    words, chars, chars_len, scores_l, global_pairwise_scores,
                                                    scores_names_l, weak_matching, allspans, printPredictions=None):
        """
        """
        if not allspans:
            begin_gm = begin_span
            end_gm = end_span
      # FOR batch
        for b in range(final_scores.shape[0]):
            filtered_spans, gm_gt_list = _filtered_spans_and_gm_gt_list(b, final_scores, cand_entities_len,
                                                                        cand_entities, begin_span, end_span,
                                                                        spans_len, begin_gm, end_gm, ground_truth,
                                                                        ground_truth_len, words_len,
                                                                        scores_names_l, scores_l, printing=True)

          # TRUE/FALSE POSITIVES
            self.gm_add(len(gm_gt_list))
            matcher = WeakMatcher(gm_gt_list) if weak_matching else StrongMatcher(gm_gt_list)
            tp_pred = []
            fp_pred = []
            for t in filtered_spans:
            # span: (best_cand_score, begin_idx, end_idx, best_cand_id,
            #        scores_text, best_cand_position, span_num)
                # if t matches gm
                if matcher.check(t[1:4]):
                    # if TP (score >= threshold)
                    if self.check_tp(t[0], chunk_id[b]):
                        tp_pred.append(t)
                    # else FN (see below)
                # else t does not match gm
                else:
                    # if FP (score >= threshold)
                    if self.check_fp(t[0], chunk_id[b]):
                        fp_pred.append(t)
                    # else TN (do nothing)

          # FALSE NEGATIVES
            temp = [t[:4] for t in filtered_spans] # (score, b, e, id)
            matcher = FNWeakMatcher(temp) if weak_matching else FNStrongMatcher(temp)
            fn_pred = []
            gt_minus_fn_pred = [] # should correspond to TP
            for gm_num, t in enumerate(gm_gt_list):
                # best score among spans overlaping with gm and linked to gt
                score = matcher.check(t)
                # if t score < self.threshold
                if self.check_fn(score, chunk_id[b]):
                    fn_pred.append((gm_num, *t))
                else:
                    gt_minus_fn_pred.append((gm_num, *t))
          # PRINT
            if printPredictions is not None:
                gmask = global_pairwise_scores[0][b] if global_pairwise_scores else None
                entity_embeddings = global_pairwise_scores[1][b] if global_pairwise_scores else None
                printPredictions.process_sample(str(chunk_id[b]),
                                                tp_pred, fp_pred, fn_pred, gt_minus_fn_pred,
                                                words[b], words_len[b],
                                                chars[b], chars_len[b],
                                                cand_entities[b], cand_entities_len[b],
                                                final_scores[b], filtered_spans,
                                                [score[b] for score in scores_l], scores_names_l,
                                                gmask, entity_embeddings)

    def print_and_log_results(self, weak_matching, tf_writer=None, eval_cnt=None):
        micro_pr, micro_re, micro_f1, macro_pr, macro_re, macro_f1 = self._score_computation(weak_matching)

        print("micro", "P: %.1f" % micro_pr, "\tR: %.1f" % micro_re, "\tF1: %.1f" % micro_f1)
        print("macro", "P: %.1f" % macro_pr, "\tR: %.1f" % macro_re, "\tF1: %.1f" % macro_f1)

        if tf_writer is None:
            return micro_f1, macro_f1

      # TensorBoard
      # MICRO SUMMARIES
        name = self.name+" micro"
        writer_name = "weak_" if weak_matching else "strong_"

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_f1)])
        tf_writer[writer_name+"f1"].add_summary(summary, eval_cnt)

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_pr)])
        tf_writer[writer_name+"pr"].add_summary(summary, eval_cnt)

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=micro_re)])
        tf_writer[writer_name+"re"].add_summary(summary, eval_cnt)

      # MACRO SUMMARIES
        name = self.name+" macro"
        writer_name = "weak_" if weak_matching else "strong_"

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_f1)])
        tf_writer[writer_name+"f1"].add_summary(summary, eval_cnt)

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_pr)])
        tf_writer[writer_name+"pr"].add_summary(summary, eval_cnt)

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=macro_re)])
        tf_writer[writer_name+"re"].add_summary(summary, eval_cnt)

        return micro_f1, macro_f1


class StrongMatcher(object):
    """
    Initialized with a list of tuples (begin_idx, end_idx, gt)

    Structure used: set (begin_idx, end_idx, gt)
    """
    def __init__(self, b_e_gt_iterator):
        self.data = set()   # tuples (begin_idx, end_idx, gt)
        for t in b_e_gt_iterator:
            self.data.add(t)

    def check(self, t):
        """
        Returns True if the triplet t matches gt else False.
        """
        return True if t in self.data else False


class WeakMatcher(object):
    """
    Initialized with a list of tuples (begin_idx, end_idx, gt)

    Structure used: dict gt --> (begin_idx, end_idx)
    """
    def __init__(self, b_e_gt_iterator):
        self.data = defaultdict(list)
        for b, e, gt in b_e_gt_iterator:
            self.data[gt].append((b, e))

    def check(self, t):
        """
        The predicted triplet t=(b,e,ent_id) is compared with all the gold spans
        with gt==ent_id and checked if they overlap (weak matching) and
        return True or False.
        """

        b, e, ent = t
        if ent in self.data:
            for b2, e2 in self.data[ent]:
                if b<=b2 and e<=e2 and b2<e:
                    return True
                elif b>=b2 and e>=e2 and b<e2:
                    return True
                elif b<=b2 and e>=e2:
                    return True
                elif b>=b2 and e<=e2:
                    return True
        return False


class FNStrongMatcher(object): # FN ~ false negative
    """
    Takes list of predictions (score, begin_idx, end_idx, ent_id).
    Builds a dictionary used to check what score we have given to gt
    i.e. gold mention plus the correct entity.

    structure used: dict (begin_idx, end_idx, ent_id) --> score
    """
    def __init__(self, filtered_spans):
        self.data = dict()
        for score, b, e, ent_id in filtered_spans:
            self.data[(b, e, ent_id)] = score

    def check(self, t):
        """
        t are tuples (begin_idx, end_idx, gt) from gm_gt_list.
        Check if the ground truth is in predictions and return score.
        """
        return self.data[t] if t in self.data else -10000


class FNWeakMatcher(object): # FN ~ false negative
    """
    Takes list of predictions (score, begin_idx, end_idx, ent_id).
    Builds a dictionary used to check what score we have given to
    gold mention plus the correct entity.

    structure used: dict ent_id --> (begin_idx, end_idx, given_score)
    """
    def __init__(self, filtered_spans):
        self.data = defaultdict(list)
        for score, b, e, ent_id in filtered_spans:
            self.data[ent_id].append((b, e, score))

    def check(self, t):
        """
        The gt triplet (b,e,gt) is compared with all the predicted spans
        linked to (gt) and check if they overlap (weak matching) and
        the highest score is returned.
        """
        s, e, gt = t
        best_score = -10000
        if gt in self.data:
            for s2, e2, score in self.data[gt]:
                if s<=s2 and e<=e2 and s2<e:
                    best_score = max(best_score, score)
                elif s>=s2 and e>=e2 and s<e2:
                    best_score = max(best_score, score)
                elif s<=s2 and e>=e2:
                    best_score = max(best_score, score)
                elif s>=s2 and e<=e2:
                    best_score = max(best_score, score)
        return best_score


def _filtered_spans_and_gm_gt_list(b, final_scores,
                                   cand_entities_len, cand_entities,
                                   begin_span, end_span, spans_len,
                                   begin_gm, end_gm, ground_truth,
                                   ground_truth_len, words_len,
                                   scores_names_l=None, scores_l=None, printing=False):
    """
    Returns list of (score, begin_idx, end_idx, ent_id) for cand. spans.
    For each span take a candidate with the best score.
    For overlapping spans only the span with the highest score is taken.

    Also returns list of (begin_idx, end_idx, gt) for gold mentions.
    """
    spans = []
  # FOR candidate span find cand entity with the highest score
    for i in range(spans_len[b]):
        begin_idx = begin_span[b][i]
        end_idx = end_span[b][i]

        best_cand_id = -1
        best_cand_score = -10000
        if printing:
            best_cand_position = -1
            scores_text = "invalid"
      # FOR candidate entities for this span
        for j in range(cand_entities_len[b][i]):
            score = final_scores[b][i][j]
            if score > best_cand_score:
                best_cand_score = score
                best_cand_id = cand_entities[b][i][j]
                if printing:
                    scores_text = ', '.join(reversed([scores_name + "=" + str(score[b][i][j])
                                                  for scores_name, score in zip(scores_names_l, scores_l)]))
                    best_cand_position = j
        if printing:
            span_num = i
            spans.append((best_cand_score, begin_idx, end_idx, best_cand_id,
                          scores_text, best_cand_position, span_num))
        else:
            spans.append((best_cand_score, begin_idx, end_idx, best_cand_id))

  # FILTER this list of spans based on SCORE
  # for the overlapping ones, keep the one with the highest score.
  # span: (best_cand_score, begin_idx, end_idx, best_cand_id,
  #        scores_text, best_cand_position, span_num)
    spans = sorted(spans, reverse=True)
    filtered_spans = []
    claimed = np.full(words_len[b], False, dtype=bool)
    for span in spans:
        best_cand_score, begin_idx, end_idx, best_cand_id = span[:4]
        if not np.any(claimed[begin_idx:end_idx]) and best_cand_id > 0:
            # nothing is claimed so take it
            claimed[begin_idx:end_idx] = True
            filtered_spans.append(span)

  # LIST of (begin_idx, end_idx, gt)
    gm_gt_list = [(begin_gm[b][i], end_gm[b][i], ground_truth[b][i]) for i in range(ground_truth_len[b])]


    return filtered_spans, gm_gt_list


def tp_fp_fn(final_scores, cand_entities_len, cand_entities,
             begin_span, end_span, spans_len, begin_gm, end_gm, ground_truth,
             ground_truth_len, words_len, chunk_id, weak_matching, allspans):
    tp_fp_batch_scores = []
    fn_batch_scores = []
    if not allspans:
        begin_gm = begin_span
        end_gm = end_span
  # FOR batch
    for b in range(final_scores.shape[0]):
        filtered_spans, gm_gt_list = _filtered_spans_and_gm_gt_list(b, final_scores, cand_entities_len, cand_entities,
                                                                    begin_span, end_span, spans_len, begin_gm, end_gm,
                                                                    ground_truth, ground_truth_len, words_len)
      # TRUE/FALSE POSITIVES
        matcher = WeakMatcher(gm_gt_list) if weak_matching else StrongMatcher(gm_gt_list)
        for t in filtered_spans: # (score, begin_idx, end_idx, ent_id)
            if matcher.check(t[1:]): # (begin_idx, end_idx, ent_id)
                tp_fp_batch_scores.append((t[0], 1))   # (score, TP)
            else:
                tp_fp_batch_scores.append((t[0], 0))   # (score, FP)

      # FALSE NEGATIVES
        matcher = FNWeakMatcher(filtered_spans) if weak_matching else FNStrongMatcher(filtered_spans)
        for t in gm_gt_list:
            score = matcher.check(t)
            fn_batch_scores.append(score)

    return tp_fp_batch_scores, fn_batch_scores
