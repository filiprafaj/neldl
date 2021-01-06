import operator
import pickle
from collections import defaultdict

import model.config as config
import numpy as np
from preprocessing.util import (load_wiki_name_id_map, load_wikiid2nnid,
                                reverse_dict)
from termcolor import colored


class GMBucketingResults(object):
    def __init__(self, gm_bucketing_pem_pos):
      # BUCKETS
        gm_bucketing_pem_pos.append(200)  # [0,1,2,7,200]
        self.gm_buckets = gm_bucketing_pem_pos
      # COUNT gms in each frequency bucket
        self.gm_cnt = defaultdict(int)
      # COUNT false negatives in bucket
        self.fn_cnt = defaultdict(int)
      # EXCLUDE the false in which winner was identical
      # to gt even if we decided not to annotate in the end
        self.fn_nowinnermatch_cnt = defaultdict(int)
      # gm for which we have only one candidate entity which is the gt
        self.gm_to_gt_unique_mapping = 0

    def reinitialize(self):
        self.gm_cnt = defaultdict(int)
        self.fn_cnt = defaultdict(int)
        self.fn_nowinnermatch_cnt = defaultdict(int)
        self.gm_to_gt_unique_mapping = 0

    def process_fn(self, pos, match_with_winner, num_of_cand_entities):
        if pos == 0 and num_of_cand_entities == 1:
            self.gm_to_gt_unique_mapping += 1
        for t in self.gm_buckets:
            if pos <= t:
                self.gm_cnt[t] += 1
                self.fn_cnt[t] += 1
                if not match_with_winner:
                    self.fn_nowinnermatch_cnt[t] += 1
                break

    def process_tp(self, pos, num_of_cand_entities):
        if pos == 0 and num_of_cand_entities == 1:
            self.gm_to_gt_unique_mapping += 1
        for t in self.gm_buckets:
            if pos <= t:
                self.gm_cnt[t] += 1
                break

    def print(self):
        print("gm_to_gt_unique_mapping =", self.gm_to_gt_unique_mapping)
        for t in self.gm_buckets:
        # GMs that our model labeled correctly (omited - printed above)
            print(str(t), "]", "gm_cnt=", str(self.gm_cnt[t]),
                  "solved=%.1f" % (100*(self.gm_cnt[t] - self.fn_cnt[t])/self.gm_cnt[t]),
                   "winner_match=%.1f" % (100*(self.gm_cnt[t] - self.fn_nowinnermatch_cnt[t])/self.gm_cnt[t]))


class PrintPredictions(object):
    def __init__(self, predictions_folder, experiment_name, gm_bucketing_pem_pos=None,
                 print_global_voters=False):
        self.thr = None
        self.predictions_folder = predictions_folder
        with open(config.base_folder/"data/experiments"/experiment_name/"word_char_maps.pickle", 'rb') as handle:
            _, self.id2word, _, self.id2char, _, _ = pickle.load(handle)

        self.nnid2wikiid = reverse_dict(load_wikiid2nnid(experiment_name), unique_values=True)
        _, self.wiki_id_name_map = load_wiki_name_id_map(print_stats=False)
        self.extra_info = ""
        self.gm_bucketing = GMBucketingResults(gm_bucketing_pem_pos) if gm_bucketing_pem_pos else None
        self.print_global_voters = print_global_voters

    def map_entity(self, nnid, onlyname=False):
        wikiid = self.nnid2wikiid[nnid]
        wikiname = self.wiki_id_name_map[wikiid].replace(' ', '_') if wikiid != "<u>" else "<u>"
        return wikiname if onlyname else "{} {}".format(wikiid, wikiname)

    def file_start(self, weak_matching, name, opt_thr):
        self.thr = opt_thr
        self.weak_matching = weak_matching
        filepath = self.predictions_folder/name
        self.fout = open(filepath, 'w')
        self.fout.write("File:"+str(filepath)+'\n'+self.extra_info+'\n')
        if self.gm_bucketing:
            self.gm_bucketing.reinitialize()

    def file_ended(self):
        self.fout.close()
        if self.gm_bucketing:
            self.gm_bucketing.print()

    def scores_text(self, scores_l, scores_names_l, i, j):
        return ', '.join(reversed([scores_name + "=" + str(score[i][j])
                                   for scores_name, score in zip(scores_names_l, scores_l)]))

    def process_sample(self, chunkid,
                       tp_pred, fp_pred, fn_pred, gt_minus_fn_pred,
                       words, words_len, chars, chars_len,
                       cand_entities, cand_entities_len, final_scores,
                       filtered_spans, scores_l, scores_names_l, gmask,
                       entity_embeddings):
        """
        words: [None] 1d the words of a sample
        words_len: scalar
        chars: [None, None] 2d  words, chars of each word
        chars_len: [None] length for each word the (num of characters)
        cand_entities: [None, None]  candidates for each gm
        cand_entitites_len: [None]  how many cand ent each gm has

        filtered_spans: list of spans sorted by score, span:
                     (score, begin_idx, end_idx, best_nnid,
                      simil_score, best_position, span_num)
        tp_pred, fp_pred: TP, FP lists like filtered_spans
        fn_pred: FN list of missed (gm_num, begin_gm, end_gm, gt)
        gt_minus_fn_pred: should correspond to TP
        """
      # WORDS FROM WORD IDs
        reconstructed_words = []
        for i in range(words_len):
            word = words[i]
            if word != 0:
                reconstructed_words.append(self.id2word[word])
            else:  # <wunk>
                word_chars = []
                for j in range(chars_len[i]):
                    word_chars.append(self.id2char[chars[i][j]])
                reconstructed_words.append(''.join(word_chars))

      # SORTED GMs (gm_num, begin_gm, end_gm, gt)
        span_num_b_e_gt = sorted(fn_pred+gt_minus_fn_pred)

      # TAGS FOR THE RECONSTRUCTED TEXT
        text_tags = defaultdict(list)


      # PRINT GOLD SPANS
        gt_legend = []
        if len(fn_pred) > 0:
            fnWeakMatcherLogging = FNWeakMatcherLogging(self, filtered_spans, cand_entities,
                                        cand_entities_len, final_scores, scores_l, scores_names_l,
                                        reconstructed_words, self.gm_bucketing, gmask, entity_embeddings,
                                        span_num_b_e_gt)
        # GREEN - GMs that our model labeled correctly (omited - printed above)
        # RED - GMs that our model labeled wrongly or missed
        for mylist, mycolor in zip([gt_minus_fn_pred, fn_pred], ["green", "red"]):
            for i, (gm_num, b, e, gt) in enumerate(mylist, 1):
                text_tags[b].append((0, mycolor+"_begin_{}".format(i)))
                text_tags[e].append((0, mycolor+"_end"))

                gt_text = ""
                if self.weak_matching is False:
                    gt_text = "RECALL MISS: gt not in candidate entities"
                    for j in range(cand_entities_len[gm_num]):
                        if cand_entities[gm_num][j] == gt:
                            gt_text = "gt_p_e_m_pos={}".format(j)
                            break
                text = "{}: {}".format(i, self.map_entity(gt))#, gt_text)
                if mycolor == "red":
                    text += fnWeakMatcherLogging.check(gm_num, b, e, gt)
                text = colored(text, mycolor)
                gt_legend.append(text)

      # PRINT TPs
        tp_legend = []
        tp_pred = sorted(tp_pred, key=operator.itemgetter(1))
        # BLUE - correct labels given by our model
        for i, (score, b, e, nnid, scores_text, p_e_m_pos, span_num) in enumerate(tp_pred, 1):
            text_tags[b].append((2, colored("[{}".format(i), "blue")))
            text_tags[e].append((1, colored("{}]".format(i), "blue")))

            text = "{}: {}\n    [pem_pos={}, {}]".format(i, self.map_entity(nnid),
                                                         p_e_m_pos, scores_text)
            tp_legend.append(colored(text,'blue'))
            if self.gm_bucketing:
                self.gm_bucketing.process_tp(p_e_m_pos, cand_entities_len[span_num])

      # PRINT FPs
        fp_legend = []
        fp_pairwise_scores_legend = []
        fp_pred = sorted(fp_pred, key=operator.itemgetter(1))
        if len(fp_pred) > 0:
            fpWeakMatcherLogging = FPWeakMatcherLogging(self, span_num_b_e_gt, cand_entities,
                                                        cand_entities_len, final_scores, scores_l,
                                                        scores_names_l, reconstructed_words,
                                                        self.gm_bucketing, gmask, entity_embeddings)
        # MAGENTA - wrong labels given by our model
        for i, (score, b, e, nnid, scores_text, p_e_m_pos, span_num) in enumerate(fp_pred, 1):
            text_tags[b].append((2, colored("[{}".format(i), "magenta")))
            text_tags[e].append((1, colored("{}]".format(i), "magenta")))

            _, pairwise_score_text = fpWeakMatcherLogging.check(b, e, span_num, p_e_m_pos)
            text = "{}: {} \n    [pem_pos={}, {}]".format(i, self.map_entity(nnid),
                                                          p_e_m_pos, scores_text)
            fp_legend.append(colored(text, "magenta"))
            fp_pairwise_scores_legend.append("\n"+text)
            fp_pairwise_scores_legend.append(pairwise_score_text)

        final_acc = ["\nSample: " + chunkid+"\n"]
      # RECONSTRUCT TEXT WITH TAGS
        green_text = False
        red_text = False
        for i in range(words_len+1):
            tags = [text for _, text in sorted(text_tags[i])]
            if tags and tags[0].startswith("green"):
                txt = tags.pop(0)
                green_text = False if txt=="green_end" else True
                if green_text:
                    tags.append(colored(txt.split('_')[-1], "green"))
                    if tags[0] == "green_end":
                        tags.pop(0)
            if tags and tags[0].startswith("red"):
                txt = tags.pop(0)
                red_text = False if txt=="red_end" else True
                if red_text:
                    tags.append(colored(txt.split('_')[-1], "red"))
                    if tags[0] == "red_end":
                        tags.pop(0)
            final_acc.extend(tags)
            if i < words_len:
                if green_text:
                    final_acc.append(colored(reconstructed_words[i], "green"))
                elif red_text:
                    final_acc.append(colored(reconstructed_words[i], "red"))
                else:
                    final_acc.append(reconstructed_words[i])
      # PRINT TAGGED TEXT
        self.fout.write(" ".join(final_acc)+"\n")
      # PRINT GLOBAL VOTERS
        if self.print_global_voters:
            self.fout.write("global score voters and weights:\n")
            gmask_print_string = self.print_gmask(gmask, span_num_b_e_gt, reconstructed_words, cand_entities)
            self.fout.write(gmask_print_string+"\n")

      # PRINT LEGENDS FOR TAGS (green, red, blue, magenta)
        self.fout.write("\n".join(gt_legend + tp_legend + fp_legend))

        self.fout.write("\n")

    def print_gmask(self, gmask, span_num_b_e_gt, reconstructed_words, cand_entities):
        """

        """
        i = 0
        document_gmask_acc = []
        for span_num, b, e, gt in span_num_b_e_gt:
            assert(i == span_num)
            text_acc = ["mention {} {}: ".format(span_num, ' '.join(reconstructed_words[b:e]))]
            for cand_ent_pos in range(gmask.shape[1]):
                mask_value = gmask[span_num][cand_ent_pos]
                assert(mask_value >= 0)
                if mask_value > 0:
                    text_acc.append("{} {:.2f} | ".format(self.map_entity(cand_entities[span_num][cand_ent_pos]),
                                                          mask_value))
            i += 1
            document_gmask_acc.append(' '.join(text_acc))
        return '\n'.join(document_gmask_acc)


class FNWeakMatcherLogging(object):
    """
    This is used to print info about the FN from the filtered spans
    i.e. the spans we keep that do not overlap.
    filtered_spans: list of tuples (best_cand_score, begin_idx, end_idx,
                                    best_cand_id, scores_text,
                                    best_cand_position, span_num)
    """
    def __init__(self, printPredictions, filtered_spans, cand_entities, cand_entities_len,
                 final_scores, scores_l, scores_names_l, reconstructed_words, gm_bucketing=None,
                 gmask=None, entity_embeddings=None, span_num_b_e_gt=None):
        self.printPredictions = printPredictions
        self.data = filtered_spans
        self.cand_entities = cand_entities
        self.cand_entities_len = cand_entities_len
        self.scores_l = scores_l
        self.scores_names_l = scores_names_l
        self.final_scores = final_scores
        self.reconstructed_words = reconstructed_words
        self.gm_bucketing = gm_bucketing
        self.gmask = gmask
        self.entity_embeddings = entity_embeddings
        self.span_num_b_e_gt = span_num_b_e_gt

    def check(self, gm_num, s, e, gt): # FN tuple
        # compare FN tuple to each span of filtered_spans
        # if they overlap, print winner entity of this span
        # and score assigned to the gt (if it was a candidate)
        acc = []
        for (best_cand_score, s2, e2, best_cand_id, scores_text, best_cand_position, span_num) in self.data:
            overlap = False
            if s<=s2 and e<=e2 and s2<e:
                overlap = True
            elif s>=s2 and e>=e2 and s<e2:
                overlap = True
            elif s<=s2 and e>=e2:
                overlap = True
            elif s>=s2 and e<=e2:
                overlap = True

            if not overlap:
                continue

        # INFO about winner of this span ang gt
          # COMPARE candidate entities of this filtered span
          # to gt
            gt_cand_position = -1
            for j in range(self.cand_entities_len[span_num]):
                if self.cand_entities[span_num][j] == gt:
                    gt_cand_position = j
                    break
          # WINNER
            assert(abs(best_cand_score - self.final_scores[span_num][best_cand_position]) < 0.001)
#            acc.append("span: {}".format(' '.join(self.reconstructed_words[s2:e2])))
            acc.append("\n    [winner: {}, pem_pos={}, {}|".format(
                self.printPredictions.map_entity(best_cand_id),
                #self.final_scores[span_num][best_cand_position],
                best_cand_position,
                self.printPredictions.scores_text(self.scores_l, self.scores_names_l, span_num, best_cand_position)))
          # GT FOUND IN CANDIDATES
            if gt_cand_position >= 0:
                acc.append("\n     gt: {}, pem_pos={}, {} ]".format(
                    self.printPredictions.map_entity(gt),
                    #self.final_scores[span_num][gt_cand_position],
                    gt_cand_position,
                    self.printPredictions.scores_text(self.scores_l, self.scores_names_l, span_num, gt_cand_position)))
                if self.gm_bucketing:
                    self.gm_bucketing.process_fn(gt_cand_position, best_cand_id == gt,
                                                 self.cand_entities_len[span_num])
          # GT NOT FOUND IN CANDIDATES
            else:
                acc.append("\n     RECALL MISS: {}]".format(self.printPredictions.map_entity(gt)))

        if acc == []:
            acc.append(" no overlap with any filtered span")

        return ' '.join(acc)


class FPWeakMatcherLogging(object):
    """
    Initialized with a list of tuples (span_num, begin_idx, end_idx, gt)
    We already know that the tuple doesn't match a ground truth.
    Now we want to find out what exactly happens.
    cases:
      1)) doesn't overlap with any gm
      2)) overlap with one or more gm. In this case for each gm
          that it overlaps we find:
          a) the gt for this gm
          b) final_score, sim_score, p_e_m position of the gt in fp

    structure used: list of (begin_idx, end_idx, gt) tuples
    """
    def __init__(self, printPredictions, span_num_b_e_gt, cand_entities, cand_entities_len,
                 final_scores, scores_l, scores_names_l, reconstructed_words, gm_bucketing=None,
                 gmask=None, entity_embeddings=None):
        self.printPredictions = printPredictions
        self.data = span_num_b_e_gt
        self.cand_entities = cand_entities
        self.cand_entities_len = cand_entities_len
        self.final_scores = final_scores
        self.scores_l = scores_l
        self.scores_names_l = scores_names_l
        self.reconstructed_words = reconstructed_words
        self.gm_bucketing = gm_bucketing
        self.gmask = gmask
        self.entity_embeddings = entity_embeddings

    def check(self, s, e, span_num, winner_pos=None): # FP tuple

      # FOR gms that overlap with the FP tuple
      # compare the tuple candidates with the gt of this gm
        acc = []
        pairwise_scores_text = ""
        for (gm_num, s2, e2, gt) in self.data:
            overlap = False
            if s<=s2 and e<=e2 and s2<e:
                overlap = True
            elif s>=s2 and e>=e2 and s<e2:
                overlap = True
            elif s<=s2 and e>=e2:
                overlap = True
            elif s>=s2 and e<=e2:
                overlap = True

            if not overlap:
                continue

          # COMPARE candidate entities of FP mention to gt
            gt_cand_position = -1
            for j in range(self.cand_entities_len[span_num]):
                if self.cand_entities[span_num][j] == gt:
                    gt_cand_position = j
                    break
          # GT FOUND in candidates
            if gt_cand_position >= 0:
                acc.append("| {}, score={}, pem_pos={}, {}".format(self.printPredictions.map_entity(gt),
                                                                   self.final_scores[span_num][gt_cand_position],
                                                                   gt_cand_position,
                                                                   self.printPredictions.scores_text(self.scores_l,
                                                                   self.scores_names_l, span_num, gt_cand_position)))
          # GT NOT FOUND in candidates
            else:
                acc.append("| RECALL MISS: {}".format(self.printPredictions.map_entity(gt)))


        if acc == []:
            acc.append("| NO OVERLAP WITH GM")

        return ' '.join(acc), pairwise_scores_text

