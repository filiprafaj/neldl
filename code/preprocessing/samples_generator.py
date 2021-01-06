from collections import namedtuple

import model.config as config
import numpy as np
import unidecode
from termcolor import colored

import preprocessing.util as util


class Chunker(object):
    def __init__(self, args):
        # I removed the possibility to choose chunk separator
        # -> chunk = document
        self.args = args
        self.chunk_endings = {"DOCEND"}
        self.parsing_errors = 0

    def new_chunk(self):
        self.chunk_words = []
        self.begin_gm = []     # the starting positions of gold mentions
        self.end_gm = []       # the end positions of gold mentions
        self.ground_truth = [] # list with the correct entity ids

    def compute_result(self, docid):
        # correctness checks
        no_errors_flag = True
        if len(self.begin_gm) != len(self.end_gm) or len(self.begin_gm) != len(self.ground_truth):
            no_errors_flag = False
        for b, e in zip(self.begin_gm, self.end_gm):
            if e <= b or b >= len(self.chunk_words) or e > len(self.chunk_words):
                no_errors_flag = False

        if no_errors_flag == False:
            self.parsing_errors += 1
            print("chunker parse error: ", docid)
            return None
        else:
            return (docid, self.chunk_words, self.begin_gm, self.end_gm, self.ground_truth)

    def process(self, filepath):
        with open(filepath, mode="r", encoding="utf-8") as fin:
            self.new_chunk()
            docid = ""
            docids = []
            for line in fin:
                line = line.rstrip() # remove '\n'
                if line in self.chunk_endings:
                    if len(self.chunk_words) > 0:
                        ch = self.compute_result(docid)
                        self.new_chunk()
                        if ch is not None:
                            yield ch
                elif line == "*NL*":
                    pass
                elif line.startswith("MMSTART_"):
                    ent_id = line[len("MMSTART_"):]
                    self.ground_truth.append(ent_id)
                    self.begin_gm.append(len(self.chunk_words))
                elif line == "MMEND":
                    self.end_gm.append(len(self.chunk_words))
                elif line.startswith("DOCSTART_"):
                    docid = line[len("DOCSTART_"):]
                    while docid in docids:
                        docid=docid+'.'
                    if self.args.bert_casing == "cased":
                        docid = unidecode.unidecode(docid)
                    docids.append(docid)
                else:
                    self.chunk_words.append(line)

        print(" chunker parsing errors: ", self.parsing_errors)
        self.parsing_errors = 0


AllspansSample = namedtuple("AllspansSample",
                            ["chunk_id", "chunk_words", "begin_spans", "end_spans",
                             "ground_truth", "cand_entities", "cand_entities_scores",
                             "begin_gm", "end_gm", "chunk_embeddings"])

GoldspansSample = namedtuple("GoldspansSample",
                          ["chunk_id", "chunk_words", "begin_gm", "end_gm", "ground_truth",
                           "cand_entities", "cand_entities_scores", "chunk_embeddings"])


class SamplesGenerator(object):
    def __init__(self, args, mode="allspans"):
        self.args = args
        self.mode = mode
        self._generator = Chunker(args)
        self.fetchCandidateEntities = util.FetchCandidateEntities(self.args, mode)
        self.all_gm_misses = 0
        self.all_gt_misses = 0
        self.all_gm = 0   # all the gm encountered in all the datasets

    def set_goldspans_mode(self):
        self.mode = "goldspans"
        self.fetchCandidateEntities.set_goldspans_mode()

    def set_allspans_mode(self):
        self.mode = "allspans"
        self.fetchCandidateEntities.set_allspans_mode()

    def is_goldspans_mode(self):
        return True if self.mode == "goldspans" else False

    def is_allspans_mode(self):
        return True if self.mode == "allspans" else False

    def process(self, filepath):
        if self.is_allspans_mode():
            return self._process_allspans(filepath)
        else:
            return self._process_goldspans(filepath)

    def _process_goldspans(self, filepath):
      # LOAD FILE EMBEDDINGS
        file_embeddings = np.load(config.base_folder/("data/new_datasets_bert_"+
                                  self.args.bert_casing+'_'+self.args.bert_size)/(filepath.name[:-4]+".npz"),
                                  allow_pickle= True)

        if self.args.calculate_stats:
            total_candidates = 0
            gt_misses = 0
            gm_misses = 0
            gm_this_file = 0
            max_span_width_violations = 0

        for chunk in self._generator.process(filepath):
            chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk
            if self.args.calculate_stats:
                gm_this_file += len(begin_gm)
          # CHUNK EMBEDDDINGS
            chunk_embeddings = np.concatenate(file_embeddings[chunk_id])
            assert_message = ("len(chunk_words):{}, len(chunk_embeddings){}, file/chunkid: "
                              .format(len(chunk_words),len(chunk_embeddings))+str(filepath)+'/'+chunk_id)
            assert len(chunk_words) == len(chunk_embeddings), assert_message

          # CANDIDATE ENTITIES
            self.fetchCandidateEntities.set_goldspans_mode()

            (
             begin_spans,
             end_spans,
             cand_entities,
             cand_entities_scores
            ) = self.fetchCandidateEntities.process(chunk_words, begin_gm=begin_gm, end_gm=end_gm,
                                                    redirections=self.args.redirections)

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

            if self.args.calculate_stats:
                spans_with_candidates = list(zip(begin_spans, end_spans))
                for left, right, cand_ent, gt in zip(begin_gm, end_gm, cand_entities, ground_truth):
                    total_candidates+=len(cand_entities)
                    if (left, right) not in spans_with_candidates:
                        gm_misses += 1
                        #print(" gm not in p_e_m:\t", colored(' '.join(chunk_words[left:right]), "red"))
                    elif gt not in cand_ent:
                        gt_misses += 1
                        #print(" gt not in cand ent:\t", colored(' '.join(chunk_words[left:right]), "green"))
                        #print(" gt: ", gt, "cand_ent: ", cand_ent)

                    if right - left > self.args.max_span_width:
                        max_span_width_violations += 1
                for b, e in zip(begin_gm, end_gm):
                    if e - b > self.args.max_span_width:
                        max_span_width_violations += 1

            if begin_gm:  #not emtpy
                yield GoldspansSample(chunk_id, chunk_words, begin_gm, end_gm, ground_truth,
                                      cand_entities, cand_entities_scores, chunk_embeddings)

        if self.args.calculate_stats:
            print(" total_candidates: ", total_candidates)
            print(" gt_misses: ", gt_misses)
            print(" gm_misses: ", gm_misses)
            print(" gm_this_file: ", gm_this_file)
            print(" recall: ", (1 - (gm_misses+gt_misses)/gm_this_file)*100, "%")
            print(" max_span_width_violations: ", max_span_width_violations)
            self.all_gt_misses += gt_misses
            self.all_gm_misses += gm_misses
            self.all_gm += gm_this_file

    def _process_allspans(self, filepath):
      # LOAD FILE EMBEDDINGS
        file_embeddings = np.load(config.base_folder/("data/new_datasets_bert_"+
                                  self.args.bert_casing+'_'+self.args.bert_size)/(filepath.name[:-4]+".npz"),
                                  allow_pickle=True)
        if self.args.calculate_stats:
            total_candidates = 0
            gt_misses = 0
            gm_misses = 0
            gm_this_file = 0
            max_span_width_violations = 0

        for chunk in self._generator.process(filepath):
            chunk_id, chunk_words, begin_gm, end_gm, ground_truth = chunk
            if self.args.calculate_stats:
                gm_this_file += len(begin_gm)

          # CHUNK EMBEDDDINGS
            chunk_embeddings = file_embeddings[chunk_id]
            separation_indexes = np.cumsum([len(x) for x in chunk_embeddings])
            chunk_embeddings = np.concatenate(chunk_embeddings)

            assert_message = ("len(chunk_words):{}, len(chunk_embeddings){}, file:chunkid: "
                              .format(len(chunk_words),len(chunk_embeddings))+str(filepath)+':'+chunk_id)
            assert len(chunk_words) == len(chunk_embeddings), assert_message

          # CANDIDATE ENTITIES
            self.fetchCandidateEntities.set_allspans_mode()
            (
             begin_spans,
             end_spans,
             cand_entities,
             cand_entities_scores
            ) = self.fetchCandidateEntities.process(chunk_words, separation_indexes=separation_indexes,
                                                    redirections=self.args.redirections)

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
            if self.args.calculate_stats:
                # check if gold mentions are inside the candidate spans
                # if yes check if ground truth is in cand ent.
                gm_spans = list(zip(begin_gm, end_gm))
                all_spans = list(zip(begin_spans, end_spans))
                for i, _ in enumerate(all_spans):
                    total_candidates+=len(cand_entities[i])
                for i, gm_span in enumerate(gm_spans):
                    if gm_span not in all_spans:
                        gm_misses += 1
                        #print(" gm not in spans:\t", colored(' '.join(chunk_words[gm_span[0]:gm_span[1]]), "red"))
                    elif ground_truth[i] not in cand_entities[all_spans.index(gm_span)]:
                        gt_misses += 1
                        #print(" gt not in cand ent:\t", colored(' '.join(chunk_words[gm_span[0]:gm_span[1]]), "yellow"))
                        #print(" gt: ", ground_truth[i], "cand_ent: ", cand_entities[all_spans.index(gm_span)])
                for b, e in zip(begin_gm, end_gm):
                    if e - b > self.args.max_span_width:
                        max_span_width_violations += 1

            if begin_spans: # there are candidate spans in the text
                yield AllspansSample(chunk_id, chunk_words, begin_spans, end_spans,
                                     ground_truth, cand_entities, cand_entities_scores,
                                     begin_gm, end_gm, chunk_embeddings)

        if self.args.calculate_stats:
            print(" total_candidates: ", total_candidates)
            print(" max_span_width_violations: ", max_span_width_violations)
            print(" gt_misses: ", gt_misses)
            print(" gm_misses: ", gm_misses)
            print(" gm_this_file: ", gm_this_file)
            print(" recall: ", (1 - (gm_misses+gt_misses)/gm_this_file)*100, "%")
            self.all_gt_misses += gt_misses
            self.all_gm_misses += gm_misses
            self.all_gm += gm_this_file


class PrintSamples(object):
    def __init__(self, only_misses=True):
        _, self.wiki_id_name_map = util.load_wiki_name_id_map(print_stats=False)
        self.only_misses = only_misses

    def print_candidates(self, ent_ids_list):
        """takes as input a list of ent_id and returns a string. This string has each ent_id
        together with the corresponding name (in the name withspaces are replaced by underscore)
        and candidates are separated with a single space. e.g.  ent_id,Barack_Obama ent_id2,US_President"""
        acc = []
        for ent_id in ent_ids_list:
            acc.append(ent_id + "," + self.wiki_id_name_map[ent_id].replace(' ', '_'))
        return ' '.join(acc)

    def print_sample(self, sample):
        (
         chunk_words,
         begin_gm,
         end_gm,
         ground_truth,
         cand_entities
        ) = (
             sample.chunk_words,
             sample.begin_gm,
             sample.end_gm,
             sample.ground_truth,
             sample.cand_entities
            )
        if isinstance(sample, GoldspansSample):
            misses_idx = []
            for i, (gt, cand_ent) in enumerate(zip(ground_truth, cand_entities)):
                if gt not in cand_ent:
                    misses_idx.append(i)  # miss detected

            if self.only_misses and misses_idx:
                print(colored("New sample", "red"))
                print(' '.join(chunk_words))
                for i in misses_idx:
                    message = (' '.join(chunk_words[begin_gm[i]:end_gm[i]]) +
                               "\tgt=" + self.print_candidates([ground_truth[i]]) +
                               "\tCandidates: " + self.print_candidates(cand_entities[i]))
                    print(colored(message, "yellow"))
            if self.only_misses == False:
                print(colored("New sample", "red"))
                print(' '.join(chunk_words))
                for i in range(len(begin_gm)):
                    message = (' '.join(chunk_words[begin_gm[i]:end_gm[i]]) +
                               "\tgt=" + self.print_candidates([ground_truth[i]]) +
                               "\tCandidates: " + self.print_candidates(cand_entities[i]))
                    print(colored(message, "yellow" if i in misses_idx else "white"))
        elif isinstance(sample, AllspansSample):
            begin_spans, end_spans = sample.begin_spans, sample.end_spans
            gm_spans = list(zip(begin_gm, end_gm))   # [(3, 5), (10, 11), (15, 18)]
            all_spans = list(zip(begin_spans, end_spans))
            print(colored("New sample", "red"))
            print(' '.join(chunk_words))
            for i, gm_span in enumerate(gm_spans):
                if gm_span not in all_spans:
                    message = ' '.join(chunk_words[begin_gm[i]:end_gm[i]]) + "\tgt=" + \
                              self.print_candidates([ground_truth[i]]) + "\tGOLDSPAN MISS"
                    print(colored(message, "magenta"))
                elif ground_truth[i] not in cand_entities[all_spans.index(gm_span)]:
                    message = ' '.join(chunk_words[begin_gm[i]:end_gm[i]]) + "\tgt=" + \
                              self.print_candidates([ground_truth[i]]) + "\tGROUD TRUTH MISS Candidates: " + \
                              self.print_candidates(cand_entities[all_spans.index(gm_span)])
                    print(colored(message, "yellow"))

            if self.only_misses == False:
                # then also print all the spans and their candidate entities
                for left, right, cand_ent in zip(begin_spans, end_spans, cand_entities):
                    # if span is a mention and includes gt then green color, otherwise white
                    if (left, right) in gm_spans and ground_truth[gm_spans.index((left, right))] in cand_ent:
                        message = ' '.join(chunk_words[left:right]) + "\tgt=" + \
                                  self.print_candidates([ground_truth[gm_spans.index((left, right))]]) + \
                                  "\tGOLDSPAN HIT Candidates: " + \
                                  self.print_candidates(cand_ent)
                        print(colored(message, "green"))
                    else:
                        message = ' '.join(chunk_words[left:right]) + \
                                  "\tNOT IN GOLDSPANS Candidates: " + \
                                  self.print_candidates(cand_ent)
                        print(colored(message, "white"))
