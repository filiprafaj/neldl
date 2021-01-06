import operator
import os
import pickle
import sys
import time
from collections import defaultdict

import gensim
import model.config as config
import numpy as np
from nltk.corpus import stopwords


class VocabularyCounter(object):
    """
       Counts the freqs of each word and each character in the corpus.
       One frequency vocab for all files that it processes.
    """
    def __init__(self, lowercase=False):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(
          config.base_folder/"data/base_data/Word2Vec/GoogleNews-vectors-negative300.bin", binary=True)
        self.word_freq = defaultdict(int)
        self.char_freq = defaultdict(int)
        self.lowercase = lowercase
        self.not_in_word2vec_cnt = 0
        self.all_words_cnt = 0

    def add(self, filepath):
        """
           Add words and chars from file to vocabularies.
           The file must be in the "new_datasets" format.
        """
        with open(filepath, mode='r', encoding="utf-8") as fin:
            for line in fin:
#                line = unidecode.unidecode(line)
                if (line.startswith("DOCSTART_") or line.startswith("DOCEND") or
                        line.startswith("MMSTART_") or line.startswith("MMEND") or
                        line.startswith("*NL*")):
                    continue
                line = line.rstrip() # remove '\n'
                word = line.lower() if self.lowercase else line
                self.all_words_cnt += 1
                if word not in self.model:
                    self.not_in_word2vec_cnt += 1
                else:
                    self.word_freq[word] += 1
                for c in line:
                    self.char_freq[c] += 1

    def print_statistics(self, word_edges=None, char_edges=None):
        """
           Print some statistics about word and char frequency.
        """
        if word_edges is None:
            word_edges = [1, 2, 3, 6, 11, 21, 31, 51, 76, 101, 201, np.inf]
        if char_edges is None:
            char_edges = [1, 6, 11, 21, 51, 101, 201, 501, 1001, 2001, np.inf]
        print("\nSTATISTICS from VocabularyCounter:")
        print("not_in_word2vec_cnt = ", self.not_in_word2vec_cnt)
        print("all_words_cnt = ", self.all_words_cnt)
        print("histogram bins are [...)")
        for d, name, edges in zip([self.word_freq, self.char_freq], ["word", "character"], [word_edges, char_edges]):
            hist_values, _ = np.histogram(list(d.values()), edges)
            cum_sum = np.cumsum(hist_values[::-1])
            print(name, " frequency histogram, edges: ", edges)
            print("absolute values:                ", hist_values)
            print("absolute cumulative (right to left):    ", cum_sum[::-1])
            print("probabilites cumulative (right to left):", (cum_sum / np.sum(hist_values))[::-1])

    def serialize(self, experiment_name, name="word_char_freq.pickle"):
        folder = config.base_folder/"data/experiments"/experiment_name
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        with open(folder/name, "wb") as f:
            pickle.dump((self.word_freq, self.char_freq), f)

    def count_datasets_vocabulary(self, datapath, experiment_name):
        """
           count all files in datapath ("data/base_data/new_datasets")

           ["aida_train.txt", "aida_dev.txt", "aida_test.txt",
            "ace2004.txt", "aquaint.txt", "clueweb.txt",
            "msnbc.txt", "wikipedia.txt"]
        """
      # COUNT FILES
        new_dataset_folder = datapath
        for dataset in new_dataset_folder.glob("**/*.txt"):
            print("VocabularyCounter processing dataset: ", dataset.name)
            self.add(dataset)

      # PRINT HISTOGRAMS
        self.print_statistics()

      # SAVE TO PICKLE FILE
        self.serialize(experiment_name=experiment_name)


def create_id_maps_and_word_embeddigns(args):
  # VOCABULARY (word and char)
    with open(config.base_folder/"data/experiments"/args.experiment_name/"word_char_freq.pickle", "rb") as f:
        word_freq, char_freq = pickle.load(f)

  # INITIALIZE WORD MAP
    word2id = dict()
    id2word = dict()
    wcnt = 0   # unknown word
    word2id["<wunk>"] = wcnt
    id2word[wcnt] = "<wunk>"
    wcnt += 1

  # INITIALIZE CHAR MAP
    char2id = dict()
    id2char = dict()
    ccnt = 0   # unknown character
    char2id["<u>"] = ccnt
    id2char[ccnt] = "<u>"
    ccnt += 1

  # FOR EVERY WORD
  # (already filtered out the words that are not in word2vec (line 43))
    for word in word_freq:
        if word_freq[word] >= args.word_freq_thr:
            word2id[word] = wcnt
            id2word[wcnt] = word
            wcnt += 1

  # FOR EVERY CHAR
    for c in char_freq:
        if char_freq[c] >= args.char_freq_thr:
            char2id[c] = ccnt
            id2char[ccnt] = c
            ccnt += 1

    assert(len(word2id) == wcnt)
    assert(len(char2id) == ccnt)
    print("\nSTATISTICS from create_id_maps_and_word_embeddigns:")
    print("words in vocabulary: ", wcnt)
    print("characters in vocabulary: ", ccnt)

  # SAVE TO FILE
    with open(config.base_folder/"data/experiments"/args.experiment_name/"word_char_maps.pickle", 'wb') as f:
        pickle.dump((word2id, id2word, char2id, id2char, args.word_freq_thr, args.char_freq_thr), f)

  # SAVE WORD EMBEDDINGS
    model = gensim.models.KeyedVectors.load_word2vec_format(
      config.base_folder/"data/base_data/Word2Vec/GoogleNews-vectors-negative300.bin", binary=True)
    embedding_dim = len(model['queen'])
    embeddings_array = np.empty((wcnt, embedding_dim))
    embeddings_array[0] = np.zeros(embedding_dim)
    for i in range(1, wcnt):
        embeddings_array[i] = model[id2word[i]]
    np.save(config.base_folder/"data/experiments"/args.experiment_name/"embeddings/word_embeddings.npy", embeddings_array)

    return word2id, char2id


def restore_id_maps(experiment_name):
    with open(config.base_folder/"data/experiments"/experiment_name/"word_char_maps.pickle", "rb") as f:
        word2id, _, char2id, _, _, _ = pickle.load(f)
    return word2id, char2id


def load_wikiid2nnid(experiment_name):
    """
        Returns a map from wikiid to nnid (for entity embeddings)
    """
    wikiid2nnid = dict()
    with open(config.base_folder/"data/experiments"/experiment_name/"wikiid2nnid.txt") as fin:
        for line in fin:
            ent_id, nnid = line.split('\t')
            wikiid2nnid[ent_id] = int(nnid) - 1  # torch starts from 1 instead of zero
        assert(wikiid2nnid["1"] == 0)
        assert(-1 not in wikiid2nnid)
        wikiid2nnid["<u>"] = 0
        del wikiid2nnid["1"]
        #print(len(wikiid2nnid))
    return wikiid2nnid


def load_redirections(lowercase, print_stats=True):
    wall_start = time.time()
    redirections = dict()
    with open(config.base_folder/"data/base_data/wiki_redirects.txt") as fin:
        redirections_errors = 0
        for line in fin:
            line = line.rstrip()
            try:
                old_title, new_title = line.split('\t')
                if lowercase:
                    old_title, new_title = old_title.lower(), new_title.lower()
                redirections[old_title] = new_title
            except ValueError:
                redirections_errors += 1
    if print_stats:
        print("\nLOAD REDIRECTIONS:\nwall time:", (time.time() - wall_start)/60, " minutes")
        print("redirections_errors: ", redirections_errors)
    return redirections


def load_disambiguations(print_stats=True):
    wall_start = time.time()
    disambiguations_ids = set()
    #disambiguations_titles = set()
    disambiguations_errors = 0
    with open(config.base_folder/"data/base_data/wiki_disambiguation_pages.txt") as fin:
        for line in fin:
            line = line.rstrip()
            try:
                article_id, title = line.split("\t")
                disambiguations_ids.add(article_id)
                #disambiguations_titles.add(title)
            except ValueError:
                disambiguations_errors += 1
    if print_stats:
        print("\nLOAD DISAMBIGUATIONS:\nwall time:", (time.time() - wall_start)/60, " minutes")
        print("disambiguations_errors: ", disambiguations_errors)
    return disambiguations_ids


def load_wiki_name_id_map(lowercase=False, filepath=None, print_stats=True):
    wall_start = time.time()
    wiki_name_id_map = dict()
    wiki_id_name_map = dict()
    wiki_name_id_map_errors = 0
    duplicate_names = 0 # different lines in the doc with the same title
    duplicate_ids = 0 # with the same id
    if filepath is None:
        filepath = config.base_folder/"data/base_data/wiki_name_id_map.txt"
    disambiguations_ids = load_disambiguations(print_stats)
    with open(filepath) as fin:
        for line in fin:
            line = line.rstrip()
            try:
                wiki_title, wiki_id = line.split('\t')
                if wiki_id in disambiguations_ids:
                    continue
                if lowercase:
                    wiki_title = wiki_title.lower()
                if wiki_title in wiki_name_id_map:
                    duplicate_names += 1
                if wiki_id in wiki_id_name_map:
                    duplicate_ids += 1

                wiki_name_id_map[wiki_title] = wiki_id
                wiki_id_name_map[wiki_id] = wiki_title
            except ValueError:
                wiki_name_id_map_errors += 1
    if print_stats:
        print("\nLOAD wiki_name_id_map:\nwall time:", (time.time() - wall_start)/60, " minutes")
        print("len(wiki_name_id_map): ", len(wiki_name_id_map))
        print("wiki_name_id_map_errors: ", wiki_name_id_map_errors)
        print("duplicate names: ", duplicate_names)
        print("duplicate ids: ", duplicate_ids)
    return wiki_name_id_map, wiki_id_name_map


def load_persons():
    wiki_name_id_map, _ = load_wiki_name_id_map()
    person_wikiids = set()
    not_found_cnt = 0
    with open(config.base_folder/"data/base_data/persons.txt") as fin:
        for line in fin:
            line = line.strip()
            if line in wiki_name_id_map:
                person_wikiids.add(wiki_name_id_map[line])
            else:
                not_found_cnt += 1
    print("\nLOAD PERSONS:\npersons not_found_cnt:", not_found_cnt)
    return person_wikiids


def custom_p_e_m(max_cand_ent, lowercase_p_e_m=False, allowed_entities_set=None):
    """
    max_cand_ent: how many candidate entities to keep for each mention
    allowed_entities_set: restrict the candidate entities to this set
    """
    wall_start = time.time()
    _, wiki_id_name_map = load_wiki_name_id_map(lowercase=False, print_stats=False)
    p_e_m = dict() # for each mention a list of tuples (ent_id, score)
    mention_total_freq = dict() # total freqs for mentions in p_e_m
    p_e_m_errors = 0
    incompatible_ent_ids = 0
    with open(config.base_folder/"data/base_data/prob_yago_crosswikis_wikipedia_p_e_m.txt") as fin:
      # READ p_e_m
        print("\nREADING p_e_m:")
        duplicate_mentions_cnt = 0
        clear_conflict_winners = 0 # higher freq, longer cand list
        not_clear_conflict_winners = 0  # higher freq, shorter cand list

        for line in fin:
        # for each mention we create a list of tuples (ent_id, score)
            line = line.rstrip()
            try:
                temp = line.split('\t')
                mention, entities = temp[0],  temp[2:]
                absolute_freq = int(temp[1])

              # COLLECT CANDIDATES
                res = []
                for e in entities:
                    if len(res) >= max_cand_ent:
                        break
                    ent_id, score, _ = map(str.strip, e.split(',', 2))
                    if not ent_id in wiki_id_name_map:
                        incompatible_ent_ids += 1
                    elif (allowed_entities_set is not None
                          and ent_id not in allowed_entities_set):
                        pass
                    else:
                        res.append((ent_id, float(score)))

              # RESOLVE DUPLICATE MENTIONS
                if res:
                    if mention in p_e_m:
                        duplicate_mentions_cnt += 1
                        if absolute_freq > mention_total_freq[mention]:
                            if len(res) > len(p_e_m[mention]):
                                clear_conflict_winners += 1
                            else:
                                not_clear_conflict_winners += 1
                            p_e_m[mention] = res
                            mention_total_freq[mention] = absolute_freq
                    else:
                        p_e_m[mention] = res
                        mention_total_freq[mention] = absolute_freq

            except Exception as esd:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                p_e_m_errors += 1
                print("error in line: ", repr(line))

    print("wall time:", (time.time() - wall_start)/60, " minutes")
    print("duplicate_mentions_cnt: ", duplicate_mentions_cnt)
    print("p_e_m_errors: ", p_e_m_errors)
    print("incompatible_ent_ids: ", incompatible_ent_ids)

    if not lowercase_p_e_m:
        return p_e_m, None, mention_total_freq

  # LOWERCASE p_e_m
    # two different p(e|m) mentions can be the same after lower()
    # so we merge the two candidate entities lists.
    # If different score for same candidate - keep the highest score.
    # As before we keep only the top max_cand_ent entities

    print("\nREADING p_e_m lowercase:")
    wall_start = time.time()
    p_e_m_lowercased = defaultdict(lambda: defaultdict(int))

    for mention, res in p_e_m.items():
        low_mention = mention.lower()
        # if low_mention in p_e_m - do nothing, already in dictionary
        # e.g. p(e|m) has Obama and obama. So when I lowercase Obama
        # I find that obama already exist - prefer the existing one
        if low_mention not in p_e_m:
            for r in res:
                ent_id, score = r
                p_e_m_lowercased[low_mention][ent_id] = max(score, p_e_m_lowercased[low_mention][ent_id])

    print("wall time:", (time.time() - wall_start)/60, " minutes")

  # KEEP TOP max_cand_ent
    p_e_m_lowercased_trim = dict()
    for mention, ent_score_map in p_e_m_lowercased.items():
        sorted_ = sorted(ent_score_map.items(), key=operator.itemgetter(1), reverse=True)
        p_e_m_lowercased_trim[mention] = sorted_[:max_cand_ent]

    return p_e_m, p_e_m_lowercased_trim, mention_total_freq


class FetchCandidateEntities(object):
    def __init__(self, args, mode="allspans"):
        self.args = args
        self.mode = mode
        if args.person_coreference:
            self.person_wikiids = load_persons()
        self.redirections = load_redirections(lowercase=args.lowercase_redirections)
        self.stopwords = set(stopwords.words("english"))
        (
         self.p_e_m,
         self.p_e_m_low,
         self.mention_total_freq
        ) = custom_p_e_m(args.max_cand_ent, args.lowercase_p_e_m)

    def set_goldspans_mode(self):
        self.mode = "goldspans"
    def set_allspans_mode(self):
        self.mode = "allspans"

    def process(self, chunk_words, begin_gm=None, end_gm=None, separation_indexes=None, redirections=True):

        begin_spans = []
        end_spans = []
        cand_entities = []   # list of lists
        cand_entities_scores = []

        spans = (self.all_spans(separation_indexes, self.args.max_span_width)
                 if self.mode=="allspans" else zip(begin_gm, end_gm))

        for span in spans:
          # SPAN
            span_text = ' '.join(chunk_words[span[0]:span[1]])
            title = span_text.title()

            # no stopwords
            if self.mode=="allspans" and span_text.lower() in self.stopwords: continue

            title_freq = self.mention_total_freq[title] if title in self.mention_total_freq else 0
            span_freq = self.mention_total_freq[span_text] if span_text in self.mention_total_freq else 0

            if title_freq == 0 and span_freq == 0:
                if self.args.lowercase_spans and span_text.lower() in self.p_e_m:
                    cand_ent, cand_ent_scores = map(list, zip(*self.p_e_m[span_text.lower()]))
                elif self.args.lowercase_p_e_m and span_text.lower() in self.p_e_m_low:
                    cand_ent, cand_ent_scores = map(list, zip(*self.p_e_m_low[span_text.lower()]))
              # REDIRECTIONS
                elif (redirections and span_text in self.redirections and self.redirections[span_text] in self.p_e_m):
                    redir = self.redirections[span_text]
                    cand_ent, cand_ent_scores = map(list, zip(*self.p_e_m[redir]))
                elif (redirections and title in self.redirections and self.redirections[title] in self.p_e_m):
                    redir = self.redirections[title]
                    cand_ent, cand_ent_scores = map(list, zip(*self.p_e_m[redir]))
                elif (redirections and self.args.lowercase_spans
                      and span_text.lower() in self.redirections
                      and self.redirections[span_text.lower()] in self.p_e_m):
                    redir = self.redirections[span_text.lower()]
                    cand_ent, cand_ent_scores = map(list, zip(*self.p_e_m[redir]))
                elif (redirections and self.args.lowercase_p_e_m
                      and span_text.lower() in self.redirections
                      and self.redirections[span_text.lower()] in self.p_e_m_low):
                    redir = self.redirections[span_text.lower()]
                    cand_ent, cand_ent_scores = map(list, zip(*self.p_e_m_low[redir]))
                else:
                    cand_ent, cand_ent_scores =  None, None
            else:
                if span_freq > title_freq:
                    cand_ent, cand_ent_scores =  map(list, zip(*self.p_e_m[span_text]))
                else:
                    cand_ent, cand_ent_scores =  map(list, zip(*self.p_e_m[title]))
                    # zip:
                    # from [("ent1", 0.4), ("ent2", 0.3), ("ent3", 0.3)] to
                    # ("ent1", "ent2", "ent3")  and (0.4, 0.3, 0.3)
                    # map:
                    # ["ent1", "ent2", "ent3"]   , [0.4, 0.3, 0.3]

            if cand_ent is not None:
                begin_spans.append(span[0])
                end_spans.append(span[1])
                cand_entities.append(cand_ent)
                cand_entities_scores.append(cand_ent_scores)
            elif self.mode == "goldspans":
                cand_entities.append([])
                cand_entities_scores.append([])

#      # REMOVE SUBSPANS
#        begin_spans_final = []
#        end_spans_final = []
#        cand_entities_final = []
#        cand_entities_scores_final = []
#        for begin, end, cand_ent, cand_ent_score in zip(begin_spans, end_spans, cand_entities, cand_entities_scores):
#            subspan = False
#            for b, e in zip(begin_spans, end_spans):
#                if begin>=b and end<=e and (begin, end)!=(b,e):
#                    subspan = True
#            if not subspan:
#                begin_spans_final.append(begin)
#                end_spans_final.append(end)
#                cand_entities_final.append(cand_ent)
#                cand_entities_scores_final.append(cand_ent_score)

        return begin_spans, end_spans, cand_entities, cand_entities_scores

    def person_coreference(self, chunk_words, begin_spans, end_spans, cand_entities, cand_entities_scores,
                           begin_gm=None, end_gm=None, separation_indexes=None):
        """
        """
        mention_spans = (list(zip(begin_spans, end_spans)) if self.mode=="allspans"
                         else list(zip(begin_gm, end_gm)))
      # PERSON MENTIONS INDICES
        person_mentions_idx = []
        for i, cand_ent in enumerate(cand_entities):
            mention = ' '.join(chunk_words[mention_spans[i][0]:mention_spans[i][1]])
            if cand_ent != [] and len(mention)>=3 and cand_ent[0] in self.person_wikiids:
                person_mentions_idx.append(i)

      # SPANS
        spans = (self.all_spans(separation_indexes, self.args.max_span_width)
                 if self.mode=="allspans"
                 else list(zip(begin_gm,end_gm)))
        # for new found spans in allspans mode
        new_begin_spans = []
        new_end_spans = []

        for span in spans:
          # SPAN
            span_text = " ".join(chunk_words[span[0]:span[1]])
            if not span_text.islower():
                span_text = span_text.title()
          # SHORT WORDS
            if len(span_text)<=3:
                continue
          # SEARCH FOR COREFERENCES
            for i in person_mentions_idx:
              # NOT SUBSPAN
                if span[0]>=mention_spans[i][0] and span[1] <= mention_spans[i][1]:
                    continue

                person_mention = ' '.join(chunk_words[mention_spans[i][0]:mention_spans[i][1]])
                idx = person_mention.find(span_text)
                idx = idx if idx!=-1 else person_mention.title().find(span_text)
                if idx != -1:
                    if len(span_text) == len(person_mention):
                    # they are identical
                        continue
                    if idx > 0 and person_mention[idx-1].isalpha():
                    # found as a subword
                        continue
                    if ((idx + len(span_text) < len(person_mention))
                         and person_mention[idx+len(span_text)].isalpha()):
                    # found as a subword
                        continue

                  # COREFERENCE FOUND
                    if span in mention_spans:
                        j = mention_spans.index(span)
                        if not self.args.person_coreference_merge:
                            cand_entities[j], cand_entities_scores[j] = cand_entities[i], cand_entities_scores[i]
                        else: # merge cand_entities and scores
                            temp1 = list(zip(cand_entities_scores[i], cand_entities[i]))
                            temp2 = list(zip(cand_entities_scores[j], cand_entities[j]))
                            temp3 = sorted(set(temp1 + temp2), reverse=True)
                            # set() to remove duplicates
                            cand_entities_scores[j], cand_entities[j] = map(list, zip(*temp3[:self.args.max_cand_ent]))
                    else: # only in allspans mode
                        new_begin_spans.append(span[0])
                        new_end_spans.append(span[1])
                        cand_entities.append(cand_entities[i])
                        cand_entities_scores.append(cand_entities_scores[i])

        begin_spans.extend(new_begin_spans)
        end_spans.extend(new_end_spans)

        return begin_spans, end_spans, cand_entities, cand_entities_scores

    @staticmethod
    def all_spans(separation_indexes, max_span_width):
        # this function produces all possible text spans
        # spans doesn't extend over "sentences"

        def all_spans_aux(begin_idx, end_idx):
            for left_idx in range(begin_idx, end_idx):
                for length in range(1, max_span_width + 1):
                    if left_idx + length > end_idx:
                        break
                    yield left_idx, left_idx + length

        begin_idx = 0
        for end_idx in separation_indexes:
            for left, right in all_spans_aux(begin_idx, end_idx):
                yield left, right
            begin_idx = end_idx

# This condition gives no improvement, to Gerbil results even
# a very slight decrease (0.02%)
#
#        neighbour_words = [chunk_words[left-1] if left - 1 >= 0 else None,
#                       chunk_words[right] if right <= len(chunk_words)-1 else None] if self.el_mode else None
#
#
#        if neighbour_words:  # this checks if allspans mode
#            if neighbour_words[0] and neighbour_words[0][0].isupper() or \
#                            neighbour_words[1] and neighbour_words[1][0].isupper():
#                # if the left or the right span neighbout has uppercased
#                # first letter - do not search for coreference
#                # since most likely it is a subspan of a mention
#                return None



def reverse_dict(d, unique_values=False):
    new_d = dict()
    for k, v in d.items():
        if unique_values:
            assert(v not in new_d)
        new_d[v] = k
    return new_d

