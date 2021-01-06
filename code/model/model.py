import pickle

import numpy as np
import tensorflow as tf

import model.config as config
import model.util as util
from model.base_model import BaseModel


class Model(BaseModel):

    def __init__(self, args, next_element):
        super().__init__(args)
      # INPUT DATA (padded)
        (
        self.chunk_id,
        self.words, # tf.int64, (batch, max_words_len)
        self.words_len, # tf.int64, (batch)
        self.chars, # tf.int64
                    # (batch, max_words_len, max_word_len)
        self.chars_len, # tf.int64, (batch, max_word_len)
        self.begin_span, # tf.int64, (batch, spans_num)
        self.end_span, # tf.int64, (batch, spans_num)
        self.spans_len, # tf.int64, (batch)
        self.cand_entities, # tf.int64
                            # (batch, spans_num, max_cand_ents)
        self.cand_entities_scores, # tf.float32
                                   # (batch, spans_num, max_cand_ents)
        self.cand_entities_labels, # tf.int64
                                   # (batch, spans_num, max_cand_ents)
        self.cand_entities_len, # tf.int64, (batch, spans_num)
        self.ground_truth, # tf.int64, (batch, spans_num)
        self.ground_truth_len, # tf.int64, (batch)
        self.begin_gm, # tf.int64, (batch, max_gold_spans)
        self.end_gm, # tf.int64, (batch, max_gold_spans)
        self.bert_word_embeddings
        ) = next_element

        # self.begin_span = tf.cast(self.begin_span, tf.int64)
        # self.end_span = tf.cast(self.end_span, tf.int64)
        # self.words_len = tf.cast(self.words_len, tf.int64)
        self.embeddings_size = config.pure_embeddings_size
        # if self.args.use_chars:
        #     self.embeddings_size += 2*self.args.char_lstm_units

        if self.args.bert_size == config.bert_size_base_string:
            self.bert_word_embeddings.set_shape([None, None, config.bert_size_base])
        elif self.args.bert_size == config.bert_size_large_string:
            self.bert_word_embeddings.set_shape([None, None, config.bert_size_large])

        # self.embeddings_size = int(self.embeddings_size/2)

        with open(config.base_folder/"data/experiments"/self.args.experiment_name/"word_char_maps.pickle", "rb") as f:
            _, id2word, _, id2char, _, _ = pickle.load(f)
            self.nwords = len(id2word)
            self.nchars = len(id2char)

        if self.args.max_cand_ent:
            self.cand_entities_len = tf.minimum(self.cand_entities_len, self.args.max_cand_ent)
            max_cand_ent = tf.reduce_max(self.cand_entities_len)

            self.cand_entities = tf.slice(self.cand_entities, [0, 0, 0],
                                          [-1, -1, max_cand_ent])
            self.cand_entities_scores = tf.slice(self.cand_entities_scores, [0, 0, 0],
                                                 [-1, -1, max_cand_ent])
            self.entity_embeddings = tf.slice(self.entity_embeddings, [0, 0, 0, 0, 0],
                                              [-1, -1, max_cand_ent, -1, -1])
            if not self.args.running_mode=="server":
                self.cand_entities_labels = tf.slice(self.cand_entities_labels, [0, 0, 0],
                                                     [-1, -1, max_cand_ent])

      # LOSS MASK
        self.loss_mask = tf.sequence_mask(self.cand_entities_len, tf.shape(self.cand_entities_scores)[2],
                                          dtype=tf.float32)


    def add_placeholders(self):
        """Define placeholders ~ entries to computational graph"""
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="dropout_keep_prob")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


    def add_embeddings_op(self):
      # CHAR EMBEDDINGS (Section 2.1)
        with tf.variable_scope("char_emb"):
            if self.args.use_chars:
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.nchars, self.args.char_lstm_units],
                        trainable=True)
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.chars,
                                                         name="char_embeddings")
                # char_embeddings: tf.float32, shape=[None, None, None, char_lstm_units],
                # shape = (batch, max_words_len, max_word_len, char_lstm_units)

                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[2], self.args.char_lstm_units],
                                             name="flatten_inputs")
                # shape = (batch*max_words_len, max_word_len, char_lstm_units)

                char_lengths = tf.reshape(self.chars_len, shape=[s[0] * s[1]], name="flatten_lengths")
                # shape = (batch*max_words_len)

              # LSTM on chars
                _, ((_, output_fw), (_, output_bw)) = util.bi_LSTM(char_embeddings,
                                                                   char_lengths, self.args.char_lstm_units)

                output = tf.concat([output_fw, output_bw], axis=-1, name="concat_fw_bw_output")
                char_embeddings = tf.reshape(output, shape=[s[0], s[1], 2 * self.args.char_lstm_units],
                                             name="unflatten_output")
                # shape = (batch, max_words_len, 2*char_lstm_units)

      # INPUT EMBEDDINGS (Section 2.2)
        with tf.variable_scope("word_emb"):
            _word_embeddings = tf.Variable(tf.constant(0.0, shape=[self.nwords, config.pure_embeddings_size]),
              name="_word_embeddings",
              dtype=tf.float32,
              trainable=False)

            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [self.nwords, config.pure_embeddings_size])
            self.word_embedding_init = _word_embeddings.assign(self.word_embeddings_placeholder)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.words, name="word_embeddings")
          # BERT WORD EMBEDDINGS
            self.word_embeddings = tf.concat([word_embeddings, self.bert_word_embeddings], axis=-1)
            if self.args.use_chars:
                self.word_embeddings = tf.concat([self.word_embeddings, char_embeddings], axis=-1,
                                                 name="concat_word_char_emb")
                #shape = (batch, max_words_len, embeddings_size+bert_size+2*char_lstm_units)

            self.pure_word_embeddings = self.word_embeddings
            self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout_keep_prob)

      # ENTITY EMBEDDINGS
        with tf.variable_scope("entities"):
            from preprocessing.util import load_wikiid2nnid
            self.nentities = len(load_wikiid2nnid(self.args.experiment_name))
          # LOAD EMBEDDINGS
            _entity_embeddings = tf.Variable(
                tf.constant(0.0, shape=[self.nentities, config.pure_embeddings_size]),
                name="_entity_embeddings",
                dtype=tf.float32,
                trainable=self.args.train_ent_vecs)

            self.entity_embeddings_placeholder = tf.placeholder(tf.float32, [None, config.pure_embeddings_size])
            self.entity_embedding_init = _entity_embeddings.assign(self.entity_embeddings_placeholder)
            self.entity_embeddings = tf.nn.embedding_lookup(_entity_embeddings, self.cand_entities,
                name="entity_embeddings")

            if self.entity_embeddings.get_shape().as_list()[-1] != self.embeddings_size:
                with tf.variable_scope("entity_emb_ffnn"):
                    self.entity_embeddings = util.projection(self.entity_embeddings, self.embeddings_size, model=self)
            self.entity_embeddings_no_dropout = self.entity_embeddings

          # REGULARIZATION
            if self.args.ent_vecs_dropout:
                self.entity_embeddings = tf.nn.dropout(self.entity_embeddings, self.dropout_keep_prob)

    # INPUT EMBEDDINGS (Section 2.2)
    def add_context_emb_op(self):
        """
        this method creates a bidirectional LSTM layer
        takes the word embeddings
        outputs the context-aware word embeddings
        """
        with tf.variable_scope("context"):
            (output_fw, output_bw), _  = util.bi_LSTM(self.word_embeddings,
                                                      self.words_len,
                                                      self.embeddings_size/2)
            output = tf.concat([output_fw, output_bw], axis=-1)
            self.context_emb = tf.nn.dropout(output, self.dropout_keep_prob)
          # shape = (batch, max_words_len, 2*(embeddings_size)

    # SPAN EMBEDDING (Section 2.4)
    # modeled by g^m = [x_q; x_r; \hat(x)^m]  (formula (2) of paper)
    def add_span_emb_op(self):
        span_emb_list = []
      # IF "boundaries" use x_q and x_r
        if self.args.span_emb.find("boundaries") != -1:
            boundaries_input_vecs = (self.context_emb if self.args.context_emb
                                     else self.word_embeddings)
#            boundaries_input_vecs = self.word_embeddings

          # SPAN START EMBEDDING
            # the tile creates a 2d tensor with indexes for batch dim
            batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span, out_type=tf.int64)[0]), 1),
                                  [1, tf.shape(self.begin_span)[1]])
            span_indices = tf.stack([batch_index, self.begin_span], 2)
            span_start_emb = tf.gather_nd(boundaries_input_vecs, span_indices)
            # shape = (batch, spans_num, emb)
            span_emb_list.append(span_start_emb)

          # SPAN END EMBEDDING
            batch_index = tf.tile(tf.expand_dims(tf.range(tf.shape(self.begin_span, out_type=tf.int64)[0]), 1),
                                  [1, tf.shape(self.begin_span)[1]])
            span_indices = tf.stack([batch_index, tf.nn.relu(self.end_span-1)], 2)
            # -1 because the end of span is exclusive  [start, end)
            # relu: 0 doesn't become -1 (no valid end index is 0)
            span_end_emb = tf.gather_nd(boundaries_input_vecs, span_indices)
            span_emb_list.append(span_end_emb)

      # IF "head" use the head mechanism \hat(x)^m (formula (3))
        if self.args.span_emb.find("head") != -1:
            # x_k
            head_input_vecs = self.context_emb if self.args.context_emb else self.word_embeddings
          # SPAN EMBEDDING
            max_span_width = tf.cast(self.args.max_span_width, dtype=tf.int64)
            self.max_span_width = tf.minimum(max_span_width, tf.reduce_max(self.end_span - self.begin_span))
            span_indices = tf.range(self.max_span_width) + tf.expand_dims(self.begin_span, 2)
            # [batch, spans_num, max_span_width]
            # e.g. [0,1] + [begin_idx] -> [begin_idx, begin_idx+1]
            span_indices = tf.minimum(tf.shape(self.word_embeddings, out_type=tf.int64)[1] - 1, span_indices)
            # [batch, spans_num, max_span_width] with values <= #words

            batch_index = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(span_indices, out_type=tf.int64)[0]),
                                  1), 2), [1, tf.shape(span_indices)[1], tf.shape(span_indices)[2]])

            span_indices = tf.stack([batch_index, span_indices], 3)
            # [batch, spans_num, max_span_width, 2]
            # the last dimension is [row,col] for gather_nd

            span_text_emb = tf.gather_nd(head_input_vecs, span_indices)

          # HEAD SCORES alpha_k = <w_alpha, x_k>
            with tf.variable_scope("head_scores_ffnn"):
                # from [batch, max_words_len, embeddings_size] to [..., 1]
                self.head_scores = util.projection(head_input_vecs, 1, model=self)
            span_head_scores = tf.gather_nd(self.head_scores, span_indices)

          # MASK invalid indices created above (span_indices)
            span_width = self.end_span - self.begin_span
            temp_mask = tf.sequence_mask(span_width, self.max_span_width, dtype=tf.float32)
            span_mask = tf.expand_dims(temp_mask, 3)
            # [batch, spans_num, max_span_width, 1]
            span_mask = tf.minimum(1.0, tf.maximum(self.args.zero, span_mask))

          # ATTENTION formula (3) computation
            span_attention = tf.nn.softmax(span_head_scores + tf.log(span_mask), axis=2)
            # [batch, spans_num, max_span_width, 1]

            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 2)
            # [batch, spans_num, emb]

            span_emb_list.append(span_head_emb)

        span_emb_concat = tf.concat(span_emb_list, 2)
        # [batch, spans_num, emb] formula (2): g^m=[x_q; x_r; \hat(x)^m]

        with tf.variable_scope("span_emb_ffnn"):
            # The span embedding have different size
            # depending on the chosen hyperparameters.
            # We project it to match the entity embeddings
            if self.args.span_emb_ffnn[0] == 0:
                self.span_emb = util.projection(span_emb_concat, self.embeddings_size, model=self)
            else:
                hidden_layers, hidden_size = self.args.span_emb_ffnn[0], self.args.span_emb_ffnn[1]
                self.span_emb = util.ffnn(span_emb_concat, hidden_layers, hidden_size, self.embeddings_size,
                                                self.dropout_keep_prob if self.args.ffnn_dropout else None,
                                                model=self)
            # [batch, spans_num, embeddings_size]

    # SIMILARITY SCORES (Section 2.6)
    # formula (6) <x^m, y_j>
    def add_similarity_score_op(self):
      with tf.variable_scope("similarity"):
          scores = tf.matmul(tf.expand_dims(self.span_emb, 2), self.entity_embeddings, transpose_b=True)
          self.similarity_scores = tf.squeeze(scores, axis=2)
          # [batch, spans_num, 30]


     # LONG-RANGE CONTEXT ATTENTION SCORE (Section 2.6.1)
     # attention model described in [Ganea and Hofmann, 2017]
     # comments should correspond to the notation in the paper
    def add_long_range_context_attention_op(self):

        attention_entity_emb = (self.entity_embeddings_no_dropout if self.args.attention_ent_vecs_no_dropout
                                else self.entity_embeddings)

        with tf.variable_scope("attention"):
          # CANDIDATE ENTITY EMBEDDINGS
            x_e_voters = attention_entity_emb # x_e
            # restrict the number of entities that participate in the forming of the x_c context vector
            if self.args.attention_max_cand_ent:
                attention_max_cand_ent = tf.reduce_max(self.cand_entities_len)
                x_e_voters = tf.slice(attention_entity_emb, [0, 0, 0, 0],
                                      [-1, -1, attention_max_cand_ent, -1])


          # WORD EMBEDDINGS (x_w)
            K = tf.cast(self.args.attention_K, dtype=tf.int64)
          # MASKS so we don't go outside of words
            left_mask = tf.sequence_mask(self.begin_span, K, dtype=tf.float32)
            right_mask = tf.sequence_mask(tf.expand_dims(self.words_len, 1) - self.end_span, K, dtype=tf.float32)
            cntxt_mask = tf.concat([left_mask, right_mask], 2)
            # [batch, spans_num, 2*K]
            # e.g.:
            #  1,  1,  1,  0,  0 |  1,  1,  0,  0,  0
            # -1, -2, -3, -4, -5   +1, +2, +3, +4, +5
            cntxt_mask = tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, cntxt_mask)))
            # 0.0 in place of ones, negative number in place of zeros
            # (the mask is then added to scores)
          # INDICES of neigbour words
            leftcntxt_indices = tf.maximum(tf.cast(0,dtype=tf.int64), tf.range(-1, -K - 1, -1) +
                                          tf.expand_dims(self.begin_span, 2))
            rightcntxt_indices = tf.minimum(tf.shape(self.pure_word_embeddings, out_type=tf.int64)[1] - 1,
                                            tf.range(K) + tf.expand_dims(self.end_span, 2))
            # [batch, spans_num, K] both

            cntxt_indices = tf.concat([leftcntxt_indices, rightcntxt_indices], 2)
            # [batch, spans_num, 2*K]

            batch_index = tf.tile(tf.expand_dims(
                                  tf.expand_dims(tf.range(tf.shape(cntxt_indices, out_type=tf.int64)[0]), 1), 2),
                                  [1, tf.shape(cntxt_indices)[1], tf.shape(cntxt_indices)[2]])

            cntxt_indices = tf.stack([batch_index, cntxt_indices], 3)
            # [batch, spans_num, 2*K, 2]
            # the last dimension is [batch, word] for gather_nd

            if self.args.context_emb and self.args.attention_using_context_emb:
                with tf.variable_scope("attention_context_emb_ffnn"):
                    if self.args.attention_context_emb_ffnn[0] == 0:
                        x_w = util.projection(self.context_emb, self.embeddings_size, model=self)
                    else:
                        hidden_layers = self.args.attention_context_emb_ffnn[0]
                        hidden_size = self.args.attention_context_emb_ffnn[1]
                        x_w = util.ffnn(self.context_emb,
                                        hidden_layers, hidden_size, self.embeddings_size,
                                        self.dropout_keep_prob if self.args.ffnn_dropout else None,
                                        model=self)
            else:
                x_w = util.projection(self.word_embeddings, self.embeddings_size)

            x_w = tf.gather_nd(x_w, cntxt_indices)
            # [batch, spans_num, 2K, embeddings_size]

          # DIAGONAL MATRIX A
            if self.args.attention_AB:
                A = tf.get_variable("att_A", [self.embeddings_size])
                x_e_voters = A * x_e_voters # paper: x_e A
                # [batch, spans_num, max_cand_ent, embeddings_size]
                # max_can_ent can be attention_max_cand_ent

          # SCORES u(w)
            scores = tf.matmul(x_w, x_e_voters, transpose_b=True)
            # [b, spans, 2K, max_num_entities]
            # paper: x_e A x_w
            scores = tf.reduce_max(scores, reduction_indices=[-1])
            # max score of words from each span context
            # paper: max x_e A x_w
            scores = scores + cntxt_mask # mask "out of window" words

          # TOP R
            # avoid words with negative or zero maximal score
            # padding candidate entities can have score 0
            topR_values, _ = tf.nn.top_k(scores, self.args.attention_R)
            # [batch, spans_num, R]
            R_value = topR_values[:, :, -1] # R-th best value
            # [batch, spans_num]
            R_value = tf.maximum(self.args.zero, R_value)
            threshold = tf.tile(tf.expand_dims(R_value, 2), [1, 1, 2 * K])
            # [batch, spans_num, 2K] 2K filled with corresp. R value
            scores = scores - tf.to_float(((scores - threshold) < 0)) * 50
            # -50 if score<thr else 0

          # WEIGHTS \beta(w)
            scores = tf.nn.softmax(scores, axis=2)
            # [batch, spans_num, 2K]
            scores = tf.expand_dims(scores, 3)
            # [batch, spans_num, 2K, 1]

          # SCORE \Psi(e,c) (from [Ganea and Hofmann, 2017])
            x_w_voters = tf.reduce_sum(scores * x_w, 2)
            # paper: /sum /beta(w) x_e B x_w
            # [batch, spans_num, 2K, 1]  *
            # [batch, spans_num, 2K, emb_size]
            #  = [bgatch, spans_num, 2K, emb_size]

            if self.args.attention_AB:
              # DIAGONAL MATRIX B
                B = tf.get_variable("att_B", [self.embeddings_size])
                x_w_voters = B * x_w_voters
            x_w_voters = tf.expand_dims(x_w_voters, 3)
            # [batch, spans_num, emb_size, 1]
            # [batch, spans_num, 30, emb_size=embeddings_size]  mul with  [batch, spans_num, emb_size, 1]
            x_e_x_w = tf.matmul(attention_entity_emb, x_w_voters)
            # paper: /sum /beta(w) x_w
            # [batch, spans_num, max_num_entities, 1]
            x_e_x_w = tf.squeeze(x_e_x_w, axis=3)
            # [batch, spans_num, max_num_entities]

            self.attention_scores = x_e_x_w


    def pem(self, log=True):
        if not log:
            return self.cand_entities_scores
        else:
            return tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, self.cand_entities_scores)))


    def add_local_scores_op(self):
        # log_cand_ent_scores not used here
        self.log_cand_entities_scores = tf.log(tf.minimum(1.0, tf.maximum(self.args.zero, self.cand_entities_scores)))
      # LOCAL SCORE COMPONENTS
        stack_values = []
        if self.args.local_score_components.find("similarity") != -1:
            stack_values.append(self.similarity_scores)
        if self.args.local_score_components.find("pem") != -1:
            self.pem_scores = self.pem(self.args.pem_log)
            stack_values.append(self.pem_scores)
        if self.args.local_score_components.find("attention") != -1:
            stack_values.append(self.attention_scores)
        assert len(stack_values) > 0
        if len(stack_values) == 1: # skip the final ffnn
            self.final_scores = stack_values[0]
            return
        predictors = tf.stack(stack_values, 3)

        with tf.variable_scope("local_score_ffnn"):
            if self.args.local_score_ffnn[0] == 0:
                self.final_scores = util.projection(predictors, 1, model=self)
            else:
                hidden_layers, hidden_size = self.args.local_score_ffnn[0], self.args.local_score_ffnn[1]
                self.final_scores = util.ffnn(predictors, hidden_layers, hidden_size, 1,
                                              self.dropout_keep_prob if self.args.ffnn_dropout else None, model=self)
            self.final_scores = tf.squeeze(self.final_scores, axis=3)
            self.final_scores =  - (1 - self.loss_mask) * 50 + self.final_scores


    def add_global_scores_op(self):
        global_entity_emb = self.entity_embeddings_no_dropout

        with tf.variable_scope("global"):
            self.final_local_scores = self.final_scores
          # CANDIDATES local score >= global_thr (~/gamma') -> voters
            gmask = tf.to_float(((self.final_local_scores - self.args.global_thr) >= 0))
            # [batch, spans_num, max_cand_ent]
            masked_entity_emb = global_entity_emb * tf.expand_dims(gmask, axis=3)
            # [batch, num_spans, max_cand_ent, embeddings_size] *
            # [batch, spans_num, max_cand_ent, 1]
            batch_size = tf.shape(masked_entity_emb)[0]

          # SUM of embeddings of all voters
            all_voters_emb = tf.reduce_sum(tf.reshape(masked_entity_emb, [batch_size, -1,
                                                                          self.embeddings_size]),
              axis=1, keepdims=True)
            # [batch, 1, embeddings_size]

          # SUM of all voters embeddings in each span
            span_voters_emb = tf.reduce_sum(masked_entity_emb, axis=2)
            # [batch, spans_num, embeddings_size]

          # SUM OF ALL OTHER SPANS voters for each span
            valid_voters_emb = all_voters_emb - span_voters_emb
            # [batch, 1, embeddings_size] - [batch, spans_num, embeddings_size]
            # = [batch, spans_num, embeddings_size]  (broadcasting)

          # NORMALIZE
            valid_voters_emb = tf.nn.l2_normalize(valid_voters_emb, axis=2)

          # COSINE SIMILARITY between the entity embedding
          # and the normalized average of all others spans voters
            self.global_voting_scores = tf.squeeze(
                    tf.matmul(global_entity_emb, tf.expand_dims(valid_voters_emb, axis=3)), axis=3)
            # [batch, spans_num, max_cand_ent, embeddings_size] *
            # [batch, spans_num, embeddings_size, 1]
            # = [batch,spans_num, cant_ent_num, 1]
            # squeeze to [batch, spans_num, max_cand_ent]

          # GLOBAL SCORE COMPONENTS
            stack_values = []
            if self.args.global_score_components.find("pem") != -1:
                self.pem_scores = self.pem(self.args.gpem_log)
                stack_values.append(self.pem_scores)
            if self.args.global_score_components.find("local") != -1:
                stack_values.append(self.final_local_scores)
            stack_values.append(self.global_voting_scores)
            predictors = tf.stack(stack_values, 3)

            with tf.variable_scope("global_score_ffnn"):
                if self.args.global_score_ffnn[0] == 0:
                    self.final_scores = util.projection(predictors, 1, model=self)
                else:
                    hidden_layers, hidden_size = self.args.global_score_ffnn[0], self.args.global_score_ffnn[1]
                    self.final_scores = util.ffnn(predictors, hidden_layers, hidden_size, 1,
                      self.dropout_keep_prob if self.args.ffnn_dropout else None, model=self)
                self.final_scores = tf.squeeze(self.final_scores, axis=3)
                # [batch, spans_num, max_cand_ent]

# previous version (ablation)
#        self.final_local_scores = self.final_scores
#        # TOP K candidates from each span (based on cosine similarity)
#        if self.args.global_topkfromallspans:
#          # MASK - masking out candidates for the same mention
#            batch_num = tf.shape(self.final_scores)[0]
#            spans_num = tf.shape(self.final_scores)[1]
#            max_cand_ent = tf.shape(self.final_scores)[2]
#            new_size = spans_num * max_cand_ent
#            temp = tf.diag(tf.ones([spans_num]))
#            temp = tf.tile(tf.expand_dims(temp, axis=2), [1, 1, max_cand_ent])
#            temp = tf.reshape(temp, [spans_num, new_size])
#            mask = tf.reshape(tf.tile(tf.expand_dims(temp, axis=1), [1, max_cand_ent, 1]),
#                              [new_size, new_size])
#            mask = 1 - mask
#
#          # COSINE SIMILARITY
#            all_entities = tf.reshape(global_entity_emb, [batch_num, new_size, self.embeddings_size])
#            all_scores = tf.matmul(all_entities, all_entities, transpose_b=True)
#            # [batch, new_size, new_size]
#            filtered_scores = all_scores * mask
#          # TOP K
#            top_values, _ = tf.nn.top_k(filtered_scores, self.args.global_topkfromallspans)
#            # [batch, new_size, K]
#            if self.args.global_topkfromallspans_onlypositive:
#                top_values = tf.maximum(top_values, self.args.zero)
#                # avoid cand ent with score < 0
#            self.global_voting_scores = tf.reduce_mean(top_values, axis=2)
#            # [batch, new_size]
#            self.global_voting_scores = tf.reshape(self.global_voting_scores,
#                                                   [batch_num, spans_num, max_cand_ent])
#        else:
#          # SPANS with one candidate entity (unambiguous)
#            if self.args.global_mask_unambiguous:
#                mask = tf.sequence_mask(tf.equal(self.cand_entities_len, 1),
#                                                  tf.shape(self.final_scores)[2],
#                                                  dtype=tf.float32)
#          # TOP K candidates for each span (based on local score)
#            elif self.args.global_topk:
#                top_values, _ = tf.nn.top_k(self.final_local_scores, self.args.global_topk)
#                # [batch, spans_num, K]
#                K_value = top_values[:, :, -1]
#                # [batch, spans_num]
#                if self.args.global_topkthr:
#                    K_value = tf.maximum(self.args.global_topkthr, K_value)
#                    # avoid keeping cand ent with score < K_value
#                    # even if they are the top for this span
#                threshold = tf.tile(tf.expand_dims(K_value, 2), [1, 1, tf.shape(self.final_scores)[-1]])
#                # [batch, spans_num, max_cand_ent]
#                mask = tf.to_float(((self.final_local_scores - threshold) >= 0))
#
#          # CANDIDATES with local score >= global_thr (~/gamma')
#            else:
#                mask = tf.to_float(((self.final_local_scores - self.args.global_thr) >= 0))
#                # [batch, spans_num, max_cand_ent]
#
#            gmask = mask * self.loss_mask
#
#            if self.args.global_mask_scale_each_span_voters_to_one:
#                temp = tf.reduce_sum(gmask, axis=2, keep_dims=True)
#                # [batch, spans_num, 1]
#                temp = tf.where(tf.less(temp, 1e-4), temp, 1. / (temp + 1e-4))
#                gmask = gmask * temp
#            elif self.args.global_mask_based_on_local_score:
#                gmask = gmask * tf.nn.softmax(self.final_local_scores)
#            self.gmask = gmask
#
#            masked_entity_emb = global_entity_emb * tf.expand_dims(gmask, axis=3)
#            # [batch, num_spans, max_cand_ent, embeddings_size] *
#            # [batch, spans_num, max_cand_ent, 1]
#            batch_size = tf.shape(masked_entity_emb)[0]
#
#          # SUM of embeddings of all voters
#            all_voters_emb = tf.reduce_sum(tf.reshape(masked_entity_emb, [batch_size, -1, self.embeddings_size]),
#                                           axis=1, keepdims=True)
#            # [batch, 1, embeddings_size]
#
#          # SUM of all embeddings in each span
#            span_voters_emb = tf.reduce_sum(masked_entity_emb, axis=2)
#            # [batch, spans_num, embeddings_size]
#
#          # FOR EACH SPAN SUM of all others spans voters
#            valid_voters_emb = all_voters_emb - span_voters_emb
#            # [batch, 1, embeddings_size] - [batch, spans_num, embeddings_size]
#            # = [batch, spans_num, embeddings_size]  (broadcasting)
#
#          # NORMALIZE
#            if self.args.global_norm_or_mean == "norm":
#                valid_voters_emb = tf.nn.l2_normalize(valid_voters_emb, axis=2)
#          # OR DIVIDE by number of voters (mean)
#            else:
#                all_voters_num = tf.reduce_sum(gmask)  # scalar
#                span_voters_num = tf.reduce_sum(gmask, axis=2)  # [batch, spans_num]
#                valid_voters_emb = valid_voters_emb / tf.expand_dims(all_voters_num - span_voters_num, axis=2)
#                # scalar - [batch, spans_num]  = [batch, spans_num]  (broadcasting)
#          # COSINE SIMILARITY between the entity embedding
#          # and the normalized average of all others spans voters
#
#            self.global_voting_scores = tf.squeeze(tf.matmul(global_entity_emb,
#                                                             tf.expand_dims(valid_voters_emb, axis=3)),
#                                                   axis=3)
#            # [batch, spans_num, max_cand_ent, embeddings_size] *
#            # [batch, spans_num, embeddings_size, 1]
#            # = [batch,spans_num, cant_ent_num, 1]
#            # squeeze to [batch, spans_num, max_cand_ent]



    def add_loss_op(self):
        cand_entities_labels = tf.cast(self.cand_entities_labels, tf.float32)
      # GOLD ENTITY ( max(0, \gamma - score) )
        loss1 = cand_entities_labels * tf.nn.relu(self.args.gamma - self.final_scores)
      # NOT GOLD ENTITY ( max(0, score) )
        loss2 = (1 - cand_entities_labels) * tf.nn.relu(self.final_scores)
        self.loss = loss1 + loss2
        if self.args.global_model:
          # ALSO LOCAL LOSS
            loss3 = cand_entities_labels * tf.nn.relu(self.args.gamma - self.final_local_scores)
            loss4 = (1 - cand_entities_labels) * tf.nn.relu(self.final_local_scores)
            self.loss = loss1 + loss2 + loss3 + loss4
      # LOSS
        self.loss = self.loss_mask * self.loss
        self.loss = tf.reduce_sum(self.loss)


    def init_embeddings(self):
      # INITIALIZE EMBEDDINGS
      print("\n ----- initialize embeddings -----\n")
      word_embeddings_npy, entity_embeddings_npy = util.load_embeddings(self.args)
      self.sess.run(self.word_embedding_init, feed_dict={self.word_embeddings_placeholder: word_embeddings_npy})
      self.sess.run(self.entity_embedding_init,feed_dict={self.entity_embeddings_placeholder: entity_embeddings_npy})


    def build(self):
        self.add_placeholders()
        self.add_embeddings_op()
        if self.args.context_emb:
            self.add_context_emb_op()
        if self.args.local_score_components.find("similarity") != -1:
            self.add_span_emb_op()
            self.add_similarity_score_op()
        if self.args.local_score_components.find("attention") != -1:
            self.add_long_range_context_attention_op()

        self.add_local_scores_op()

        if self.args.global_model:
            self.add_global_scores_op()

        if self.args.running_mode.startswith("train"):
            self.add_loss_op()
            self.add_train_op(self.args.lr_method, self.args.adam_beta1, self.args.adam_beta2, self.lr,
              self.loss, self.args.clip)
        if self.args.running_mode == "train_continue":
            self.restore_session()
        elif self.args.running_mode == "train":
            self.initialize_session()
            self.init_embeddings()
