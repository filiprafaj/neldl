from pathlib import Path

import preprocessing.bert_wrapper as bert_wrapper

base_folder = Path("..")
unk_ent_id = '0'
pure_embeddings_size = 300
bert_size_base = 768
bert_size_large = 1024
bert_size_base_string = bert_wrapper.BertWrapper.SIZE_BASE
bert_size_large_string = bert_wrapper.BertWrapper.SIZE_LARGE
bert_cased_string = bert_wrapper.BertWrapper.CASING_CASED
bert_uncased_string = bert_wrapper.BertWrapper.CASING_UNCASED
max_cand_ent = 20
max_span_width = 10
