import argparse
import zipfile

import model.config as config
import numpy as np
import unidecode

import preprocessing.bert_wrapper as bert_wrapper


def generate_bert_embeddings(args, filepath, output_npz):
  # PROCESS FILE, split by sentence
    with open(filepath) as fin:
        print("Processing file ", filepath)
        sentences = [[]]
        docid = ''
        docids = []
        last_sentence_of_doc_to_docid = {}
        for line in fin:
            line = line.rstrip()  # omit the '\n' character
            if args.bert_casing == "cased": # uncased does it itself
                line = unidecode.unidecode(line)
            # accents causes problems in bert_wrapper
            # (see bert/generate_entity_embeddings.py)
            if line == "DOCEND":
                if len(sentences[-1]) > 0:
                    sentences.append([])
                last_sentence_of_doc_to_docid[len(sentences)-2] = docid
            elif line == "*NL*":
                if len(sentences[-1]) > 0:
                    sentences.append([])
            elif line == '.':
                if len(sentences[-1])<512:
                    sentences[-1].append(line)
                    sentences.append([])
                else:
                    sentences.append([line])
                    sentences.append([])
            elif line.startswith("MMSTART_"):
                pass
            elif line == "MMEND":
                pass
            elif line.startswith("DOCSTART_"):
                docid = line[len("DOCSTART_"):]
                while docid in docids:
                    docid=docid+'.'
                docids.append(docid)
            else:
                if len(sentences[-1])<512:
                    sentences[-1].append(line)
                else:
                    sentences.append([line])

    if len(sentences[-1]) == 0:
        del sentences[-1]

    print("Loaded file with {} sentences and {} words.".format(len(sentences), sum(map(len, sentences))))

  # BERT, save embeddings of file: each sentence ~ list of embeddings
    bert = bert_wrapper.BertWrapper(language=args.language, size=args.bert_size, casing=args.bert_casing,
                                    layer_indices=args.layer_indices, with_cls=args.with_cls,
                                    threads=args.threads, batch_size=args.batch_size)
    with zipfile.ZipFile(output_npz, mode='w', compression=zipfile.ZIP_STORED) as output_npz:
        doc_embeddings = []
        for i, embeddings in enumerate(bert.bert_embeddings(sentences)):
            if (i + 1) % 100 == 0: print("Processed {}/{} sentences.".format(i + 1, len(sentences)))
            assert len(sentences[i])==len(embeddings)
            doc_embeddings.append(embeddings)
            if i in last_sentence_of_doc_to_docid:
                with output_npz.open(last_sentence_of_doc_to_docid[i], mode='w') as embeddings_file:
                    np.save(embeddings_file, np.array(doc_embeddings))
                doc_embeddings = []
        print("Embeddings saved to "+str(output_npz))

def main(args):
    output_folder = config.base_folder/("data/new_datasets_bert_"+args.bert_casing+'_'+args.bert_size)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    for filepath in (config.base_folder/"data/base_data/new_datasets").glob("*.txt"):
        output_npz = output_folder/(filepath.name[:-4]+".npz")
        generate_bert_embeddings(args, filepath, output_npz)

    print("Done, all embeddings saved to "+str(output_folder))

def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_casing", default=bert_wrapper.BertWrapper.CASING_UNCASED, help="Bert model casing")
    parser.add_argument("--bert_size", default=bert_wrapper.BertWrapper.SIZE_BASE, help="Bert model size")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--language", default=bert_wrapper.BertWrapper.LANGUAGE_ENGLISH, help="Bert model language")
    parser.add_argument("--layer_indices", default="-1,-2,-3,-4", type=str, help="Bert model layers to average")
    parser.add_argument("--threads", default=4, type=int, help="Threads to use")
    parser.add_argument("--with_cls", default=False, action="store_true", help="Return also CLS embedding")
    args = parser.parse_args()
    args.layer_indices = list(map(int, args.layer_indices.split(',')))
    return args
if __name__ == "__main__":
    main(_parse_args())
