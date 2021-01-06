import argparse
import json
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer

import model.config as config
import preprocessing.bert_wrapper as bert_wrapper
import preprocessing.util as util
from model.util import load_train_args

from server.nn_processing import NNProcessing


class GetHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print('\n', post_data)
        self.send_response(200)
        self.end_headers()

        text, spans, redirections, threshold = read_json(post_data)

        response = nnprocessing.process(text, spans, redirections=redirections, threshold=threshold)

        print("Response from server.py:\n", response)
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return


def read_json(post_data):
    data = json.loads(post_data.decode("utf-8"))
    text = data["text"]
    spans = [(int(j["start"]), int(j["length"])) for j in data["spans"]]
    redirections = data["redirections"]
    threshold = data["threshold"] if "threshold" in data else None
    return text, spans, redirections, threshold


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default=None, help="under folder data/experiments/")
    parser.add_argument("--training_name", default=None)
    parser.add_argument("--checkpoint_model_num", default=None, help="e.g. '7 if you want checkpoints/model-7")

    parser.add_argument("--each_entity_only_once", dest='each_entity_only_once', action='store_true')
    parser.add_argument("--no_each_entity_only_once", dest='each_entity_only_once', action='store_false')
    parser.set_defaults(each_entity_only_once=False)

    # those are for building the entity set
    parser.add_argument("--person_coreference", dest='person_coreference', action='store_true')
    parser.add_argument("--no_person_coreference", dest='person_coreference', action='store_false')
    parser.set_defaults(person_coreference=True)

    parser.add_argument("--person_coreference_merge", dest='person_coreference_merge', action='store_true')
    parser.add_argument("--no_person_coreference_merge", dest='person_coreference_merge', action='store_false')
    parser.set_defaults(person_coreference_merge=True)

  # BERT
    parser.add_argument("--bert_casing", default=bert_wrapper.BertWrapper.CASING_UNCASED, help="Bert model casing")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--language", default=bert_wrapper.BertWrapper.LANGUAGE_ENGLISH, help="Bert model language")
    parser.add_argument("--layer_indices", default="-1,-2,-3,-4", type=str, help="Bert model layers to average")
    parser.add_argument("--size", default=bert_wrapper.BertWrapper.SIZE_BASE, help="Bert model size")
    parser.add_argument("--threads", default=4, type=int, help="Threads to use")
    parser.add_argument("--with_cls", default=False, action="store_true", help="Return also CLS embedding")

    args = parser.parse_args()
    args.layer_indices = list(map(int, args.layer_indices.split(',')))
    if args.person_coreference_merge:
        args.person_coreference = True
    print("\nargs:\n", args)

    args.experiment_folder = config.base_folder/"data/experiments"/args.experiment_name

    args.output_folder = (config.base_folder/"data/experiments"/
                          args.experiment_name/"training_folder"/args.training_name)

    train_args = load_train_args(args.output_folder, "server")

    print("\ntrain_args:\n", train_args)
    return args, train_args


def terminate():
    tee.close()


if __name__ == "__main__":
    args, train_args = _parse_args()
    with open(args.experiment_folder/"args_and_logs/create_tfrecords_args.pickle", 'rb') as handle:
        create_tfrecords_args = pickle.load(handle)
    print("\ncreate_tfrecords_args:\n", create_tfrecords_args)

    fetchCandidateEntities = util.FetchCandidateEntities(create_tfrecords_args)
    bert = bert_wrapper.BertWrapper(language=args.language, size=args.size, casing=args.bert_casing,
                                    layer_indices=args.layer_indices, with_cls=args.with_cls,
                                    threads=args.threads, batch_size=args.batch_size)

    nnprocessing = NNProcessing(train_args, args, fetchCandidateEntities, bert)

    server = HTTPServer(('localhost', 5555), GetHandler)
    print('Starting server at http://localhost:5555')

    from model.util import Tee
    tee = Tee('server.txt', 'w')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        terminate()
        exit(0)

