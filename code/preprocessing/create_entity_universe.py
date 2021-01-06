import argparse
import pickle

import model.config as config
from model.util import Tee

import preprocessing.util as util
from preprocessing.samples_generator import SamplesGenerator


def create_entity_universe(args, output_folder):
    entity_universe = set()
    def create_entity_universe_aux(generator, datasets):
        for dataset in datasets:
            print("\nProcessing dataset: ", dataset)
            for sample in generator.process(filepath=dataset):
                entity_universe.update(*sample.cand_entities)
                entity_universe.update(sample.ground_truth)

        if args.calculate_stats:
            print("\nSTATISTICS from create_entity_universe, allspans_mode:", generator.is_allspans_mode())
            print(" all_gm_misses: ", generator.all_gm_misses)
            print(" all_gt_misses: ", generator.all_gt_misses)
            print(" all_gm: ", generator.all_gm)
            print(" recall %   : ", (1 - (generator.all_gm_misses+generator.all_gt_misses)/generator.all_gm)*100, " %")
            print(" len(entity_universe):\t", len(entity_universe))

        return

    datasets = [d for d in args.datapath.glob("*.txt")]
    samplesGenerator = SamplesGenerator(args)

  # goldspans
    samplesGenerator.set_goldspans_mode()
    create_entity_universe_aux(samplesGenerator, datasets)
  # allspans
    samplesGenerator.set_allspans_mode()
    create_entity_universe_aux(samplesGenerator, datasets)




  # SAVE entities of our universe to a file in "id\ttitle)" format
    with open(output_folder/"entity_universe.txt", 'w') as fout:
        _, wiki_id_name_map = util.load_wiki_name_id_map(print_stats=False)
        for ent_id in entity_universe:
            fout.write(ent_id + '\t' + wiki_id_name_map[ent_id] + '\n')

    return

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="no_experiment_name", help="under folder data/experiments/")
    # bert casing and size needed in SamplesGenerator, but doesn't change result
    parser.add_argument("--bert_casing", default=config.bert_uncased_string)
    parser.add_argument("--bert_size", default=config.bert_size_base_string)
    #
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

    parser.add_argument("--count_datasets_vocabulary", dest="count_datasets_vocabulary", action="store_true")
    parser.add_argument("--no_count_datasets_vocabulary", dest="count_datasets_vocabulary", action="store_false")
    parser.set_defaults(count_datasets_vocabulary=True)


    return parser.parse_args()


def log_args(args, folder):
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    with open(folder/"create_entity_universe_args.txt", 'w') as f:
        attrs = vars(args)
        f.write('\n'.join("%s: %s" % item for item in attrs.items()))
    with open(folder/"create_entity_universe_args.pickle", "wb") as f:
        pickle.dump(args, f)


def main(args):
    args.datapath = config.base_folder/"data/base_data/new_datasets"
    print(args, '\n')

    output_folder = config.base_folder/"data/experiments"/args.experiment_name
    args_and_logs_folder = output_folder/"args_and_logs"
    if not args_and_logs_folder.exists():
        args_and_logs_folder.mkdir(parents=True, exist_ok=True)

    tee = Tee(args_and_logs_folder/"create_entity_universe_log.txt", 'a')

    if args.count_datasets_vocabulary:
        vocabularyCounter = util.VocabularyCounter()
        vocabularyCounter.count_datasets_vocabulary(args.datapath, args.experiment_name)

    log_args(args, args_and_logs_folder)

    create_entity_universe(args, output_folder)

    tee.close()


if __name__ == "__main__":
    main(_parse_args())

