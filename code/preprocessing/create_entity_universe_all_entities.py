import argparse
import os
import pickle
import sys

import model.config as config
from model.util import Tee

import preprocessing.util as util
from preprocessing.samples_generator import SamplesGenerator


def create_entity_universe(output_folder):
    entity_universe = set()
    _, wiki_id_name_map = util.load_wiki_name_id_map()

    incompatible_ent_ids = 0
    with open(config.base_folder/"data/base_data/prob_yago_crosswikis_wikipedia_p_e_m.txt") as fin:
      # READ p_e_m
        print("\nREADING p_e_m")
        for line in fin:
            line = line.rstrip()
            try:
                temp = line.split('\t')
                mention, entities = temp[0],  temp[2:]
                absolute_freq = int(temp[1])

              # COLLECT CANDIDATES
                for e in entities:
                    ent_id, score, _ = map(str.strip, e.split(',', 2))
                    if not ent_id in wiki_id_name_map:
                        incompatible_ent_ids += 1
                    else:
                        entity_universe.add(ent_id)
            except Exception as esd:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print("error in line: ", repr(line))

    print("\nincompatible_ent_ids:\t", incompatible_ent_ids)
    print("\nlen(entity_universe):\t", len(entity_universe))

  # SAVE entities of our universe to a file in "id\ttitle)" format
    with open(output_folder/"entity_universe.txt", 'w') as fout:
        for ent_id in entity_universe:
            fout.write(ent_id + '\t' + wiki_id_name_map[ent_id] + '\n')

    return

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="no_experiment_name", help="under folder data/experiments/")

    return parser.parse_args()


def log_args(args, folder):
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    with open(folder/"create_entity_universe_all_entities_args.txt", 'w') as f:
        attrs = vars(args)
        f.write('\n'.join("%s: %s" % item for item in attrs.items()))
    with open(folder/"create_entity_universe_all_entities_args.pickle", "wb") as f:
        pickle.dump(args, f)


def main(args):
    args.datapath = config.base_folder/"data/base_data/new_datasets"
    print(args, '\n')

    output_folder = config.base_folder/"data/experiments"/args.experiment_name
    args_and_logs_folder = output_folder/"args_and_logs"
    if not args_and_logs_folder.exists():
        args_and_logs_folder.mkdir(parents=True, exist_ok=True)

    tee = Tee(args_and_logs_folder/"create_entity_universe_all_entities_log.txt", 'a')

    if args.count_datasets_vocabulary:
        vocabularyCounter = util.VocabularyCounter()
        vocabularyCounter.count_datasets_vocabulary(args.datapath, args.experiment_name)

    log_args(args, args_and_logs_folder)

    create_entity_universe(args, output_folder)

    tee.close()

if __name__ == "__main__":
    main(_parse_args())

