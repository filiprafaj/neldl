import numpy as np
import argparse

import model.config as config


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="no_experiment_name", help="under folder data/experiments/")

    return parser.parse_args()

def main(args):
    folder = config.base_folder/"data/experiments"/args.experiment_name/"embeddings"
    entity_embeddings = np.loadtxt(folder/"entity_embeddings.txt")
    np.save(folder/"entity_embeddings.npy", entity_embeddings)

if __name__ == "__main__":
    args = _parse_args()
    main(args)
