import os
import argparse
import sys
sys.path.append("../")
from dp.preprocess import preprocess
from dp.train import train

def read_data(path):
    data = []
    with open(path, encoding="utf-8", mode="r") as f:
        for line in f:
            lang, word = line.strip().split()
            data.append((lang, word, word))
    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="train deepphonemizer")

    parser.add_argument("--train-path", type=str, required=True, help="path to the training data")
    parser.add_argument("--valid-path", type=str, required=True, help="path to the validation data")
    parser.add_argument("--config-path", type=str, required=True, help="path to the config file")
    parser.add_argument("--restore-path", type=str, required=None, default=None, help="path to the lastest model file")

    args = parser.parse_args()

    train_data, val_data = read_data(args.train_path), read_data(args.valid_path)
    config_file = args.config_path
    restore_path = args.restore_path

    preprocess(config_file=config_file,
               train_data=train_data,
               val_data=val_data,
               deduplicate_train_data=True)

    train(config_file=config_file, checkpoint_file=restore_path)