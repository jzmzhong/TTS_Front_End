import os
import random

INPUT_PATH = "../3_train_and_eval_data/En_wordfreq/unigram_freq.txt"
OUTPUT_PATH_TRAIN = INPUT_PATH.replace(".txt", "_train.txt")
OUTPUT_PATH_VALID = INPUT_PATH.replace(".txt", "_valid.txt")

with open(INPUT_PATH, encoding="utf-8", mode="r") as f:
    lines = f.readlines()
    random.shuffle(lines)

num = len(lines)
train_data, valid_data = lines[:-int(num/10)], lines[-int(num/10):]

for (path, data) in zip([OUTPUT_PATH_TRAIN, OUTPUT_PATH_VALID], [train_data, valid_data]):
    with open(path, encoding="utf-8", mode="w") as f:
        f.write("".join(data))