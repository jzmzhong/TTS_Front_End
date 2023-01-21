import os
import random

INPUT_PATH = "../3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone.txt"
OUTPUT_PATH_TRAIN = INPUT_PATH.replace(".txt", "_train.txt")
OUTPUT_PATH_VALID = INPUT_PATH.replace(".txt", "_valid.txt")
OUTPUT_PATH_TEST = INPUT_PATH.replace(".txt", "_test.txt")

with open(INPUT_PATH, encoding="utf-8", mode="r") as f:
    lines = f.readlines()
    random.shuffle(lines)

num = len(lines)
train_data, valid_data, test_data = lines[:-int(num/10*2)], lines[-int(num/10*2):-int(num/10)], lines[-int(num/10):]

for (path, data) in zip([OUTPUT_PATH_TRAIN, OUTPUT_PATH_VALID, OUTPUT_PATH_TEST], [train_data, valid_data, test_data]):
    with open(path, encoding="utf-8", mode="w") as f:
        f.write("".join(data))