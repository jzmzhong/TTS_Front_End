import os
import random
import copy

PY_PATH = os.path.dirname(__file__)

INPUT_PATH = os.path.join(PY_PATH, "../3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone.txt")

DIR = os.path.join(PY_PATH, "../3_train_and_eval_data/EnUs_sort_by_freq")
INPUT_PATH_TRAIN = os.path.join(DIR, "EnUs_dict_exclude_polyphone_random_train.txt")
INPUT_PATH_VALID = os.path.join(DIR, "EnUs_dict_exclude_polyphone_random_valid.txt")
OUTPUT_PATH_TRAIN = os.path.join(DIR, "EnUs_dict_exclude_polyphone_sort_by_freq_finetune_0.1_train.txt")
OUTPUT_PATH_VALID = os.path.join(DIR, "EnUs_dict_exclude_polyphone_sort_by_freq_finetune_0.1_valid.txt")


WORD_FREQ_PATH = os.path.join(PY_PATH, "../2_preprocessed/En_wordfreq/unigram_freq.txt")

word_freq = {}
with open(WORD_FREQ_PATH, encoding="utf-8", mode="r") as f:
    for i, line in enumerate(f):
        word, freq = line.strip().split()
        word_freq[word] = int(freq)

lines_freq = {}
lines_train = set()
count_low_freq = 0
with open(INPUT_PATH_TRAIN, encoding="utf-8", mode="r") as f:
    for i, line in enumerate(f):
        word = line.split()[1]
        if word in word_freq:
            freq = word_freq[word]
        # word freq list does not include \' or \-, therefore we extract freq as follows
        elif word.replace("'", "").replace("-", "") in word_freq:
            freq = word_freq[word.replace("'", "").replace("-", "")]
        elif word.endswith("'s") and word[:-2] in word_freq:
            freq = word_freq[word[:-2]]
        else:
            freq = 0
        if freq < 5:
            count_low_freq += 1
        lines_freq[line] = freq
        lines_train.add(line)

lines_valid = set()
with open(INPUT_PATH_VALID, encoding="utf-8", mode="r") as f:
    for i, line in enumerate(f):
        word = line.split()[1]
        if word in word_freq:
            freq = word_freq[word]
        # word freq list does not include \' or \-, therefore we extract freq as follows
        elif word.replace("'", "").replace("-", "") in word_freq:
            freq = word_freq[word.replace("'", "").replace("-", "")]
        elif word.endswith("'s") and word[:-2] in word_freq:
            freq = word_freq[word[:-2]]
        else:
            freq = 0
        if freq < 5:
            count_low_freq += 1
        lines_freq[line] = freq
        lines_valid.add(line)

train_data_finetune = sorted(lines_train, key=lambda x: lines_freq[x], reverse=True)[-int(len(lines_train)/10):]
valid_data_finetune = sorted(lines_valid, key=lambda x: lines_freq[x], reverse=True)[-int(len(lines_valid)/10):]

random.shuffle(train_data_finetune)
random.shuffle(valid_data_finetune)

for (path, data) in zip([OUTPUT_PATH_TRAIN, OUTPUT_PATH_VALID], [train_data_finetune, valid_data_finetune]):
    with open(path, encoding="utf-8", mode="w") as f:
        f.write("".join(data))