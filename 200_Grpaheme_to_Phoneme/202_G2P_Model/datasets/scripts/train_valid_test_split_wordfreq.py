import os
import random
import copy

PY_PATH = os.path.dirname(__file__)

INPUT_PATH = os.path.join(PY_PATH, "../3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone.txt")

OUTPUT_DIR = os.path.join(PY_PATH, "../3_train_and_eval_data/EnUs_sort_by_freq")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH_TRAIN = os.path.join(OUTPUT_DIR, "EnUs_dict_exclude_polyphone_random_train.txt")
OUTPUT_PATH_VALID = os.path.join(OUTPUT_DIR, "EnUs_dict_exclude_polyphone_random_valid.txt")
OUTPUT_PATH_TRAIN_2 = os.path.join(OUTPUT_DIR, "EnUs_dict_exclude_polyphone_sort_by_freq_finetune_train.txt")
OUTPUT_PATH_VALID_2 = os.path.join(OUTPUT_DIR, "EnUs_dict_exclude_polyphone_sort_by_freq_finetune_valid.txt")
OUTPUT_PATH_TEST = os.path.join(OUTPUT_DIR, "EnUs_dict_exclude_polyphone_sort_by_freq_test.txt")


WORD_FREQ_PATH = os.path.join(PY_PATH, "../2_preprocessed/En_wordfreq/unigram_freq.txt")

word_freq = {}
with open(WORD_FREQ_PATH, encoding="utf-8", mode="r") as f:
    for i, line in enumerate(f):
        word, freq = line.strip().split()
        word_freq[word] = int(freq)

lines_freq = {}
lines = set()
count_low_freq = 0
with open(INPUT_PATH, encoding="utf-8", mode="r") as f:
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
        lines.add(line)
print("Number of words with <5 freq: " + str(count_low_freq))

lines = sorted(lines, key=lambda x: lines_freq[x], reverse=True)
train_valid_data, test_data = lines[:-int(count_low_freq)], lines[-int(count_low_freq):]
num = len(train_valid_data)

# generate random train/valid set
train_valid_data_random = copy.deepcopy(train_valid_data)
random.shuffle(train_valid_data_random)
# train:valid = 9:1 random
train_data_random, valid_data_random = train_valid_data_random[:-int(num/10)], train_valid_data_random[-int(num/10):]

# extract bottom 20% freq of train/develop set to finetune
train_data_finetune = sorted(train_data_random, key=lambda x: lines_freq[x], reverse=True)
train_data_finetune = train_data_finetune[-int(len(train_data_finetune)/5):]
random.shuffle(train_data_finetune)
valid_data_finetune = sorted(valid_data_random, key=lambda x: lines_freq[x], reverse=True)
valid_data_finetune = valid_data_finetune[-int(len(valid_data_finetune)/5):]
random.shuffle(valid_data_finetune)

for (path, data) in zip([OUTPUT_PATH_TRAIN, OUTPUT_PATH_VALID, OUTPUT_PATH_TRAIN_2, OUTPUT_PATH_VALID_2, OUTPUT_PATH_TEST], \
    [train_data_random, valid_data_random, train_data_finetune, valid_data_finetune, test_data]):
    with open(path, encoding="utf-8", mode="w") as f:
        f.write("".join(data))