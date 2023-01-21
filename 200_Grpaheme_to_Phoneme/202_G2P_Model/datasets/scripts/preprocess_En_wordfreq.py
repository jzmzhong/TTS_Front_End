import os
import csv
import re

IN_FILE = "../1_raw/En_wordfreq/unigram_freq.csv"
OUT_FILE_1 = "../2_preprocessed/En_wordfreq/unigram_freq.txt"
OUT_FILE_2 = "../3_train_and_eval_data/En_wordfreq/unigram_freq.txt"

word_freq = {}
with open(IN_FILE) as f:
    csv_reader = csv.reader(f, delimiter=',')
    for i, row in enumerate(csv_reader):
        assert len(row) == 2, i
        word_freq[row[0]] = row[1]

with open(OUT_FILE_1, encoding="utf-8", mode="w") as f:
    with open(OUT_FILE_2, encoding="utf-8", mode="w") as f2:
        for word, freq in word_freq.items():
            f.write(word + "\t" + freq + "\r\n")
            f2.write(word + "\r\n")