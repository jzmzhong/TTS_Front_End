import os
import re

IN_FILE = "../1_raw/EnUs_cmudict/cmudict.dict"
OUT_FILE_1 = "../2_preprocessed/EnUs_cmudict/EnUs_dict.txt"
OUT_FILE_2 = "../3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone.txt"

monophone_dict, polyphone_dict = {}, {}
with open(IN_FILE, encoding="utf-8", mode="r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        # INPUT line format: word(x) phone1 phone2 phone3 ...
        word_iden, phones = line.strip().split()[0], line.strip().split()[1:]
        if "(" in word_iden:
            assert word_iden.count("(") == 1 and word_iden.endswith(")"), i
            word, iden = word_iden.split("(")[0], word_iden.split("(")[1].split(")")[0]
            try:
                iden_int = int(iden)
            except:
                raise Exception(i)
            if not re.fullmatch("^[a-z\'\-]+$", word):
                print("Ignoring:", word)
                continue
            if word in monophone_dict:
                # import pdb; pdb.set_trace()
                # print()
                polyphone_dict[word] = {}
                polyphone_dict[word]["1"] = monophone_dict[word]
                monophone_dict.pop(word)
            polyphone_dict[word][iden] = " ".join(phones).lower()
        else:
            word = word_iden
            if not re.fullmatch("^[a-z\'\-]+$", word):
                print("Ignoring:", word)
                continue
            monophone_dict[word] = " ".join(phones).lower()

with open(OUT_FILE_1, encoding="utf-8", mode="w") as f:
    # OUTPUT_1 line format: word, pos, identifier, phonemes, morphemes, frequency
    for word, phones in monophone_dict.items():
        f.write(",".join([word, "", "", phones, "", ""]) + "\r\n")
    for word, iden_phones in polyphone_dict.items():
        for iden, phones in iden_phones.items():
            f.write(",".join([word, "", iden, phones, "", ""]) + "\r\n")

with open(OUT_FILE_2, encoding="utf-8", mode="w") as f:
    # OUTPUT_2 line format: word phonemes
    for word, phones in monophone_dict.items():
        f.write(" ".join(["EnUs", word, phones]) + "\r\n")
