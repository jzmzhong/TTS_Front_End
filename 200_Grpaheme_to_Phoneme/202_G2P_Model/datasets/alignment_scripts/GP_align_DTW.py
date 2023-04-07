import re
import numpy as np
from tqdm import tqdm
from collections import Counter
# from fastdtw import fastdtw

def get_mapping(path, MAPPING):
    with open(path, encoding="utf-8", mode="r") as f:
        for line in f:
            G, Ps = line.split("#")[0].strip().split(":")
            Ps = [P for P in Ps.strip(";").split(";") if P]
            MAPPING[G] = Ps

# def compute_euclidean_distance_matrix(x, y) -> np.array:
#     """Calculate distance matrix
#     This method calcualtes the pairwise Euclidean distance between two sequences.
#     The sequences can have different lengths.
#     """
#     dist = np.zeros((len(y), len(x)))
#     for i in range(len(y)):
#         for j in range(len(x)):
#             dist[i,j] = (x[j]-y[i])**2
#     return dist

def compute_gp_mismatch_distance_matrix(x, y) -> np.array:
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):

            ## G->3Ps
            if i > 2 and re.sub(r'[0-9]+', '', "_".join(y[i-2:i+1])) in MAPPING[x[j]]:
                dist[i,j] = 1
            elif i < len(y) - 2 and re.sub(r'[0-9]+', '', "_".join(y[i:i+3])) in MAPPING[x[j]]:
                dist[i,j] = 1
            elif i > 1 and i < len(y) - 1 and re.sub(r'[0-9]+', '', "_".join(y[i-1:i+2])) in MAPPING[x[j]]:
                dist[i,j] = 1
            ## G->2Ps
            elif i > 1 and re.sub(r'[0-9]+', '', "_".join(y[i-1:i+1])) in MAPPING[x[j]]:
                dist[i,j] = 2
            elif i < len(y) - 1 and re.sub(r'[0-9]+', '', "_".join(y[i:i+2])) in MAPPING[x[j]]:
                dist[i,j] = 2
            ## G->P
            elif re.sub(r'[0-9]+', '', y[i]) in MAPPING[x[j]]:
                dist[i,j] = 3
            ## G -> nothing
            elif len(MAPPING[x[j]]) == 0:
                dist[i,j] = 0

            # ## G->P
            # if re.sub(r'[0-9]+', '', y[i]) in MAPPING[x[j]]:
            #     dist[i,j] = 1
            # ## G->2Ps
            # elif i > 1 and re.sub(r'[0-9]+', '', "_".join(y[i-1:i+1])) in MAPPING[x[j]]:
            #     dist[i,j] = 2
            # elif i < len(y) - 1 and re.sub(r'[0-9]+', '', "_".join(y[i:i+2])) in MAPPING[x[j]]:
            #     dist[i,j] = 2
            # ## G->3Ps
            # elif i > 2 and re.sub(r'[0-9]+', '', "_".join(y[i-2:i+1])) in MAPPING[x[j]]:
            #     dist[i,j] = 3
            # elif i < len(y) - 2 and re.sub(r'[0-9]+', '', "_".join(y[i:i+3])) in MAPPING[x[j]]:
            #     dist[i,j] = 3
            # elif i > 1 and i < len(y) - 1 and re.sub(r'[0-9]+', '', "_".join(y[i-1:i+2])) in MAPPING[x[j]]:
            #     dist[i,j] = 3

            else:
                if x[j] in "aeiouy":
                    dist[i,j] = 50
                else:
                    dist[i,j] = 100
            
            # dist[i,j] = (x[j]-y[i])**2
    return dist

def compute_accumulated_cost_matrix(x, y, w=3) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    # distances = compute_euclidean_distance_matrix(x, y)
    distances = compute_gp_mismatch_distance_matrix(x, y)

    # Initialization
    cost = np.ones((len(y), len(x))) * 1000. # all inf.
    cost[0,0] = distances[0,0]
    for i in range(1, len(y)):
        for j in range(max(0, i-w), min(len(x), i+w)):
            cost[i, j] = 0
    
    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
        
    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]  

    # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            ) + distances[i, j] 
    
    # Backtracking best possible warp path
    i, j = len(y) - 1, len(x) - 1
    alignment = [(x[j], y[i])]
    while i > 0 or j > 0:
        flag_combine_phone = False
        flag_remove_last_phone = False
        if i == 0:
            j -= 1
            flag_remove_last_phone = True
        elif j == 0:
            i -= 1
            flag_combine_phone = True
        else:
            target = min(cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            )
            ## multiple routes -> decided based on mapping restrictions than based on priority rules
            if re.sub(r'[0-9]+', '', y[i-1]) in MAPPING[x[j-1]] and cost[i-1, j-1] == target:
                i -= 1
                j -= 1
            elif re.sub(r'[0-9]+', '', y[i-1]) in MAPPING[x[j]] and cost[i-1, j] == target:
                i -= 1
                flag_combine_phone = True
            elif re.sub(r'[0-9]+', '', y[i]) in MAPPING[x[j-1]] and cost[i, j-1] == target:
                j -= 1
                flag_remove_last_phone = True
            elif cost[i-1, j-1] == target:
                i -= 1
                j -= 1
            elif cost[i-1, j] == target: # prioritize 1 graph -> many phones over graph -> nothing
                i -= 1
                flag_combine_phone = True
            elif cost[i, j-1] == target:
                j -= 1
                flag_remove_last_phone = True
            else:
                raise Exception("DEBUG NEEDED!")
        if flag_combine_phone:
            alignment[-1] = ((x[j], y[i] + "_" + alignment[-1][-1]))
        elif flag_remove_last_phone:
            alignment[-1] = ((alignment[-1][0], "^"))
            alignment.append((x[j], y[i]))
        else:
            alignment.append((x[j], y[i]))
    alignment.reverse()
    return alignment, cost

if __name__ == '__main__':
    
    # get GP mapping
    MAPPING_PATH = "./EnUs_G2P_mappings.txt"
    global MAPPING
    MAPPING = {}
    get_mapping(MAPPING_PATH, MAPPING)

    # # sample align - for debug purpose
    # # x, y = "extricate", "eh1 k s t r ah0 k ey2 t".split()
    # # x, y = "punctuating", "p ah1 ng k ch uw0 ey2 d ih0 ng".split()
    # # x, y = "telex's", "t eh1 l eh2 k s ih0 z".split()
    # # x, y = "pasch", "p ae1 s k".split()
    # x, y = "schools", "s k uw1 l z".split()
    # dist_a = compute_gp_mismatch_distance_matrix(x, y)
    # alignment_a, cost_a = compute_accumulated_cost_matrix(x, y)
    # print(dist_a)
    # print(alignment_a)
    # print(cost_a)
    
    # get & align training data
    TRAIN_DATA_PATH = "../3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_train.txt"
    TRAIN_ALIGNED_DATA_PATH = TRAIN_DATA_PATH.replace(".txt", "_aligned_DTW.txt")
    VALID_DATA_PATH = "../3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_valid.txt"
    VALID_ALIGNED_DATA_PATH = VALID_DATA_PATH.replace(".txt", "_aligned_DTW.txt")
    ALIGNED_DATA_LOG = TRAIN_DATA_PATH.replace("_train.txt", "_aligned_log.txt")
    
    graphones_set = Counter()
    phonemes_set = Counter()
    unaligned_words = set()
    forced_aligned_words = set()
    
    def process(f, fw):
        for i, line in tqdm(enumerate(f)):
            # if i > 100:
            #     break
            flag_error = False
            elements = line.strip().split()
            lang, word, phones = elements[0], elements[1], elements[2:]
            alignment, cost = compute_accumulated_cost_matrix(word, phones)
            assert len(alignment) == len(word), (word, cost, alignment)
            new_phones = [p for _, p in alignment]
            for _, p in alignment:
                if p.count("_") > 2:
                    # print("ERROR! " + word + " " + " ".join(new_phones))
                    # print(cost)
                    flag_error = True
                    break
            if flag_error:
                unaligned_words.add(word)
                continue
            for g, p in alignment:
                if p != "^" and re.sub(r'[0-9]+', '', p) not in MAPPING[g]:
                    # import pdb; pdb.set_trace()
                    forced_aligned_words.add(word)
                    break
            fw.write(" ".join([lang, word] + new_phones) + "\r\n")
            # fw.write(str(alignment) + "\r\n")
            phonemes_set.update(new_phones)
            graphones_set.update(alignment)
    
    with open(TRAIN_DATA_PATH, encoding="utf-8", mode="r") as f:
        with open(TRAIN_ALIGNED_DATA_PATH, encoding="utf-8", mode="w") as fw:
            process(f, fw)
    with open(VALID_DATA_PATH, encoding="utf-8", mode="r") as f:
        with open(VALID_ALIGNED_DATA_PATH, encoding="utf-8", mode="w") as fw:
            process(f, fw)
    
    with open(ALIGNED_DATA_LOG, encoding="utf-8", mode="w") as f_log:
        f_log.write("Number of words that are unaligned: {}\r\n".format(len(unaligned_words)))
        f_log.write("\r\n".join(unaligned_words) + "\r\n")
        f_log.write("Number of words that are forced aligned: {}\r\n".format(len(forced_aligned_words)))
        f_log.write("\r\n".join(forced_aligned_words) + "\r\n")
        f_log.write("Number of new phones: {}\r\n".format(len(phonemes_set)))
        for phoneme, freq in phonemes_set.most_common():
            f_log.write(phoneme + "\t" + str(freq) + "\r\n")
        f_log.write("List of missing graphones:\r\n")
        for graphone, freq in graphones_set.most_common():
            g, p = graphone
            if re.sub(r'[0-9]+', '', p) in MAPPING[g] or p == "^":
                continue
            f_log.write(":".join(graphone) + "\t" + str(freq) + "\r\n")
