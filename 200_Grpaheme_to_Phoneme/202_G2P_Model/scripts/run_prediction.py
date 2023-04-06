import os
import sys
sys.path.append("../")
from dp.model.model import load_checkpoint
from dp.model.predictor import Predictor
from tqdm import tqdm 

if __name__ == '__main__':

    # model_names = ["autoreg_V1.0_EnUs_4_512_4_8_lr0.0005", "autoreg_V1.1_EnUs_3_512_4_8_lr0.0005", "autoreg_V1.2_EnUs_2_512_4_8_lr0.0005", 
    #                "autoreg_V1.3_EnUs_4_384_4_6_lr0.0005", "autoreg_V1.4_EnUs_3_384_4_6_lr0.0005", "autoreg_V1.5_EnUs_2_384_4_6_lr0.0005", 
    #                "autoreg_V1.6_EnUs_4_256_4_4_lr0.0005", "autoreg_V1.7_EnUs_3_256_4_4_lr0.0005", "autoreg_V1.8_EnUs_2_256_4_4_lr0.0005"]
    # checkpoints_paths = ["../checkpoints/{}/best_model_no_optim.pt".format(model_name) for model_name in model_names]
    
    model_names = ["baseline", "trimmed"]
    checkpoints_paths = ["../checkpoints/3_forward_trimmed/forward_EnUs_random106k_layer3_dim384_ffn2_head6/model_step_188k.pt", \
                        "../checkpoints/3_forward_trimmed/forward_trimmed_EnUs_random106k_layer3_dim384_ffn2_head6/model_step_206k.pt"]
    
    # test_path = "../datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_test.txt"
    # result_paths = ["../datasets/5_predictions/EnUs/EnUs_dict_exclude_polyphone_test_predictions_{}.txt".format(model_name) for model_name in model_names]
    # log_path = "../datasets/5_predictions/EnUs/EnUs_dict_exclude_polyphone_test_predictions_autoregressive_WER.log"
    
    test_path = "../datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_valid.txt"
    result_paths = ["../datasets/5_predictions/EnUs/EnUs_dict_exclude_polyphone_valid_predictions_{}.txt".format(model_name) for model_name in model_names]
    # log_path = "../datasets/5_predictions/EnUs/EnUs_dict_exclude_polyphone_valid_predictions_autoregressive_WER.log"
    log_path = "../datasets/5_predictions/EnUs/temp.log"
    
    # model_names = ["pretrain", "finetune_0.2", "finetune_0.1", "scratch_0.2"]
    # checkpoints_paths = ["../checkpoints/forward_trimmed_EnUs_pretrain_random92k_layer3_dim384_ffn2_head6/model_step_126k.pt", \
    #                     "../checkpoints/forward_trimmed_EnUs_finetune_sortbyfreq18k_layer3_dim384_ffn2_head6/model_step_212k.pt", \
    #                     "../checkpoints/forward_trimmed_EnUs_finetune_sortbyfreq9k_layer3_dim384_ffn2_head6/model_step_134k.pt", \
    #                     "../checkpoints/forward_trimmed_EnUs_scratch_sortbyfreq18k_layer3_dim384_ffn2_head6/model_step_192k.pt"]
        
    # test_path = "../datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_test.txt"
    # result_paths = ["../datasets/5_predictions/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_test_predictions_{}.txt".format(model_name) for model_name in model_names]

    # test_path = "../datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_finetune_0.1_valid.txt"
    # result_paths = ["../datasets/5_predictions/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_0.1_valid_predictions_{}.txt".format(model_name) for model_name in model_names]
    
    # test_path = "../datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_finetune_0.2_valid.txt"
    # result_paths = ["../datasets/5_predictions/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_0.2_valid_predictions_{}.txt".format(model_name) for model_name in model_names]
    
    # test_path = "../datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_random_valid.txt"
    # result_paths = ["../datasets/5_predictions/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_random_valid_predictions_{}.txt".format(model_name) for model_name in model_names]

    # log_path = "../datasets/5_predictions/EnUs_sort_by_freq/temp.log"
    
    LANG = "EnUs"
    DISPLAY_TOKEN2IDX = False
    DEVICE = "cuda"

    f_log = open(log_path, encoding="utf-8", mode="a")
    f_log.write(test_path + "\r\n")

    for model_name, checkpoint_path, result_path in zip(model_names, checkpoints_paths, result_paths):
        
        f_log.write(model_name + "\r\n")

        model, checkpoint = load_checkpoint(checkpoint_path, device=DEVICE)
        preprocessor = checkpoint["preprocessor"]
        
        f_log.write("Steps: " + str(checkpoint["step"]) + "\r\n")
        result_path = result_path.replace(".txt", "_steps{}.txt".format(str(checkpoint["step"])))

        predictor = Predictor(model=model, preprocessor=preprocessor)
        print("Predictor Loaded!")
    
        if DISPLAY_TOKEN2IDX:
            print(predictor.text_tokenizer.token_to_ids)
            print(predictor.phoneme_tokenizer.token_to_ids)
        print("Token2Index Displayed!")

        test_data = {}
        with open(test_path, encoding="utf-8", mode="r") as f:
            for line in f:
                elements = line.strip().split()
                lang, word, phones = elements[0], elements[1], elements[2:]
                test_data[word] = phones
        print("Test Data Loaded!")

        test_data_results = {}
        count_T, count_F = 0, 0
        predictions = predictor([k for k, v in test_data.items()], lang=LANG, batch_size=1024)
        with open(result_path, encoding="utf-8", mode="w") as f:
            for pred in predictions:
                word, phonemes, phoneme_tokens = pred.word, pred.phonemes, pred.phoneme_tokens
                test_data_results[word] = test_data[word], phonemes
                correct_predict = (" ".join(test_data[word]) == phonemes)
                f.write("\t".join([word, " ".join(test_data[word]), phonemes, str(correct_predict)]) + "\r\n")
                if correct_predict:
                    count_T += 1
                else:
                    count_F += 1
        print("Correct Predictions:", str(count_T))
        print("Wrong Predictions:", str(count_F))
        print("Word Accuracy:", str(100.*count_T/(count_T+count_F))+"%")
        f_log.write("Correct Predictions: " + str(count_T) + "\r\n")
        f_log.write("Wrong Predictions: " + str(count_F) + "\r\n")
        f_log.write("Word Accuracy: " + str(100.*count_T/(count_T+count_F)) + "%\r\n")

f_log.close()