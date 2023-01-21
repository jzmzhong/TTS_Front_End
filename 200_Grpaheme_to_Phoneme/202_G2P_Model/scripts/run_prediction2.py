import os
import sys
sys.path.append("../")
from dp.model.model import load_checkpoint
from dp.model.predictor import Predictor
from tqdm import tqdm 

if __name__ == '__main__':

    checkpoint_path = '../checkpoints/V1.0_EnUs_forward_6_512_2_8/model_step_40k.pt'
    test_path = "../datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_test.txt"
    result_path = "../datasets/5_predictions/EnUs/EnUs_dict_exclude_polyphone_test_predictions_modelV1.0_step40k.txt"
    LANG = "EnUs"
    DISPLAY_TOKEN2IDX = False
    DEVICE = "cpu"
    
    model, checkpoint = load_checkpoint(checkpoint_path, device=DEVICE)
    preprocessor = checkpoint["preprocessor"]
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
    predictions = predictor([k for k, v in test_data.items()], lang=LANG, batch_size=128)
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
