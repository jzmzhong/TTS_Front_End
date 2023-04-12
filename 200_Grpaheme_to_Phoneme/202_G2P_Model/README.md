# Experiment

## 1. Transformer-LTS Baseline ([DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer))

### 1.1 Data Composition

| language | dictionary | train  | valid  | test   |
| :------- | :--------- | :----- | :----- | :----- |
| EnUs     | [CMU Dict](https://github.com/cmusphinx/cmudict) | 94,001 | 11,750 | 11,750 |


## 2. Grapheme BERT ([GBERT](https://github.com/ldong1111/GraphemeBERT))

### 2.1 Data Composition

| task     | language | source                                                                                                              | train   | valid  | test   | note |
| :------- | :------- | :------------------------------------------------------------------------------------------------------------------ | :------ | :----- | :----- | :--- |
| pretrain | EnUs     | [Google Web Trillion Word Corpus](https://www.kaggle.com/datasets/rtatman/english-word-frequency?resource=download) | 300,000 | 33,333 | 0      | |
| G2P      | EnUs     | [CMU Dict](https://github.com/cmusphinx/cmudict)                                                                    | 10,000  | 11,750 | 11,750 | Valid and test data are the same as taht in experiment 1; train data is random 10k from that in experiment 1.|

### 2.2 Scripts & Configs

* Config: 
    ```bash
    ./dp/configs/2_GBERT/GBERT_config_EnUs_layer6_dim512_ffn4_head8.yaml # pretrain
    ./dp/configs/2_GBERT/GBERT_config_EnUs_layer6_dim384_ffn2_head6.yaml # pretrain
    ```
* Train Script:
    ```bash
    cd ./experiments/2_GBERT
    sh train_GBERT_EnUs_layer6_dim512_ffn4_head8.sh # pretrain
    sh train_GBERT_EnUs_layer6_dim384_ffn2_head6.sh # pretrain
    ```

### 2.3 Results - Pretrain

| Model Name                    | Mask Valid Acc. | Note       |
| :---------------------------- | :-------------- | :--------- |
| EnUs_layer6_dim512_ffn4_head8 | 65.42%          | 935k steps |
| EnUs_layer6_dim384_ffn2_head6 | 62.76%          | 430k steps |

### 2.4 Results - G2P, Train from Scratch vs Finetune from Pretrained GBERT

| Model Name                 | Layers | Dimension | Pretrained LR | Base LR | Valid Acc. | Test Acc. | Note |
| :------------------------- | :----- | :-------- | :------------ | :-----  | :--------- | :-------- | :---- |
| Scratch_10k                | 6+4    | 512       | 5e-5          | 5e-5    | 53.91%     | x%    | 184k steps |
| Finetune_10k               | 6+4    | 512       | 5e-5          | 5e-5    | **55.86%** | x%    | 176k steps |
| Finetune_10k_EncoderLR0.5  | 6+4    | 512       | 2.5e-5        | 5e-5    | 55.11%     | x%    | 158k steps |
| Finetune_10k_EncoderLR0.1  | 6+4    | 512       | 5e-6          | 5e-5    | x%     | x%    | |
<!-- | Scratch_2k                 | 6+4    | 512       |               |         | x%     | x%    | |
| Scratch_5k                 | 6+4    | 512       |               |         | x%     | x%    | | -->

## 3. Move the Char Repeat Operation in Forward Transformer

### 3.1 Data Composition

Same as Experiment 1.

### 3.2 Motivation

In industrial deployment, forward transformer with faster parrellel inference is preferred. However, there is still room for model trimming, especially to deal with the long-sequence input. To repeat the input characters by three times before all encoder layers is unnecessary and would make inference for longer words in particular slow.

### 3.3 Model Architecture

<center>

![Baseline Forward Transformer](./assets/forward_transformer.png)

Baseline Forward Transformer

![Trimmed Forward Transformer](./assets/forward_transformer_trimmed.png)

Trimmed Forward Transformer

</center>

Major Changes:
* Move the char repeat operation from before all encoder layers to just before the last encoder layer.
* Add additional positional encoding after the repeat is done.

### 3.4 Scripts & Configs

* Config: 
    ```bash
    ./dp/configs/3_forward_trimmed/forward_config_EnUs_random106k_layer4_dim512_ffn4_head8.yaml # Small_Baseline
    ./dp/configs/3_forward_trimmed/forward_trimmed_config_EnUs_random106k_layer4_dim512_ffn4_head8.yaml # Small_Trimmed
    ./dp/configs/3_forward_trimmed/forward_config_EnUs_random106k_layer3_dim384_ffn2_head6.yaml # Tiny_Baseline
    ./dp/configs/3_forward_trimmed/forward_trimmed_config_EnUs_random106k_layer3_dim384_ffn2_head6.yaml # Tiny_Trimmed
    ```
* Train Script:
    ```bash
    cd ./experiments/3_forward_trimmed
    sh train_forward_EnUs_random106k_small.sh # Small_Baseline
    sh train_forward_trimmed_EnUs_random106k_small.sh # Small_Trimmed
    sh train_forward_EnUs_random106k_tiny.sh # Tiny_Baseline
    sh train_forward_trimmed_EnUs_random106k_tiny.sh # Tiny_Trimmed
    ```

### 3.5 Results

| Model          | Layers | Dimension | Char Repeat | Valid Acc. | Test Acc.  | Params\*   | Flops(Average)\*  | Flops(Extreme)\* | Note       |
| :------------- | :----- | :-------- | :---------- | :--------- | :--------- | :--------- | :---------------- | :--------------- | :--------- |
| Small_Baseline | 4      | 512       | 3           | **71.53%** | **71.63%** | 8.445M     | 194.210MFlops     | 523.522MFlops    | 180k steps |
| Small_Trimmed  | 4      | 512       | 1 -> 3      | 70.70%     | 71.10%     | 8.446M     | 106.172MFlops     | 185.766MFlops    | 206k steps |
| Tiny_Baseline  | 3      | 384       | 3           | 69.76%     | 70.13%     | **1.806M** | 41.581MFlops      | 112.088MFlops    | 188k steps |
| Tiny_Trimmed   | 3      | 384       | 1 -> 3      | 70.25%     | 70.24%     | **1.807M** | **25.141MFlops**  | **39.773MFlops** | 206k steps |

Notes on \*: The Calculation of Params and Flops:
* The calculation is done by using [thop](https://github.com/Lyken17/pytorch-OpCounter).
* For Flops calculation, in the average scenario, OOVs of 7 characters long are used as inputs;
* in the extreme scenario, OOVs of 20 characters long are used as inputs. 

### 3.6 Conclusions

* By moving the char repeat operation in the forward transformer to just before the last encoder layer, the model is optimized to be faster with comparable word accuracy rate. 
* In average scenario, the small model is theoretically 45.33% faster and the tiny model is theoretically 39.54% faster;
* in extreme scenario, both the small model and the tiny model are theoretically 64.52% faster.
* For word accuracy rate, the small model suffers 0.53% drop and the tiny model gains 0.11% after char repeat optimization, which is within error margins.

### 3.7 Future Work

* Use pretrained GBERT to initialize encoder layers before the char repeat operation to see if word accuracy rate can be increased.
* Use distillation to improve the word accuracy rate of tiny models.

## 4. Finetuning Low-freq Words for OOVs

### 4.1 Motivation

G2P model faces huge disparity between training and inference. In training, we use dictionary words which are often times high-frequency since most dictionaries aim at collecting high-frequency words to improve text coverage rate. In testing, however, most OOVs that would be passed into G2P models to get pronunciations are irregular and low-frequency, such as named entities, loanwords, and even misspelled tokens. This experiment aimed at addressing this issue to improve the efficacy of G2P model in realistic deployment scenarios.

### 4.2 Data Composition

For the 117,501 words in [CMU Dict](https://github.com/cmusphinx/cmudict) that do not have multiple pronunciations, 25,353 words\* are not among the most frequent 1/3 million words ([Google Web Trillion Word Corpus](https://www.kaggle.com/datasets/rtatman/english-word-frequency?resource=download)) and are used for testing. The remaining words that have at least 12,711 occurences in the corpus are used for training and validation.

\* When extracting word frequencies, ' and - are removed since the word frequencies list does not consider these two as valid english alphabets.

| Stage        | Train  | Valid | Test   | Note   |
| :----------- | :----- | :---- | :----- | :----- |
| Pretrain     | 82,934 | 9,214 | 25,353 | Training : Validation = 90% to 10%. |
| Finetune_0.2 | 16,586 | 1,842 | 25,353 | Finetune top 20% frequency pretrain data. |
| Finetune_0.1 | 8,293  | 921   | 25,353 | Finetune top 10% frequency pretrain data. |
| Scratch_0.2  | 16,586 | 1,842 | 25,353 | Exactly same data as finetune_0.2. |

### 4.3 Methodology

Finetune_0.2 and Finetune_0.1 are initialized by Pretrain.

### 4.4 Scripts & Configs

The model architecture and configuration follow the "Tiny_Trimmed" model in Experiment 3.

* Config: 
    ```bash
    ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_pretrain_random92k_layer3_dim384_ffn2_head6.yaml # Pretrain
    ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_finetune_sortbyfreq18k_layer3_dim384_ffn2_head6.yaml # Finetune_0.2
    ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_finetune_sortbyfreq9k_layer3_dim384_ffn2_head6.yaml # Finetune_0.1
    ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_scratch_sortbyfreq18k_layer3_dim384_ffn2_head6.yaml # Scratch_0.2
    ```
* Train Script:
    ```bash
    cd ./experiments/4_finetune_OOV
    sh train_forward_trimmed_EnUs_pretrain_random92k.sh # Pretrain
    sh train_forward_trimmed_EnUs_finetune_sortbyfreq18k.sh # Finetune_0.2
    sh train_forward_trimmed_EnUs_finetune_sortbyfreq9k.sh # Finetune_0.1
    sh train_forward_trimmed_EnUs_scratch_sortbyfreq18k.sh # Scratch_0.2
    ```

### 4.5 Results

| Stage        | Valid Acc. | Valid (20%) Acc. | Valid (10%) Acc.  | Test Acc.     | Note       |
| :----------- | :--------- | :--------------  | :---------------- | :------------ | :--------- |
| Pretrain     | **73.17%** | 66.02%           | 63.63%            | 52.45%        | 126k steps |
| Finetune_0.2 | 71.83%     | **67.81%**       | **65.36%**        | **53.97%**    | 212k steps |
| Finetune_0.1 | 72.30%     | 66.99%           | **65.36%**        | 53.20%        | 134k steps |
| Scratch_0.2  | 53.39%     | 58.90%           | 58.31%            | 46.82%        | 192k steps |

### 4.6 Conclusions

* Finetuning can lead to better word accuracy rate on both low frequency validation data and testing data. Finetuning the lowest 20% frequency words can lead to a slight improvement of word accuracy rate by 1.52% on testing data.
* Catastrophic forgetting has occured as the word accuracy rate on high frequency words has dropped after finetuning.

### 4.7 Future Work

* Change the finetuning data: The finetuning strategy can also be used to address the disparity among datasets with different annotators. Most dictionaries are developed based on purchased / open-source dictionaries by adding new entries. These new entries are more similar to the OOVs that would be passed into G2P models and pronunciations closer to the annotation standard of newly added entries are preferred.
* Mix a small portion of high frequency words in finetuning low frequency words to avoid catastrophic forgetting.

## 5. Explicit Alignment

## 5.1 Hard Alignment for Training Data

### Scripts & Configs
* Data Alignment Mapping:
    ```bash
    ./datasets/alignment_scripts/EnUs_G2P_mappings.txt
    ```
* Data Alignment Script:
    ```bash
    cd ./datasets/alignment_scripts/
    python GP_align_DTW.py
    ```
* Config: 
    ```bash
    ./dp/configs/5_alignment/forward_aligned_config_EnUs_random106k_layer3_dim384_ffn2_head6.yaml
    ```
* Train Script:
    ```bash
    cd ./experiments/5_alignment
    sh train_forward_aligned_EnUs_random106k.sh
    ```

### Results

| Model          | Valid Acc. | Test Acc. | Note   |
| :------------- | :--------- | :-------  | :----- |
| Tiny_Baseline  | 69.76%     | 70.13%    | From Experiment 3 |
| Tiny_Trimmed   | 70.25%     | 70.24%    | From Experiment 3 |
| Tiny_Aligned   | 69.63%     | 70.14%    | |