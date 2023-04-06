# Experiment

## 1. Adjusting Model Layers and Dimensions

### Data Composition

| language | dictionary | train  | valid  | test   |
| :------- | :--------- | :----- | :----- | :----- |
| EnUs     | [CMU Dict](https://github.com/cmusphinx/cmudict) | 94,001 | 11,750 | 11,750 |

### Common Configurations

| TYPE | CONSTANT |
| :--- | :------- |
| Data | EnUs CMU Dict |
| FFN Ratio | 4 |
| Attention Dimension | 64 |
| LR Scheduler | Plateau (factor 0.5, patience 1W steps) |
| Char Repeats (for Forward only) | 3 |

### Autoregressive Transformer

Note: Autoregressive Transformers are trained to replicate works in academia. They usually are not used in commercial system since the autoregressive nature means they are slow in inference.

| Model Name   | Layers | Dimension | Valid Acc. | Test Acc. |Flops | MACs | Note |
| :----------- | :----- | :-------- | :--------  | :-------  | :---- | :--- | :---- |
| autoreg_V1.0 | 4+4    | 512       | 72.45%     | 73.12%    | | | |
| autoreg_V1.1 | 3+3    | 512       | 72.77%     | 73.15%    | | | |
| autoreg_V1.2 | 2+2    | 512       | 72.35%     | 72.84%    | | | |
| autoreg_V1.3 | 4+4    | 384       | 72.55%     | 73.17%    | | | |
| autoreg_V1.4 | 3+3    | 384       | 72.60%     | 72.87%    | | | |
| autoreg_V1.5 | 2+2    | 384       | 72.21%     | 72.20%    | | | |
| autoreg_V1.6 | 4+4    | 256       | 72.16%     | 72.83%    | | | |
| autoreg_V1.7 | 3+3    | 256       | 72.20%     | 73.18%    | | | |
| autoreg_V1.8 | 2+2    | 256       | 71.92%     | 72.97%    | | | |
| autoreg_V1.9 | 6+4    | 512       | 72.67%     | x%    | | | For Comparison with 6+4 with Encoder Pretrained by GBERT|

### Forward Transformer

Note: Forward Transformers are used in industrial deployment since they support parrellel inference.

| Model Name   | Layers | Dimension | Valid Acc. | Test Acc. |Flops | MACs |
| :----------- | :----- | :-------- | :--------  | :-------  | :---- | :--- |
| forward_V1.0 | 6      | 512       | x%     | x%    | | |
| forward_V1.1 | 4      | 512       | x%     | x%    | | |
| forward_V1.2 | 3      | 512       | x%     | x%    | | |
| forward_V1.3 | 6      | 384       | x%     | x%    | | |
| forward_V1.4 | 4      | 384       | x%     | x%    | | |
| forward_V1.5 | 3      | 384       | x%     | x%    | | |

## 2. Grapheme BERT ([GBERT](https://github.com/ldong1111/GraphemeBERT))

### Data Composition

| language | source                                                                                                              | train   | valid  | test   |
| :------- | :------------------------------------------------------------------------------------------------------------------ | :------ | :----- | :----- |
| EnUs     | [Google Web Trillion Word Corpus](https://www.kaggle.com/datasets/rtatman/english-word-frequency?resource=download) | 300,000 | 33,333 | 0      |

### Pretraining 

| Model Name                    | Mask Valid Acc. | Note |
| :---------------------------- | :-------------  | :--- |
| EnUs_layer6_dim512_ffn4_head8 | 64.74%          | |


### Finetuning

| Model Name   | Layers | Dimension | Valid Acc. | Test Acc. |Flops | MACs | Note |
| :----------- | :----- | :-------- | :--------  | :-------  | :---- | :--- | :---- |
| autoreg_V2.0 | 6+4    | 512       | x%     | x%    | | | |


## 3. Move the Char Repeat Operation in Forward Transformer

### Data Composition

Same as Experiment 1.

### Motivation

In industrial deployment, forward transformer with faster parrellel inference is preferred. However, there is still room for model trimming, especially to deal with the long-sequence input. To repeat the input characters by three times before all encoder layers is unnecessary and would make inference for longer words in particular slow.

### Model Architecture

<center>

![Baseline Forward Transformer](./assets/forward_transformer.png)

Baseline Forward Transformer

![Trimmed Forward Transformer](./assets/forward_transformer_trimmed.png)

Trimmed Forward Transformer

</center>

### Scripts & Configs

Baseline:
* Config: 
    ```bash
    ./dp/configs/3_forward_trimmed/forward_config_EnUs_random106k_layer4_dim512_ffn4_head8.yaml
    # ./dp/configs/3_forward_trimmed/forward_config_EnUs_random106k_layer3_dim384_ffn2_head6.yaml
    ```
* Train Script:
    ```bash
    cd ./experiments/3_forward_trimmed
    sh train_forward_EnUs_random106k_large.sh
    # sh train_forward_EnUs_random106k.sh
    ```

Trimmed:
* Config: 
    ```bash
    ./dp/configs/3_forward_trimmed/forward_trimmed_config_EnUs_random106k_layer3_dim384_ffn2_head6.yaml
    ```
* Train Script:
    ```bash
    cd ./experiments/3_forward_trimmed
    sh train_forward_trimmed_EnUs_random106k.sh
    ```

### Results

| Model          | Valid Acc. | Test Acc. | Note   |
| :------------- | :--------- | :-------  | :----- |
| Small_Baseline | 71.32%     | %    | |
| Small_Trimmed  | %     | %    | |
| Tiny_Baseline  | 69.76%     | 70.13%    | |
| Tiny_Trimmed   | 70.21%     | 70.26%    | |

## 4. Finetuning Low-freq Words for OOVs

### Motivation

G2P model faces huge disparity between training and inference. In training, we use dictionary words which are often times high-frequency since most dictionaries aim at collecting high-frequency words to improve text coverage rate. In testing, however, most OOVs that would be passed into G2P models to get pronunciations are irregular and low-frequency, such as named entities, loanwords, and even misspelled tokens. This experiment aimed at addressing this issue to improve the efficacy of G2P model in realistic deployment scenarios.

### Data Composition

For the 117,501 words in [CMU Dict](https://github.com/cmusphinx/cmudict) that do not have multiple pronunciations, 25,353 words\* are not among the most frequent 1/3 million words ([Google Web Trillion Word Corpus](https://www.kaggle.com/datasets/rtatman/english-word-frequency?resource=download)) and are used for testing. The remaining words that have at least 12,711 occurences in the corpus are used for training and validation.

\* When extracting word frequencies, ' and - are removed since the word frequencies list does not consider these two as valid english alphabets.

| Stage        | Train  | Valid | Test   | Note   |
| :----------- | :----- | :---- | :----- | :----- |
| pretrain     | 82,934 | 9,214 | 25,353 | Training : Validation = 90% to 10%. |
| finetune_0.2 | 16,586 | 1,842 | 25,353 | Finetune top 20% frequency pretrain data. |
| finetune_0.1 | 8,293  | 921   | 25,353 | Finetune top 10% frequency pretrain data. |
| scratch_0.2  | 16,586 | 1,842 | 25,353 | Exactly same data as finetune_0.2. |

### Scripts & Configs

Pretrain:
* Config: 
    ```bash
    ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_pretrain_random92k_layer3_dim384_ffn2_head6.yaml
    ```
* Train Script:
    ```bash
    cd ./experiments/4_finetune_OOV
    sh train_forward_trimmed_EnUs_pretrain_random92k.sh
    ```

Finetune:
* Config: 
    ```bash
    ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_finetune_sortbyfreq18k_layer3_dim384_ffn2_head6.yaml
    # ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_finetune_sortbyfreq9k_layer3_dim384_ffn2_head6.yaml
    ```
* Train Script:
    ```bash
    cd ./experiments/4_finetune_OOV
    sh train_forward_trimmed_EnUs_finetune_sortbyfreq18k.sh
    # sh train_forward_trimmed_EnUs_finetune_sortbyfreq9k.sh
    ```

Scratch:
* Config: 
    ```bash
    ./dp/configs/4_finetune_OOV/forward_trimmed_config_EnUs_scratch_sortbyfreq18k_layer3_dim384_ffn2_head6.yaml
    ```
* Train Script:
    ```bash
    cd ./experiments/4_finetune_OOV
    sh train_forward_trimmed_EnUs_scratch_sortbyfreq18k.sh
    ```

### Results

| Stage        | Valid Acc. | Valid (20%) Acc. | Valid (10%) Acc.  | Test Acc.     | Note   |
| :----------- | :--------- | :--------------  | :---------------- | :------------ | :----- |
| pretrain     | **73.23%** | 66.02%           | 63.63%            | 52.42%        | |
| finetune_0.2 | 71.86%     | **67.75%**       | **65.37%**        | **53.92%**    | |
| finetune_0.1 | 70.90%     | 66.18%           | 64.60%            | 53.31%        | |
| scratch_0.2  | 53.43%     | 58.90%           | 58.31%            | 46.81%        | |

### Analysis

### Future Work

The finetuning strategy can also be used to address the disparity among datasets with different annotators. Most dictionaries are developed based on purchased / open-source dictionaries by adding new entries. These new entries are more similar to the OOVs that would be passed into G2P models and pronunciations closer to the annotation standard of newly added entries are preferred.

## 5. Explicit Alignment

