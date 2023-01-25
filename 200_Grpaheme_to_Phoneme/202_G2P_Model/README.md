# I. Data

## 1. G2P Data

| language | dictionary | train  | valid  | test   |
| :------- | :--------- | :----- | :----- | :----- |
| EnUs     | [CMU Dict](https://github.com/cmusphinx/cmudict) | 94,001 | 11,750 | 11,750 |

## 2. GBERT Data

| language | source                                                                                                              | train   | valid  | test   |
| :------- | :------------------------------------------------------------------------------------------------------------------ | :------ | :----- | :----- |
| EnUs     | [Google Web Trillion Word Corpus](https://www.kaggle.com/datasets/rtatman/english-word-frequency?resource=download) | 300,000 | 33,333 | 0      |

# II. Experiment

## 1. Adjusting Model Layers and Dimensions

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

| Model Name   | Layers | Dimension | Valid Acc. | Test Acc. |Flops | MACs |
| :----------- | :----- | :-------- | :--------  | :-------  | :---- | :--- |
| autoreg_V1.0 | 4+4    | 512       | 72.45%     | 73.12%    | | |
| autoreg_V1.1 | 3+3    | 512       | 72.77%     | 73.15%    | | |
| autoreg_V1.2 | 2+2    | 512       | 72.35%     | 72.84%    | | |
| autoreg_V1.3 | 4+4    | 384       | 72.55%     | 73.17%    | | |
| autoreg_V1.4 | 3+3    | 384       | 72.60%     | 72.87%    | | |
| autoreg_V1.5 | 2+2    | 384       | 72.21%     | 72.20%    | | |
| autoreg_V1.6 | 4+4    | 256       | 72.16%     | 72.83%    | | |
| autoreg_V1.7 | 3+3    | 256       | 72.20%     | 73.18%    | | |
| autoreg_V1.8 | 2+2    | 256       | 71.92%     | 72.97%    | | |

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

### Pretraining 

| Model Name    | Layers | Dimension | Mask Valid Acc. | Note                                                    |
| :------------ | :----- | :-------- | :-------------  | :------------------------------------------------------ |
| pretrain_V1.0 | 6      | 512       | x%              | Predict entire sequence instead of just the masked part |
| pretrain_V1.1 | 6      | 512       | x%              | Predict only the masked part                            |

### Finetuning