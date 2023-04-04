PORJECT_DIR=`pwd`/..

TRAIN_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/En_wordfreq/unigram_freq_train.txt
VALID_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/En_wordfreq/unigram_freq_valid.txt
CONFIG_PATH=$PORJECT_DIR/dp/configs/GBERT_config_EnUs_layer6_dim512_ffn4_head8.yaml
RESTORE_PATH=$PORJECT_DIR/checkpoints/GBERT_EnUs_layer6_dim512_ffn4_head8/latest_model.pt

CUDA_VISIBLE_DEVICES=0 python3 $PORJECT_DIR/scripts/run_training_GBERT.py --train-path $TRAIN_PATH --valid-path $VALID_PATH --config-path $CONFIG_PATH --restore-path $RESTORE_PATH
