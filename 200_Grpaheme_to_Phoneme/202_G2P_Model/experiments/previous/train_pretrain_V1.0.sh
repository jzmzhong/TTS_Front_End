PORJECT_DIR=`pwd`/..

TRAIN_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/En_wordfreq/unigram_freq_train.txt
VALID_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/En_wordfreq/unigram_freq_valid.txt
CONFIG_PATH=$PORJECT_DIR/dp/configs/pretrain_config_V1.0.yaml
RESTORE_PATH=$PORJECT_DIR/checkpoints/pretrain_V1.0_EnUs_forward_6_512_4_8/model_step_292k.pt

CUDA_VISIBLE_DEVICES=0 python3 $PORJECT_DIR/scripts/run_training_GBERT.py --train-path $TRAIN_PATH --valid-path $VALID_PATH --config-path $CONFIG_PATH --restore-path $RESTORE_PATH >> pretrain_V1.0_continue.log
