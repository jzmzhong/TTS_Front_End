PORJECT_DIR=`pwd`/..

TRAIN_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_finetune_train.txt
VALID_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_sort_by_freq_finetune_valid.txt
CONFIG_PATH=$PORJECT_DIR/dp/configs/forward_trimmed_config_EnUs_sortbyfreq18k.yaml
RESTORE_PATH=$PORJECT_DIR/checkpoints/forward_trimmed_EnUs_random92k/xxx.pt

CUDA_VISIBLE_DEVICES=0 python3 $PORJECT_DIR/scripts/run_training.py --train-path $TRAIN_PATH --valid-path $VALID_PATH --config-path $CONFIG_PATH --restore_path $RESTORE_PATH >> forward_trimmed_EnUs_sortbyfreq18k.log
