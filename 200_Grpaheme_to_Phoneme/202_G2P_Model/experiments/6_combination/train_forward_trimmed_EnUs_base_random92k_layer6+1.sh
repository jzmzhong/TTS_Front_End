PORJECT_DIR=`pwd`/../..

TRAIN_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_random_train.txt
VALID_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs_sort_by_freq/EnUs_dict_exclude_polyphone_random_valid.txt
CONFIG_PATH=$PORJECT_DIR/dp/configs/6_combination/forward_trimmed_config_EnUs_base_random92k_layer7_dim384_ffn2_head6.yaml
RESTORE_PATH=$PORJECT_DIR/checkpoints/6_combination/forward_trimmed_EnUs_base_random92k_layer7_dim384_ffn2_head6/latest_model.pt

CUDA_VISIBLE_DEVICES=0 python3 $PORJECT_DIR/scripts/run_training.py --train-path $TRAIN_PATH --valid-path $VALID_PATH --config-path $CONFIG_PATH --restore-path $RESTORE_PATH>> forward_trimmed_EnUs_base_random92k_layer7_dim384_ffn2_head6.log
