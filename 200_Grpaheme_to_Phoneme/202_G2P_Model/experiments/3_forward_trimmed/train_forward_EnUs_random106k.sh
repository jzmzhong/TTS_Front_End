PORJECT_DIR=`pwd`/../..

TRAIN_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_train.txt
VALID_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_valid.txt
CONFIG_PATH=$PORJECT_DIR/dp/configs/3_forward_trimmed/forward_config_EnUs_random106k_layer3_dim384_ffn2_head6.yaml

CUDA_VISIBLE_DEVICES=0 python3 $PORJECT_DIR/scripts/run_training.py --train-path $TRAIN_PATH --valid-path $VALID_PATH --config-path $CONFIG_PATH >> forward_EnUs_random106k_layer3_dim384_ffn2_head6.log
