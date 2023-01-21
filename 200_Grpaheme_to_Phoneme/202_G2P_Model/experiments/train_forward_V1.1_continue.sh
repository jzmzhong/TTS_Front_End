PORJECT_DIR=`pwd`/..

TRAIN_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_train.txt
VALID_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_valid.txt
CONFIG_PATH=$PORJECT_DIR/dp/configs/forward_config_V1.1.yaml
RESTORE_PATH=$PORJECT_DIR/checkpoints/V1.1_EnUs_forward_6_512_4_8_lr0.0005/latest_model.pt

CUDA_VISIBLE_DEVICES=0 python3 $PORJECT_DIR/scripts/run_training.py --train-path $TRAIN_PATH --valid-path $VALID_PATH --config-path $CONFIG_PATH --restore-path $RESTORE_PATH >> forward_V1.1_continue.log
