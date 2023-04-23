PORJECT_DIR=`pwd`/../..

TRAIN_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_train_10k.txt
VALID_PATH=$PORJECT_DIR/datasets/3_train_and_eval_data/EnUs/EnUs_dict_exclude_polyphone_valid.txt
CONFIG_PATH=$PORJECT_DIR/dp/configs/2_GBERT/autoreg_finetune_config_EnUs_random10k_layer6+4_dim512_ffn4_head8_encoderlr0.5.yaml

CUDA_VISIBLE_DEVICES=0 python3 $PORJECT_DIR/scripts/run_training.py --train-path $TRAIN_PATH --valid-path $VALID_PATH --config-path $CONFIG_PATH >> autoreg_finetune_EnUs_random10k_layer6+4_dim512_ffn4_head8_encoderlr0.5.log