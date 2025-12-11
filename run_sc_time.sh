#!/bin/bash

# ================= 設定區 =================
DATA_PATH="./dataset/speech_commands"
MODEL_PATH="./pretrained_models/pretrain_vit_b.pth"

OUTPUT_DIR="./output/speech_commands_time_mask"

BATCH_SIZE=16
EPOCHS=10
BLR=1e-4
CLASS_NUM=35
MASK_RATIO=0.5
# =========================================

mkdir -p $OUTPUT_DIR

python main_finetune_esc.py \
    --dataset speechcommands \
    --model vit_base_patch16 \
    --nb_classes $CLASS_NUM \
    --data_path $DATA_PATH \
    --data_train $DATA_PATH/train.json \
    --data_eval $DATA_PATH/val.json \
    --label_csv $DATA_PATH/class_labels_indices.csv \
    --finetune $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --blr $BLR \
    --layer_decay 0.75 \
    --dist_eval \
    --log_dir $OUTPUT_DIR \
    --mask_type time \
    --mask_ratio $MASK_RATIO
