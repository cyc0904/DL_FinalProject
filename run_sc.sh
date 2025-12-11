#!/bin/bash

# ================= 設定區 =================
# JSON 資料夾路徑
DATA_PATH="./dataset/speech_commands"

# 模型路徑 (請確認檔名正確)
MODEL_PATH="./pretrained_models/pretrain_vit_b.pth"

# 輸出路徑
OUTPUT_DIR="./output/speech_commands_finetune"

# 參數設定
BATCH_SIZE=16 #16
EPOCHS=10    #30
BLR=1e-4
CLASS_NUM=35
# =========================================

mkdir -p $OUTPUT_DIR

# 執行訓練
# 注意：這裡我們明確指定了 data_train, data_eval 和 label_csv 的位置
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
    --log_dir $OUTPUT_DIR