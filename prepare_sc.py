import os
import json
import glob
import random

# ================= 設定區 =================
# 您的 Speech Commands 資料夾路徑 (根據截圖，它在當前目錄下)
root_path = "/home/cyc0904/DL/speech_commands_v2" 

# 設定要輸出的 JSON 存放資料夾
output_dir = "./dataset/speech_commands"
os.makedirs(output_dir, exist_ok=True)
# =========================================

def get_files(path):
    # 搜尋所有的 .wav 檔案
    files = glob.glob(os.path.join(path, "*", "*.wav"))
    data = []
    
    # 定義要忽略的資料夾 (背景噪音)
    ignore_folders = ["_background_noise_"]
    
    for f in files:
        # 取得資料夾名稱作為標籤 (例如 "up", "down")
        label = os.path.basename(os.path.dirname(f))
        
        if label in ignore_folders:
            continue
            
        # 轉成絕對路徑，確保訓練時不會找不到
        abs_path = os.path.abspath(f)
        
        # AudioMAE/AST 通常需要的格式
        # 這裡我們建立一個字典，包含路徑和標籤
        entry = {
            "wav": abs_path,
            "labels": label
        }
        data.append(entry)
    return data

print(f"正在掃描資料夾: {root_path} ...")
all_data = get_files(root_path)
print(f"總共找到 {len(all_data)} 個音訊檔案。")

# --- 簡單的資料切分 (80% 訓練, 10% 驗證, 10% 測試) ---
# 為了確保隨機性，先打亂順序
random.seed(42)
random.shuffle(all_data)

n_total = len(all_data)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)

train_data = all_data[:n_train]
val_data = all_data[n_train : n_train + n_val]
test_data = all_data[n_train + n_val:]

print(f"切分結果 -> 訓練集: {len(train_data)}, 驗證集: {len(val_data)}, 測試集: {len(test_data)}")

# --- 寫入 JSON 檔案 ---
def save_json(data_list, filename):
    full_path = os.path.join(output_dir, filename)
    with open(full_path, 'w') as f:
        # 這是 AudioMAE/AST 常見的 JSON 結構: {"data": [...]}
        json.dump({"data": data_list}, f, indent=4)
    print(f"已儲存: {full_path}")

save_json(train_data, "train.json")
save_json(val_data, "val.json")
save_json(test_data, "test.json")

# --- 生成 class_labels_indices.json (標籤對照表) ---
# 這很重要，模型需要知道 "up" 是數字 0 還是 1
unique_labels = sorted(list(set([d['labels'] for d in all_data])))
label_map = []
for idx, label in enumerate(unique_labels):
    label_map.append({
        "index": str(idx),
        "display_name": label,
        "mid": label # 使用 label 作為 ID
    })

save_json(label_map, "class_labels_indices.json")
print("完成！")