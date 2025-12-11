import json
import csv
import os

# 設定路徑
json_path = "./dataset/speech_commands/class_labels_indices.json"
csv_path = "./dataset/speech_commands/class_labels_indices.csv"

print(f"正在讀取: {json_path}")

with open(json_path, 'r') as f:
    content = json.load(f)

# 根據我們之前的 prepare_sc.py，內容被包在 "data" 裡面
if "data" in content:
    data_list = content["data"]
else:
    data_list = content # 防呆機制

print(f"讀取到 {len(data_list)} 個類別，正在轉換為 CSV...")

# 寫入 CSV
# AudioMAE 需要的欄位通常是: index, mid, display_name
headers = ["index", "mid", "display_name"]

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    
    for item in data_list:
        # 確保寫入的字典只包含我們需要的 headers
        row = {
            "index": item["index"],
            "mid": item["mid"],
            "display_name": item["display_name"]
        }
        writer.writerow(row)

print(f"成功建立 CSV 檔案: {csv_path}")