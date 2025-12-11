import csv
import os

# 這是 Speech Commands V2 官方標準的 35 個類別 (按字母順序排列)
# 這就是預訓練模型 "腦袋" 裡的順序
OFFICIAL_LABELS = [
    "backward", "bed", "bird", "cat", "dog", 
    "down", "eight", "five", "follow", "forward", 
    "four", "go", "happy", "house", "learn", 
    "left", "marvin", "nine", "no", "off", 
    "on", "one", "right", "seven", "sheila", 
    "six", "stop", "three", "tree", "two", 
    "up", "visual", "wow", "yes", "zero"
]

output_path = "./dataset/speech_commands/class_labels_indices.csv"

print(f"正在建立標準官方 CSV: {output_path}")

with open(output_path, 'w', newline='') as f:
    # 定義 header
    writer = csv.DictWriter(f, fieldnames=["index", "mid", "display_name"])
    writer.writeheader()
    
    for idx, label in enumerate(OFFICIAL_LABELS):
        writer.writerow({
            "index": str(idx),      # 0, 1, 2...
            "mid": label,           # bed, bird...
            "display_name": label   # bed, bird...
        })

print(f"成功！已寫入 {len(OFFICIAL_LABELS)} 個類別。")
print("現在 index 0 是 'backward'，index 1 是 'bed'...")