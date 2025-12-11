import matplotlib.pyplot as plt
import json
import os

# ================= 設定區 =================
# 您的 Log 檔案路徑
log_file = "./output/speech_commands_timefreq_mask/log.txt"

# 輸出的圖片檔名
output_image = "training_results.png"
# =========================================

def parse_log(file_path):
    epochs = []
    train_loss = []
    test_acc = []
    
    if not os.path.exists(file_path):
        print(f"錯誤: 找不到檔案 {file_path}")
        return None, None, None

    print(f"正在讀取: {file_path}")
    
    with open(file_path, 'r') as f:
        for line in f:
            # AudioMAE 的 log 每一行通常是一個 JSON 物件
            # 我們嘗試解析每一行
            try:
                data = json.loads(line)
                
                # 確認這一行有我們需要的資料
                if 'epoch' in data and 'train_loss' in data and 'test_acc1' in data:
                    epochs.append(data['epoch'] + 1) # Epoch 從 0 開始，我們加 1 方便閱讀
                    train_loss.append(data['train_loss'])
                    test_acc.append(data['test_acc1'])
            except json.JSONDecodeError:
                # 忽略非 JSON 的行 (例如一般的 print 訊息)
                continue
                
    return epochs, train_loss, test_acc

def plot_graph(epochs, train_loss, test_acc):
    if not epochs:
        print("未讀取到有效的訓練數據，請檢查 log.txt 內容。")
        return

    plt.figure(figsize=(12, 5))

    # 1. 繪製 Loss 曲線 (越低越好)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 2. 繪製 Accuracy 曲線 (越高越好)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_acc, 'r-o', label='Test Accuracy')
    plt.title('Test Accuracy (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"成功！圖表已儲存為: {output_image}")

# 執行
epochs, t_loss, t_acc = parse_log(log_file)
if epochs:
    print(f"共讀取到 {len(epochs)} 個 Epoch 的資料。")
    print(f"最終準確率: {t_acc[-1]:.2f}%")
    plot_graph(epochs, t_loss, t_acc)