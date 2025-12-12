import matplotlib.pyplot as plt
import json
import os

# ================= 設定區 =================
# 要比較的實驗（名稱 & log 檔路徑）
log_files = {
    "Baseline":        "./output/speech_commands_finetune/log.txt",
    "Time Mask":       "./output/speech_commands_time_mask/log.txt",
    "Freq Mask":       "./output/speech_commands_freq_mask/log.txt",
    "TimeFreq Block":  "./output/speech_commands_timefreq_mask/log.txt",
    "Patch Mask":      "./output/speech_commands_patch_mask/log.txt", 
}

# 輸出的圖片檔名
output_image = "compare_masks_accuracy.png"
# =========================================


def parse_log(file_path):
    """解析單一 log.txt，回傳 epochs, train_loss, test_acc 陣列"""
    epochs = []
    train_loss = []
    test_acc = []

    if not os.path.exists(file_path):
        print(f"[警告] 找不到檔案: {file_path}")
        return None, None, None

    print(f"[讀取中] {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # AudioMAE log: 每行一個 JSON，裡面會有 epoch / train_loss / test_acc1
                if 'epoch' in data and 'train_loss' in data and 'test_acc1' in data:
                    # epoch 從 0 開始，這裡 +1 讓圖比較好看
                    epochs.append(data['epoch'] + 1)
                    train_loss.append(data['train_loss'])
                    test_acc.append(data['test_acc1'])
            except json.JSONDecodeError:
                # 不是 JSON 的行就略過
                continue

    if len(epochs) == 0:
        print(f"[警告] 檔案 {file_path} 中沒有讀到有效資料")
        return None, None, None

    return epochs, train_loss, test_acc


def plot_compare_accuracy(results_dict):
    """根據多個實驗的結果畫出一張 Accuracy 統整圖"""
    if not results_dict:
        print("沒有任何有效實驗結果可畫圖。")
        return

    plt.figure(figsize=(8, 5))

    for name, (epochs, _, acc) in results_dict.items():
        plt.plot(epochs, acc, marker='o', label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Comparison of Masking Strategies on Test Accuracy")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"[完成] 圖表已儲存為: {output_image}")
    plt.show()


def main():
    results = {}

    for exp_name, path in log_files.items():
        epochs, t_loss, t_acc = parse_log(path)
        if epochs is not None:
            print(f"  -> {exp_name} 讀到 {len(epochs)} 個 epoch，最終準確率: {t_acc[-1]:.2f}%")
            results[exp_name] = (epochs, t_loss, t_acc)
        else:
            print(f"  -> {exp_name} 略過（沒有資料）")

    plot_compare_accuracy(results)


if __name__ == "__main__":
    main()
