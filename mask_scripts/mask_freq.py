#（純頻率軸遮罩）
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask_freq.py
以「頻率軸遮罩」的方式在聲譜圖上做遮罩實驗。

流程：
1. 載入 WAV 音訊
2. 計算 magnitude spectrogram
3. 在「頻率軸」消去某些頻帶
4. 將 masked spectrogram 丟進模型（目前是 dummy baseline）
5. 畫出原始 / 遮罩後 / 重建後聲譜圖
6. Griffin-Lim 重建成音訊
"""

import argparse
import math
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf

# 從 mask_time.py 引用工具函式，以避免重複寫
from mask_time import (
    load_waveform,
    compute_spectrogram,
    griffin_lim_reconstruct,
    plot_spectrogram,
)


# ----------------------------------------------------
# 1. 頻率遮罩（核心策略）
# ----------------------------------------------------
def freq_mask(spec, mask_ratio=0.5):
    """
    在頻率軸遮掉連續的一段頻率區域。

    spec shape: [1, F, T]
    mask_ratio: 遮掉多少比例的頻率（0~1）

    回傳：
    masked_spec: [1, F, T] 遮罩後聲譜圖
    mask:        [1, F, 1] 1 = 保留, 0 = 遮掉
    """

    if spec.dim() != 3 or spec.size(0) != 1:
        raise ValueError(f"Expected spectrogram shape [1, F, T], got {tuple(spec.shape)}")

    mask_ratio = float(max(0.0, min(1.0, mask_ratio)))

    _, F, T = spec.shape  # F = 頻率 bins, T = 時間 frames

    masked_spec = spec.clone()
    mask = torch.ones((1, F, 1), dtype=spec.dtype, device=spec.device)

    # 要遮掉的頻率 bins 數量（例如 50% 的 bins）
    num_masked_bins = min(F, max(0, int(math.floor(F * mask_ratio))))

    # 若比例過低，不遮
    if num_masked_bins == 0:
        return masked_spec, mask

    # 隨機選一段頻率範圍
    start_high = max(1, F - num_masked_bins + 1)
    start = torch.randint(low=0, high=start_high, size=(1,), device=spec.device).item()
    end = start + num_masked_bins

    # 把該頻率範圍全部設為 0（所有時間都消失該頻率成分）
    masked_spec[:, start:end, :] = 0.0
    mask[:, start:end, :] = 0.0

    return masked_spec, mask


# ----------------------------------------------------
# 2. dummy 模型（你會換成 AudioMAE）
# ----------------------------------------------------
def run_model(masked_spec):
    """
    此函式目前只是把 masked_spec 原封不動回傳。
    你將來會換成：

    model = AudioMAE(...)
    reconstructed = model(masked_spec)
    """
    return masked_spec.clone()


# ----------------------------------------------------
# 3. 主程式流程
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="輸入音檔路徑")
    parser.add_argument("--out_dir", type=str, default="outputs_freq_mask")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: 載入音訊
    waveform, sr = load_waveform(args.wav)
    orig_len = waveform.size(-1)

    # Step 2: 計算 magnitude spectrogram
    spec = compute_spectrogram(waveform, n_fft=args.n_fft, hop_length=args.hop_length)

    # Step 3: 套用頻率遮罩（核心）
    masked_spec, mask = freq_mask(spec, mask_ratio=args.mask_ratio)

    # Step 4: 使用模型重建（目前只是 dummy baseline）
    reconstructed_spec = run_model(masked_spec)

    # Step 5: 輸出圖片
    plot_spectrogram(spec, "Original Spectrogram", os.path.join(args.out_dir, "spec_orig.png"))
    plot_spectrogram(masked_spec, f"Freq Masked (ratio={args.mask_ratio})", os.path.join(args.out_dir, "spec_masked.png"))
    plot_spectrogram(reconstructed_spec, "Reconstructed Spectrogram", os.path.join(args.out_dir, "spec_recon.png"))

    # Step 6: Griffin-Lim 重建聲音
    recon_waveform = griffin_lim_reconstruct(
        reconstructed_spec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        length=orig_len,
    )

    sf.write(os.path.join(args.out_dir, "orig.wav"), waveform.squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "masked.wav"),
             griffin_lim_reconstruct(masked_spec, args.n_fft, args.hop_length, orig_len).squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "reconstructed.wav"),
             recon_waveform.squeeze(0).cpu().numpy(), sr)

    print("Done. Results saved in:", args.out_dir)


if __name__ == "__main__":
    main()
