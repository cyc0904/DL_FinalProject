#（時間 + 頻率結合遮罩）
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask_timefreq.py
同時在「時間軸」與「頻率軸」做遮罩，形成矩形區塊 (time × freq block)。
此遮罩策略比單純的時間遮罩或頻率遮罩更困難，
也能更全面測試模型對於聲音的時頻結構理解能力。

流程：
1. 讀 WAV
2. 計算 magnitude spectrogram
3. 在 spectrogram 上打 num_blocks 個矩形遮罩
4. 丟進模型（目前 dummy baseline）
5. 產生三張 spectrogram 圖與三個 wav（原始、遮罩後、重建）
"""

import argparse
import math
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf

# 從 mask_time.py 引用公用函式
from mask_time import (
    load_waveform,
    compute_spectrogram,
    griffin_lim_reconstruct,
    plot_spectrogram,
)


# ----------------------------------------------------
# 1. 時頻矩形遮罩（核心）
# ----------------------------------------------------
def time_freq_block_mask(spec,
                         time_block_ratio=0.25,
                         freq_block_ratio=0.25,
                         num_blocks=4):
    """
    在 spectrogram 上隨機打 num_blocks 個 time-freq 矩形遮罩。

    spec shape: [1, F, T]
    time_block_ratio: 單個遮罩在時間軸的比例（如 0.25 = 遮 25% 時間）
    freq_block_ratio: 單個遮罩在頻率軸的比例（如 0.25 = 遮 25% 頻率）
    num_blocks: 打幾個矩形 block

    回傳：
    masked_spec : [1, F, T]
    mask        : [1, F, T]（1=保留，0=遮掉）
    """

    if spec.dim() != 3 or spec.size(0) != 1:
        raise ValueError(f"Expected spectrogram shape [1, F, T], got {tuple(spec.shape)}")

    time_block_ratio = float(max(0.0, min(1.0, time_block_ratio)))
    freq_block_ratio = float(max(0.0, min(1.0, freq_block_ratio)))

    _, F, T = spec.shape
    masked_spec = spec.clone()
    mask = torch.ones_like(spec)

    if num_blocks <= 0 or T == 0 or F == 0 or time_block_ratio == 0.0 or freq_block_ratio == 0.0:
        return masked_spec, mask

    # 計算一個 block 的大小
    time_block = min(T, max(1, int(math.floor(T * time_block_ratio))))
    freq_block = min(F, max(1, int(math.floor(F * freq_block_ratio))))

    max_t_start = max(1, T - time_block + 1)
    max_f_start = max(1, F - freq_block + 1)

    for _ in range(num_blocks):
        # 在可行範圍內隨機選一個起點
        t_start = torch.randint(0, max_t_start, (1,), device=spec.device).item()
        f_start = torch.randint(0, max_f_start, (1,), device=spec.device).item()

        t_end = t_start + time_block
        f_end = f_start + freq_block

        # 將該區塊遮掉（設為 0）
        masked_spec[:, f_start:f_end, t_start:t_end] = 0.0
        mask[:, f_start:f_end, t_start:t_end] = 0.0

    return masked_spec, mask


# ----------------------------------------------------
# 2. Dummy 模型，之後你會替換成 AudioMAE
# ----------------------------------------------------
def run_model(masked_spec):
    """
    TODO: 在此換成真正的 MAE / AudioMAE 推論。
    目前只是 baseline：回傳 masked_spec 本身。
    """
    return masked_spec.clone()


# ----------------------------------------------------
# 3. 主流程
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs_timefreq_mask")
    parser.add_argument("--time_block_ratio", type=float, default=0.25)
    parser.add_argument("--freq_block_ratio", type=float, default=0.25)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: 讀取音訊
    waveform, sr = load_waveform(args.wav)
    orig_len = waveform.size(-1)

    # Step 2: 計算 spectrogram
    spec = compute_spectrogram(waveform, n_fft=args.n_fft, hop_length=args.hop_length)

    # Step 3: 套用 Time+Freq 遮罩
    masked_spec, mask = time_freq_block_mask(
        spec,
        time_block_ratio=args.time_block_ratio,
        freq_block_ratio=args.freq_block_ratio,
        num_blocks=args.num_blocks,
    )

    # Step 4: 模型重建（目前 dummy baseline）
    reconstructed_spec = run_model(masked_spec)

    # Step 5: 視覺化
    plot_spectrogram(spec, "Original Spectrogram", os.path.join(args.out_dir, "spec_orig.png"))
    plot_spectrogram(masked_spec,
                     f"Time+Freq Block Mask (blocks={args.num_blocks})",
                     os.path.join(args.out_dir, "spec_masked.png"))
    plot_spectrogram(reconstructed_spec,
                     "Reconstructed Spectrogram",
                     os.path.join(args.out_dir, "spec_recon.png"))

    # Step 6: Griffin-Lim 重建音訊
    recon_waveform = griffin_lim_reconstruct(
        reconstructed_spec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        length=orig_len,
    )

    # Step 7: 輸出三個 wav
    sf.write(os.path.join(args.out_dir, "orig.wav"), waveform.squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "masked.wav"),
             griffin_lim_reconstruct(masked_spec, args.n_fft, args.hop_length, orig_len).squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "reconstructed.wav"),
             recon_waveform.squeeze(0).cpu().numpy(), sr)

    print("Done. Results saved in:", args.out_dir)


if __name__ == "__main__":
    main()
