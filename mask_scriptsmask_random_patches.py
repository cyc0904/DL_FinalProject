#（隨機 patch 遮罩，類似 MAE）
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask_random_patches.py
模仿 MAE-style 遮罩：將 Spectrogram 切成 patches，然後隨機遮掉一部分。

流程：
1. 讀 WAV
2. 算 Spectrogram
3. 切成 patch (patch_freq x patch_time)
4. 隨機選擇大量 patches 遮掉（通常 75%）
5. 模型重建（目前 dummy baseline）
6. 視覺化 + Griffin-Lim 重建音訊
"""

import argparse
import math
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf

from mask_time import (
    load_waveform,
    compute_spectrogram,
    griffin_lim_reconstruct,
    plot_spectrogram,
)


# ----------------------------------------------------
# 1. MAE style patch masking（核心）
# ----------------------------------------------------
def random_patch_mask(spec,
                      patch_freq=16,
                      patch_time=16,
                      mask_ratio=0.75):
    """
    spec shape = [1, F, T]

    patch_freq : patch 的高度（freq 方向）
    patch_time : patch 的寬度（time 方向）
    mask_ratio : 遮掉多少比例的 patch（例如 0.75）

    回傳：
    masked_spec
    patch_mask_map: [num_patches]，1=保留、0=遮罩
    meta: 用來記錄 padding（後續可用來還原）
    """

    if spec.dim() != 3 or spec.size(0) != 1:
        raise ValueError(f"Expected spectrogram shape [1, F, T], got {tuple(spec.shape)}")
    if patch_freq <= 0 or patch_time <= 0:
        raise ValueError("patch_freq and patch_time must be positive integers")

    mask_ratio = float(max(0.0, min(1.0, mask_ratio)))

    _, F, T = spec.shape
    masked_spec = spec.clone()

    # -------- Padding step：確保 F 與 T 可整除 patch size --------
    F_pad = (patch_freq - F % patch_freq) % patch_freq
    T_pad = (patch_time - T % patch_time) % patch_time

    if F_pad > 0 or T_pad > 0:
        masked_spec = torch.nn.functional.pad(
            masked_spec,
            (0, T_pad, 0, F_pad),  # (left, right, top, bottom)
            mode="constant",
            value=0.0,
        )

    _, F2, T2 = masked_spec.shape

    # 計算 patch grid 尺寸
    num_f = F2 // patch_freq
    num_t = T2 // patch_time
    num_patches = num_f * num_t

    # --------------------------
    # 取一個 random permutation
    # --------------------------
    num_mask = min(num_patches, max(0, int(math.floor(num_patches * mask_ratio))))     # 你要遮掉幾個 patch
    patch_indices = torch.randperm(num_patches, device=spec.device)  # random order
    mask_ids = patch_indices[:num_mask].tolist()          # 前 num_mask 個 ID

    # 建立 patch mask map（用於之後 reconstruct）
    patch_mask_map = torch.ones(num_patches, dtype=spec.dtype, device=spec.device)

    if num_mask == 0:
        return masked_spec[:, :F, :T], patch_mask_map, (F_pad, T_pad, F2, T2, num_f, num_t)

    # --------------------------
    # 遮掉指定的 patches
    # --------------------------
    for pid in mask_ids:
        f_idx = pid // num_t   # patch 在 freq grid 的 index
        t_idx = pid % num_t    # patch 在 time grid 的 index

        f_start = f_idx * patch_freq
        f_end   = f_start + patch_freq
        t_start = t_idx * patch_time
        t_end   = t_start + patch_time

        masked_spec[:, f_start:f_end, t_start:t_end] = 0.0
        patch_mask_map[pid] = 0.0

    # 去掉 padding，讓 output 與原本 spec 對齊
    masked_spec = masked_spec[:, :F, :T]

    return masked_spec, patch_mask_map, (F_pad, T_pad, F2, T2, num_f, num_t)


# ----------------------------------------------------
# 2. Dummy 模型（你會換成 MAE）
# ----------------------------------------------------
def run_model(masked_spec):
    """
    TODO: 這裡接你自己的 MAE / AudioMAE 推論邏輯。
    現在只是 baseline（什麼都不做，直接回傳原樣）。
    """
    return masked_spec.clone()


# ----------------------------------------------------
# 3. 主程式流程
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs_random_patch_mask")
    parser.add_argument("--patch_freq", type=int, default=16)
    parser.add_argument("--patch_time", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Load audio
    waveform, sr = load_waveform(args.wav)
    orig_len = waveform.size(-1)

    # Step 2: Spectrogram
    spec = compute_spectrogram(waveform, n_fft=args.n_fft, hop_length=args.hop_length)

    # Step 3: 隨機 patch 遮罩（核心）
    masked_spec, patch_map, meta = random_patch_mask(
        spec,
        patch_freq=args.patch_freq,
        patch_time=args.patch_time,
        mask_ratio=args.mask_ratio,
    )

    # Step 4: 模型重建（目前 dummy）
    reconstructed_spec = run_model(masked_spec)

    # Step 5: 視覺化
    plot_spectrogram(spec, "Original Spectrogram", os.path.join(args.out_dir, "spec_orig.png"))
    plot_spectrogram(masked_spec,
                     f"Random Patch Mask (ratio={args.mask_ratio})",
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

    # Step 7: 寫出音檔
    sf.write(os.path.join(args.out_dir, "orig.wav"),
             waveform.squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "masked.wav"),
             griffin_lim_reconstruct(masked_spec, args.n_fft, args.hop_length, orig_len)
             .squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "reconstructed.wav"),
             recon_waveform.squeeze(0).cpu().numpy(), sr)

    print("Done. Results saved in:", args.out_dir)


if __name__ == "__main__":
    main()
