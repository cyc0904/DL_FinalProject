#（純時間軸遮罩）
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mask_time.py
以「時間軸遮罩」的方式在聲譜圖上做遮罩實驗。
這份程式會：
1. 讀取輸入音檔 (.wav)
2. 計算 magnitude spectrogram
3. 在時間軸上遮掉一段連續的時間區間
4. 將遮罩後的聲譜圖送進模型（目前是 dummy baseline）
5. 畫出 spectrogram（原始、遮罩後、重建後）
6. 用 Griffin-Lim 轉回音訊
"""

import argparse
import math
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf


# ----------------------------------------------------
# 1. 載入音檔
# ----------------------------------------------------
def load_waveform(path, target_sr=16000):
    """
    讀取 wav 檔並轉成單聲道。
    若取樣率與 target_sr 不同，則重取樣。
    """
    waveform, sr = torchaudio.load(path)  # waveform shape = [ch, T]

    # 如果音檔是雙聲道，平均成 mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 如果取樣率不符，做 resample
    if sr != target_sr:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resample(waveform)
        sr = target_sr

    return waveform, sr


# ----------------------------------------------------
# 2. 計算聲譜圖
# ----------------------------------------------------
def compute_spectrogram(waveform, n_fft=1024, hop_length=256):
    """
    計算 magnitude spectrogram（不取 log，方便做 inverse）。
    輸出 shape = [1, F, T]
    """
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,  # 代表 magnitude（不是 power=2）
    )
    spec = spec_transform(waveform)  # [1, F, T]
    return spec


# ----------------------------------------------------
# 3. Griffin-Lim：把 magnitude spectrogram 轉回 waveform
# ----------------------------------------------------
def griffin_lim_reconstruct(mag_spec, n_fft=1024, hop_length=256, length=None):
    """
    使用 Griffin-Lim 從 magnitude 重建波形。
    n_iter=32 已可得到可聽品質（非完美，但足夠 debug）。
    """
    griffin = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        power=1.0,
        n_iter=32,
    )
    wav = griffin(mag_spec.squeeze(0))  # [T]

    # optional：裁切成與原音訊長度一致
    if length is not None and wav.numel() > length:
        wav = wav[:length]

    return wav.unsqueeze(0)  # [1, T]


# ----------------------------------------------------
# 4. 時間軸遮罩（核心）
# ----------------------------------------------------
def time_mask(spec, mask_ratio=0.5):
    """
    在時間軸上遮掉一段連續的區間。

    spec shape: [1, F, T]
    mask_ratio: 遮掉多少時間（0~1）

    回傳：
    masked_spec: [1, F, T] 遮罩後聲譜
    mask:        [1, 1, T] 1=保留, 0=遮掉
    """
    if spec.dim() != 3 or spec.size(0) != 1:
        raise ValueError(f"Expected spectrogram shape [1, F, T], got {tuple(spec.shape)}")

    mask_ratio = float(max(0.0, min(1.0, mask_ratio)))

    _, F, T = spec.shape

    masked_spec = spec.clone()
    mask = torch.ones((1, 1, T), dtype=spec.dtype, device=spec.device)  # 1=有訊號、0=遮罩

    # 要遮掉的 frame 數量
    num_masked_frames = min(T, max(0, int(math.floor(T * mask_ratio))))

    if num_masked_frames == 0:
        return masked_spec, mask

    # 隨機選一個起點
    start_high = max(1, T - num_masked_frames + 1)
    start = torch.randint(0, start_high, (1,), device=spec.device).item()
    end = start + num_masked_frames

    # 在 spec 上設為 0
    masked_spec[:, :, start:end] = 0.0
    mask[:, :, start:end] = 0.0

    return masked_spec, mask


# ----------------------------------------------------
# 5. Placeholder 模型（你之後換成自己的 MAE）
# ----------------------------------------------------
def run_model(masked_spec):
    """
    這是範例：目前只是把 masked_spec 原封不動回傳。

    你要自己接：
    model = AudioMAE(...)
    reconstructed = model(masked_input)
    """
    return masked_spec.clone()


# ----------------------------------------------------
# 6. 繪圖
# ----------------------------------------------------
def plot_spectrogram(spec, title, out_path):
    """
    將 spectrogram 轉成 dB 後存圖。
    """
    spec_db = 20 * torch.log10(spec.clamp(min=1e-8))
    spec_np = spec_db.squeeze(0).cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_np, origin="lower", aspect="auto")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ----------------------------------------------------
# 7. 主程式流程
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True, help="輸入音檔路徑")
    parser.add_argument("--out_dir", type=str, default="outputs_time_mask")
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 讀取音訊
    waveform, sr = load_waveform(args.wav)
    orig_len = waveform.size(-1)

    # 計算聲譜圖
    spec = compute_spectrogram(waveform, n_fft=args.n_fft, hop_length=args.hop_length)

    # 套時間遮罩（核心）
    masked_spec, mask = time_mask(spec, mask_ratio=args.mask_ratio)

    # 這裡換成你的模型推論
    reconstructed_spec = run_model(masked_spec)

    # 產圖
    plot_spectrogram(spec, "Original Spectrogram", os.path.join(args.out_dir, "spec_orig.png"))
    plot_spectrogram(masked_spec, f"Time Masked (ratio={args.mask_ratio})", os.path.join(args.out_dir, "spec_masked.png"))
    plot_spectrogram(reconstructed_spec, "Reconstructed Spectrogram", os.path.join(args.out_dir, "spec_recon.png"))

    # 重建音訊
    recon_waveform = griffin_lim_reconstruct(
        reconstructed_spec,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        length=orig_len,
    )

    # 寫出 3 個 wav
    sf.write(os.path.join(args.out_dir, "orig.wav"), waveform.squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "masked.wav"),
             griffin_lim_reconstruct(masked_spec, args.n_fft, args.hop_length, orig_len).squeeze(0).cpu().numpy(), sr)
    sf.write(os.path.join(args.out_dir, "reconstructed.wav"),
             recon_waveform.squeeze(0).cpu().numpy(), sr)

    print("Done. Results saved in:", args.out_dir)


if __name__ == "__main__":
    main()
