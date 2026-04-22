import math
import os
import random
import zipfile

import numpy as np
import librosa


def calculate_metrics(clean_wav, denoised_wav, sr=16000):
    from pesq import pesq
    from pystoi import stoi

    min_len = min(len(clean_wav), len(denoised_wav))
    clean_wav = clean_wav[:min_len]
    denoised_wav = denoised_wav[:min_len]

    try:
        pesq_score = pesq(sr, clean_wav, denoised_wav, 'wb')
    except Exception:
        pesq_score = 1.0

    stoi_score = stoi(clean_wav, denoised_wav, sr, extended=False)

    noise = clean_wav - denoised_wav
    snr_score = 10 * math.log10(np.sum(clean_wav ** 2) / (np.sum(noise ** 2) + 1e-9))

    return pesq_score, stoi_score, snr_score


def get_random_test_samples(voice_zip, noise_zip, temp_dir='/content/temp_test'):
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(voice_zip, 'r') as z:
        min_bytes = 16000 * 2 * 4  # 4초 이상 (16kHz, 16bit)
        valid_infos = [
            info for info in z.infolist()
            if info.filename.endswith('.pcm') and info.file_size >= min_bytes
        ]
        target_info = random.choice(valid_infos)
        z.extract(target_info, temp_dir)
        voice_path = os.path.join(temp_dir, target_info.filename)

    with zipfile.ZipFile(noise_zip, 'r') as z:
        wav_files = [f for f in z.namelist() if f.endswith('.wav')]
        target_wav = random.choice(wav_files)
        z.extract(target_wav, temp_dir)
        noise_path = os.path.join(temp_dir, target_wav)

    return voice_path, noise_path


def load_test_pair(voice_path, noise_path, sample_len=64000, sr=16000):
    with open(voice_path, 'rb') as f:
        clean = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0

    try:
        noise, _ = librosa.load(noise_path, sr=sr, duration=6.0)
        if len(noise) == 0:
            noise = np.zeros(sample_len, dtype=np.float32)
    except Exception:
        noise = np.zeros(sample_len, dtype=np.float32)

    clean = clean[:sample_len]
    if len(noise) < sample_len:
        noise = np.resize(noise, sample_len)
    else:
        noise = noise[:sample_len]

    return clean, clean + noise * 0.5
