import os
import glob
import random
import shutil
import zipfile

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


def manage_disk_and_extract(zip_path, extract_dir):
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)
    print(f"압축 해제 중... ({os.path.basename(zip_path)})")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("압축 해제 완료.")


class RobustDataset(Dataset):
    """
    noise_ext: '.wav' (v3 정제 노이즈) 또는 '.mp3' (v1/v2 원본 노이즈)
    """

    def __init__(self, voice_dir, noise_dir, samples=64000, noise_ext='.wav'):
        all_voices = glob.glob(os.path.join(voice_dir, '**', '*.pcm'), recursive=True)
        self.voice_list = [f for f in all_voices if os.path.getsize(f) >= samples * 2]
        self.noise_list = glob.glob(os.path.join(noise_dir, '**', f'*{noise_ext}'), recursive=True)
        self.samples = samples
        print(f"데이터 로드: 목소리 {len(self.voice_list)}개, 소음 {len(self.noise_list)}개")

    def _load_pcm(self, path):
        try:
            with open(path, 'rb') as f:
                return np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return np.zeros(self.samples, dtype=np.float32)

    def __len__(self):
        return len(self.voice_list)

    def __getitem__(self, idx):
        v_data = self._load_pcm(self.voice_list[idx])
        if len(v_data) < self.samples:
            return torch.zeros(1, self.samples), torch.zeros(1, self.samples)

        start = np.random.randint(0, len(v_data) - self.samples + 1)
        clean = v_data[start:start + self.samples]

        n_path = random.choice(self.noise_list)
        try:
            noise, _ = librosa.load(n_path, sr=16000, duration=6.0)
            if len(noise) == 0:
                noise = np.zeros(self.samples, dtype=np.float32)
        except Exception:
            noise = np.zeros(self.samples, dtype=np.float32)

        if len(noise) < self.samples:
            noise = np.tile(noise, int(np.ceil(self.samples / len(noise))))
        start_n = np.random.randint(0, len(noise) - self.samples + 1)
        noise = noise[start_n:start_n + self.samples]

        snr = np.random.uniform(5, 15)
        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-9)
        noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-9)
        noise = noise * (clean_rms / (10 ** (snr / 20)) / noise_rms)

        return torch.FloatTensor(clean + noise).unsqueeze(0), torch.FloatTensor(clean).unsqueeze(0)
