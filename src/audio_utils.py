import numpy as np
import torch


def get_spec(audio):
    window = torch.hann_window(512).to(audio.device)
    spec = torch.stft(
        audio.squeeze(1), n_fft=512, hop_length=160,
        window=window, return_complex=True, center=True,
    )
    return torch.abs(spec), torch.angle(spec)


def spec_to_wav(mag, phase):
    window = torch.hann_window(512).to(mag.device)
    complex_spec = torch.polar(mag, phase)
    return torch.istft(complex_spec, n_fft=512, hop_length=160, window=window, center=True)


def process_overlap_add(audio_np, model, device, sample_rate=16000):
    chunk_size = 64000  # 4초
    hop_size = 32000    # 2초 (50% 겹침)

    if len(audio_np) < chunk_size:
        audio_np = np.pad(audio_np, (0, chunk_size - len(audio_np)), mode='constant')

    num_chunks = int(np.ceil((len(audio_np) - chunk_size) / hop_size)) + 1
    target_len = (num_chunks - 1) * hop_size + chunk_size

    if len(audio_np) < target_len:
        audio_np = np.pad(audio_np, (0, target_len - len(audio_np)), mode='constant')

    output_audio = np.zeros(target_len, dtype=np.float32)
    window_weight = np.zeros(target_len, dtype=np.float32)
    hanning_window = np.hanning(chunk_size)

    with torch.no_grad():
        for i in range(num_chunks):
            start = i * hop_size
            end = start + chunk_size
            chunk_t = torch.FloatTensor(audio_np[start:end]).unsqueeze(0).unsqueeze(0).to(device)
            mag, phase = get_spec(chunk_t)
            denoised_mag, _ = model(mag.unsqueeze(1))
            denoised_wav = spec_to_wav(denoised_mag.squeeze(1), phase).cpu().numpy()[0]
            output_audio[start:end] += denoised_wav * hanning_window
            window_weight[start:end] += hanning_window

    return output_audio / np.where(window_weight > 1e-10, window_weight, 1.0)


def apply_agc(audio_np, target_rms=0.05):
    current_rms = np.sqrt(np.mean(audio_np ** 2) + 1e-9)
    gain = np.clip(target_rms / current_rms, 0.1, 5.0)
    return np.clip(audio_np * gain, -1.0, 1.0)
