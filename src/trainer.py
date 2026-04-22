import gc
import json
import os
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .models import UNetMasker, Discriminator, UpgradedUNet, UpgradedDiscriminator
from .audio_utils import get_spec
from .dataset import RobustDataset, manage_disk_and_extract


# ---------------------------------------------------------------------------
# 체크포인트 / 상태 관리
# ---------------------------------------------------------------------------

def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return {'current_zip_idx': 0, 'current_epoch': 0}


def save_state(state_file, zip_idx, epoch):
    with open(state_file, 'w') as f:
        json.dump({'current_zip_idx': zip_idx, 'current_epoch': epoch}, f)
    print(f"상태 저장: [Zip {zip_idx + 1}], [Epoch {epoch}]")


def load_checkpoint(path, device, model, discriminator=None, opt_G=None, opt_D=None):
    checkpoint = torch.load(path, map_location=device)
    if 'generator_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['generator_state_dict'])
        if discriminator and 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if opt_G and 'opt_G_state_dict' in checkpoint:
            opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
        if opt_D and 'opt_D_state_dict' in checkpoint:
            opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if opt_G and 'optimizer_state_dict' in checkpoint:
            opt_G.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"체크포인트 복구 완료: {os.path.basename(path)}")


def _extract_noise_once(config, temp_noise_dir):
    noise_zip = config['noise_zip']
    if not os.path.exists(temp_noise_dir) or not os.listdir(temp_noise_dir):
        os.makedirs(temp_noise_dir, exist_ok=True)
        print(f"소음 압축 해제 중: {os.path.basename(noise_zip)}")
        with zipfile.ZipFile(noise_zip, 'r') as zf:
            zf.extractall(temp_noise_dir)
        print("소음 압축 해제 완료.")


# ---------------------------------------------------------------------------
# v1: UNet only (MSE Loss)
# ---------------------------------------------------------------------------

def train_unet(config):
    """
    config keys:
      voice_zips, noise_zip, checkpoint_dir,
      batch_size, epochs_per_zip, lr, num_workers, noise_ext
    """
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    state_file = os.path.join(config['checkpoint_dir'], 'training_state.json')
    ckpt_path = os.path.join(config['checkpoint_dir'], 'unet_only_model.pth')
    temp_voice_dir = '/content/temp_voice'
    temp_noise_dir = '/content/temp_noise'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetMasker().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
    criterion = nn.MSELoss()

    state = load_state(state_file)
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, device, model, opt_G=optimizer)

    _extract_noise_once(config, temp_noise_dir)

    for zip_idx in range(state['current_zip_idx'], len(config['voice_zips'])):
        manage_disk_and_extract(config['voice_zips'][zip_idx], temp_voice_dir)
        dataset = RobustDataset(temp_voice_dir, temp_noise_dir,
                                noise_ext=config.get('noise_ext', '.wav'))
        loader = DataLoader(dataset, batch_size=config.get('batch_size', 32),
                            shuffle=True, num_workers=config.get('num_workers', 2),
                            pin_memory=True)

        for epoch in range(state['current_epoch'], config.get('epochs_per_zip', 10)):
            model.train()
            total_loss = 0.0
            print(f"[Zip {zip_idx + 1}/{len(config['voice_zips'])}] Epoch {epoch + 1} 시작")

            for i, (noisy, clean) in enumerate(loader):
                noisy, clean = noisy.to(device), clean.to(device)
                noisy_mag, _ = get_spec(noisy)
                clean_mag, _ = get_spec(clean)
                noisy_mag = noisy_mag.unsqueeze(1)
                clean_mag = clean_mag.unsqueeze(1)

                denoised_mag, _ = model(noisy_mag)
                loss = criterion(denoised_mag, clean_mag)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print(f"  Step [{i + 1}/{len(loader)}] Loss: {loss.item():.6f}")

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
            save_state(state_file, zip_idx, epoch + 1)
            print(f"Epoch {epoch + 1} 완료. Avg Loss: {total_loss / len(loader):.6f}")

        state['current_epoch'] = 0
        del dataset, loader
        torch.cuda.empty_cache()
        gc.collect()

    print("UNet only 학습 완료.")


# ---------------------------------------------------------------------------
# v2: UNet + GAN (LSGAN, no AMP)
# ---------------------------------------------------------------------------

def train_unet_gan(config):
    """
    config keys:
      voice_zips, noise_zip, checkpoint_dir,
      batch_size, epochs_per_zip, lr_g, lr_d, lambda_adv, num_workers, noise_ext
    """
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    state_file = os.path.join(config['checkpoint_dir'], 'training_state.json')
    ckpt_path = os.path.join(config['checkpoint_dir'], 'unet_gan_model.pth')
    temp_voice_dir = '/content/temp_voice'
    temp_noise_dir = '/content/temp_noise'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = UNetMasker().to(device)
    discriminator = Discriminator().to(device)
    opt_G = optim.Adam(generator.parameters(), lr=config.get('lr_g', 1e-4))
    opt_D = optim.Adam(discriminator.parameters(), lr=config.get('lr_d', 1e-5))
    criterion_L1 = nn.L1Loss()
    criterion_GAN = nn.MSELoss()
    lambda_adv = config.get('lambda_adv', 0.05)

    state = load_state(state_file)
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, device, generator, discriminator, opt_G, opt_D)

    _extract_noise_once(config, temp_noise_dir)

    for zip_idx in range(state['current_zip_idx'], len(config['voice_zips'])):
        manage_disk_and_extract(config['voice_zips'][zip_idx], temp_voice_dir)
        dataset = RobustDataset(temp_voice_dir, temp_noise_dir,
                                noise_ext=config.get('noise_ext', '.wav'))
        loader = DataLoader(dataset, batch_size=config.get('batch_size', 32),
                            shuffle=True, num_workers=config.get('num_workers', 2),
                            pin_memory=True)

        for epoch in range(state['current_epoch'], config.get('epochs_per_zip', 10)):
            generator.train()
            discriminator.train()
            print(f"[Zip {zip_idx + 1}/{len(config['voice_zips'])}] Epoch {epoch + 1} 시작")

            for i, (noisy, clean) in enumerate(loader):
                noisy, clean = noisy.to(device), clean.to(device)
                noisy_mag, _ = get_spec(noisy)
                clean_mag, _ = get_spec(clean)
                noisy_mag = noisy_mag.unsqueeze(1)
                clean_mag = clean_mag.unsqueeze(1)

                # Discriminator
                opt_D.zero_grad()
                denoised_mag, _ = generator(noisy_mag)
                d_loss = 0.5 * (
                    criterion_GAN(discriminator(clean_mag), torch.ones_like(discriminator(clean_mag))) +
                    criterion_GAN(discriminator(denoised_mag.detach()), torch.zeros_like(discriminator(denoised_mag.detach())))
                )
                d_loss.backward()
                opt_D.step()

                # Generator
                opt_G.zero_grad()
                g_loss = criterion_L1(denoised_mag, clean_mag) + \
                         lambda_adv * criterion_GAN(discriminator(denoised_mag),
                                                    torch.ones_like(discriminator(denoised_mag)))
                g_loss.backward()
                opt_G.step()

                if (i + 1) % 100 == 0:
                    print(f"  Step [{i + 1}/{len(loader)}] D: {d_loss.item():.4f} G: {g_loss.item():.4f}")

            torch.save({'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'opt_G_state_dict': opt_G.state_dict(),
                        'opt_D_state_dict': opt_D.state_dict()}, ckpt_path)
            save_state(state_file, zip_idx, epoch + 1)

        state['current_epoch'] = 0
        del dataset, loader
        torch.cuda.empty_cache()
        gc.collect()

    print("UNet GAN 학습 완료.")


# ---------------------------------------------------------------------------
# v3: Upgraded GAN (LSGAN + AMP) — 최종 모델
# ---------------------------------------------------------------------------

def train_upgraded_gan(config):
    """
    config keys:
      voice_zips, noise_zip, checkpoint_dir,
      batch_size, epochs_per_zip, lr_g, lr_d, lambda_adv, num_workers, noise_ext
    """
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    state_file = os.path.join(config['checkpoint_dir'], 'training_state.json')
    ckpt_path = os.path.join(config['checkpoint_dir'], 'upgraded_gan_model.pth')
    temp_voice_dir = '/content/temp_voice'
    temp_noise_dir = '/content/temp_noise'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    gc.collect()

    generator = UpgradedUNet().to(device)
    discriminator = UpgradedDiscriminator().to(device)
    scaler = torch.amp.GradScaler('cuda')
    opt_G = optim.Adam(generator.parameters(), lr=config.get('lr_g', 1e-4))
    opt_D = optim.Adam(discriminator.parameters(), lr=config.get('lr_d', 1e-5))
    criterion_L1 = nn.L1Loss()
    criterion_GAN = nn.MSELoss()
    lambda_adv = config.get('lambda_adv', 0.01)

    state = load_state(state_file)
    if os.path.exists(ckpt_path):
        load_checkpoint(ckpt_path, device, generator, discriminator, opt_G, opt_D)

    _extract_noise_once(config, temp_noise_dir)

    for zip_idx in range(state['current_zip_idx'], len(config['voice_zips'])):
        manage_disk_and_extract(config['voice_zips'][zip_idx], temp_voice_dir)
        dataset = RobustDataset(temp_voice_dir, temp_noise_dir,
                                noise_ext=config.get('noise_ext', '.wav'))
        loader = DataLoader(dataset, batch_size=config.get('batch_size', 32),
                            shuffle=True, num_workers=config.get('num_workers', 2),
                            pin_memory=True, drop_last=True)

        for epoch in range(state['current_epoch'], config.get('epochs_per_zip', 10)):
            generator.train()
            discriminator.train()
            print(f"[Zip {zip_idx + 1}/{len(config['voice_zips'])}] Epoch {epoch + 1} 시작")

            for i, (noisy, clean) in enumerate(loader):
                noisy = noisy.to(device, non_blocking=True)
                clean = clean.to(device, non_blocking=True)

                with autocast():
                    noisy_mag, _ = get_spec(noisy)
                    clean_mag, _ = get_spec(clean)
                    noisy_mag = noisy_mag.unsqueeze(1)
                    clean_mag = clean_mag.unsqueeze(1)

                    # Discriminator
                    opt_D.zero_grad()
                    denoised_mag, _ = generator(noisy_mag)
                    d_loss = 0.5 * (
                        criterion_GAN(discriminator(clean_mag), torch.ones_like(discriminator(clean_mag))) +
                        criterion_GAN(discriminator(denoised_mag.detach()), torch.zeros_like(discriminator(denoised_mag.detach())))
                    )

                scaler.scale(d_loss).backward()
                scaler.step(opt_D)

                with autocast():
                    # Generator
                    opt_G.zero_grad()
                    g_loss = criterion_L1(denoised_mag, clean_mag) + \
                             lambda_adv * criterion_GAN(discriminator(denoised_mag),
                                                        torch.ones_like(discriminator(denoised_mag)))

                scaler.scale(g_loss).backward()
                scaler.step(opt_G)
                scaler.update()

                if (i + 1) % 100 == 0:
                    print(f"  Step [{i + 1}/{len(loader)}] D: {d_loss.item():.4f} G: {g_loss.item():.4f}")

            torch.save({'generator_state_dict': generator.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'opt_G_state_dict': opt_G.state_dict(),
                        'opt_D_state_dict': opt_D.state_dict()}, ckpt_path)
            save_state(state_file, zip_idx, epoch + 1)

        state['current_epoch'] = 0
        del dataset, loader
        torch.cuda.empty_cache()
        gc.collect()

    print("Upgraded GAN 학습 완료.")
