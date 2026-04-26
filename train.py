import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from generator import Generator
from discriminator import Discriminator
from feature_extractor import FeatureExtractor
from dataset import GAN_Dataset
from utils import save_comparison, calculate_psnr, calculate_ssim

# Device setup
device = torch.device("cuda")

# Model setup
generator = Generator().to(device)
discriminator = Discriminator().to(device)
feature_extractor = FeatureExtractor().to(device)

# Dataset & Dataloader
dataset = GAN_Dataset("data/original", "data/noisy")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Loss functions
pixel_loss = nn.MSELoss()
adv_loss = nn.BCELoss()
content_loss = nn.MSELoss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)
optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)

# Training loop
epochs = 500
print("🚀 Starting training...\n")

for epoch in range(epochs):
    total_g, total_d, total_psnr, total_ssim = 0, 0, 0, 0

    print(f"\n🔄 Epoch {epoch+1}/{epochs}")

    for i, batch in enumerate(loader):
        print(f"🔁 Training batch {i+1}/{len(loader)}...", end="\r")

        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)

        # Discriminator
        optimizer_d.zero_grad()
        fake = generator(noisy).detach()
        real_out = discriminator(clean)
        fake_out = discriminator(fake)

        real_labels = torch.ones_like(real_out)
        fake_labels = torch.zeros_like(fake_out)

        d_loss = (adv_loss(real_out, real_labels) + adv_loss(fake_out, fake_labels)) / 2
        d_loss.backward()
        optimizer_d.step()

        # Generator
        optimizer_g.zero_grad()
        enhanced = generator(noisy)
        pred_fake = discriminator(enhanced)

        loss_pixel = pixel_loss(enhanced, clean)
        loss_perceptual = content_loss(feature_extractor(enhanced), feature_extractor(clean))
        loss_adversarial = adv_loss(pred_fake, real_labels)

        g_loss = loss_pixel + 0.006 * loss_perceptual + 1e-3 * loss_adversarial
        g_loss.backward()
        optimizer_g.step()

        # Metrics
        total_g += g_loss.item()
        total_d += d_loss.item()
        total_psnr += calculate_psnr(enhanced, clean)
        total_ssim += calculate_ssim(enhanced, clean)

    avg_g = total_g / len(loader)
    avg_d = total_d / len(loader)
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)

    print(f"\n📈 Epoch {epoch+1}/{epochs} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

    # Save output image
    save_comparison(noisy, enhanced, epoch)  # ✅ Correct

# Save model
torch.save(generator.state_dict(), "generator.pth")
print("\n✅ Training complete. Generator saved as 'generator.pth'")
