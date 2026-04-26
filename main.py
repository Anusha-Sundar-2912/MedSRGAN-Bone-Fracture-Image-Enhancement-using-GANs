import torch
from generator import Generator
from utils import calculate_psnr, calculate_ssim
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
import os

# Paths
noisy_path = 'data/noisy/spiral (1).jpg'     # <- Replace with your test image
clean_path = 'data/original/spiral (1).jpg'  # <- Matching clean version

# Load model
device = torch.device('cuda')
model = Generator().to(device)
model.load_state_dict(torch.load('generator.pth', map_location=device))
model.eval()

# Preprocessing
transform = Compose([
    Resize((256, 256)),
    ToTensor()
])

noisy_img = Image.open(noisy_path).convert('RGB')
clean_img = Image.open(clean_path).convert('RGB')

noisy_tensor = transform(noisy_img).unsqueeze(0).to(device)
clean_tensor = transform(clean_img).unsqueeze(0).to(device)

# Enhance
with torch.no_grad():
    enhanced_tensor = model(noisy_tensor)

# Normalize for display
def norm(t):
    return (t - t.min()) / (t.max() - t.min() + 1e-8)

# Metrics
psnr_score = calculate_psnr(enhanced_tensor, clean_tensor)
ssim_score = calculate_ssim(enhanced_tensor, clean_tensor)

print(f"\n📊 PSNR: {psnr_score:.2f} | SSIM: {ssim_score:.4f}")

# Display side-by-side
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(norm(noisy_tensor.squeeze()).permute(1, 2, 0).cpu())
axs[0].set_title("Noisy Input")
axs[1].imshow(norm(enhanced_tensor.squeeze()).permute(1, 2, 0).cpu())
axs[1].set_title("Enhanced Output")
axs[2].imshow(norm(clean_tensor.squeeze()).permute(1, 2, 0).cpu())
axs[2].set_title("Original Clean Image")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
