import os
import torch
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def save_comparison(noisy, enhanced, epoch=None, folder='output/comparisons', filename=None):
    """
    Save side-by-side comparison image (noisy vs enhanced).
    Can be used in both training (with epoch) and GUI (with filename).
    """
    os.makedirs(folder, exist_ok=True)

    def norm(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    noisy = norm(noisy)
    enhanced = norm(enhanced)

    # Ensure batch size is handled correctly
    if noisy.dim() == 4 and noisy.size(0) == 1:
        noisy = noisy[0]
    if enhanced.dim() == 4 and enhanced.size(0) == 1:
        enhanced = enhanced[0]

    grid = torch.cat((noisy, enhanced), dim=2)  # side-by-side width-wise

    # Generate save path
    if filename:
        save_path = os.path.join(folder, filename)
    elif epoch is not None:
        save_path = os.path.join(folder, f'epoch_{epoch}_comparison.png')
    else:
        save_path = os.path.join(folder, 'comparison.png')

    save_image(grid, save_path)
    print(f"🖼 Saved: {save_path}")

def calculate_psnr(output, target):
    """
    Calculates PSNR between output and target tensors.
    """
    if output.dim() == 4:
        output = output[0]
        target = target[0]
    output = output.detach().cpu().numpy().transpose(1, 2, 0)
    target = target.detach().cpu().numpy().transpose(1, 2, 0)
    return psnr(target, output, data_range=1.0)

def calculate_ssim(output, target):
    """
    Calculates SSIM between output and target tensors.
    """
    if output.dim() == 4:
        output = output[0]
        target = target[0]
    output = output.detach().cpu().numpy().transpose(1, 2, 0)
    target = target.detach().cpu().numpy().transpose(1, 2, 0)
    return ssim(target, output, data_range=1.0, channel_axis=-1)
