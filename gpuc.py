import re
import matplotlib.pyplot as plt


def parse_training_log(file_path):
    """
    Parse the training log file to extract:
      - epoch number,
      - generator loss (G Loss),
      - discriminator loss (D Loss),
      - PSNR, and
      - SSIM.

    Returns five lists.
    """
    epochs = []
    g_losses = []
    d_losses = []
    psnrs = []
    ssims = []

    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+G Loss:\s+([\d.]+)\s+\|\s+D Loss:\s+([\d.]+)\s+\|\s+PSNR:\s+([\d.]+)\s+\|\s+SSIM:\s+([\d.]+)"
    )

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                g_losses.append(float(match.group(2)))
                d_losses.append(float(match.group(3)))
                psnrs.append(float(match.group(4)))
                ssims.append(float(match.group(5)))

    return epochs, g_losses, d_losses, psnrs, ssims


def plot_metrics(epochs, g_losses, d_losses, psnrs, ssims):
    """
    Create a 2x2 subplot figure that charts:
      - Generator Loss,
      - Discriminator Loss,
      - PSNR, and
      - SSIM across epochs with simple lines.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot Generator Loss
    axs[0, 0].plot(epochs, g_losses, label='G Loss')
    axs[0, 0].set_title("Generator Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("G Loss")
    axs[0, 0].grid(True)

    # Plot Discriminator Loss
    axs[0, 1].plot(epochs, d_losses, color='orange', label='D Loss')
    axs[0, 1].set_title("Discriminator Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("D Loss")
    axs[0, 1].grid(True)

    # Plot PSNR
    axs[1, 0].plot(epochs, psnrs, color='green', label='PSNR')
    axs[1, 0].set_title("PSNR")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("PSNR (dB)")
    axs[1, 0].grid(True)

    # Plot SSIM
    axs[1, 1].plot(epochs, ssims, color='red', label='SSIM')
    axs[1, 1].set_title("SSIM")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("SSIM")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    log_file = "training_log.txt"
    epochs, g_losses, d_losses, psnrs, ssims = parse_training_log(log_file)
    plot_metrics(epochs, g_losses, d_losses, psnrs, ssims)
