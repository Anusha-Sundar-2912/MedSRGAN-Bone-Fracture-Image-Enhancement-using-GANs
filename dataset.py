from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import Compose, ToTensor, Resize

class GAN_Dataset(Dataset):
    def __init__(self, original_dir, noisy_dir, image_size=(256, 256)):
        self.original_dir = original_dir
        self.noisy_dir = noisy_dir

        # Keep only matching files in both folders
        original_files = set(os.listdir(original_dir))
        noisy_files = set(os.listdir(noisy_dir))
        self.common_files = sorted(list(original_files & noisy_files))

        if not self.common_files:
            raise RuntimeError("❌ No matching filenames found in both folders.")

        self.transform = Compose([
            Resize(image_size),
            ToTensor()
        ])

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        filename = self.common_files[idx]

        original_path = os.path.join(self.original_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        try:
            original_img = Image.open(original_path).convert("RGB")
            noisy_img = Image.open(noisy_path).convert("RGB")
        except Exception as e:
            print(f"❌ Failed to load image pair {filename}: {e}")
            from torchvision.transforms import ToTensor
            dummy = ToTensor()(Image.new("RGB", (256, 256)))
            return {'noisy': dummy, 'clean': dummy}

        return {
            'noisy': self.transform(noisy_img),
            'clean': self.transform(original_img)
        }
