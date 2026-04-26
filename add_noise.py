import os
import cv2
import numpy as np

original_dir = 'data/original'
noisy_dir = 'data/noisy'
os.makedirs(noisy_dir, exist_ok=True)

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy = cv2.add(image, gauss)
    return noisy

count = 0
for filename in os.listdir(original_dir):
    filepath = os.path.join(original_dir, filename)
    if not os.path.isfile(filepath):
        continue
    img = cv2.imread(filepath)
    if img is None:
        print(f"⚠️ Skipping unreadable: {filename}")
        continue
    noisy_img = add_gaussian_noise(img)  # ✅ FIXED LINE
    save_path = os.path.join(noisy_dir, filename)
    cv2.imwrite(save_path, noisy_img)
    count += 1
    print(f"✅ Saved: {save_path}")

print(f"\n🎉 Done. Total noisy images saved: {count}")
