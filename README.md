# 🩺 MedSRGAN: Bone Fracture Image Enhancement using GANs

A deep learning-based framework for enhancing **bone fracture X-ray images** using Generative Adversarial Networks (GANs). This project improves low-quality radiographs by restoring fine anatomical details and enhancing diagnostic clarity.

---

## 🚀 Overview

Medical X-ray images often suffer from:
- Low contrast  
- Noise and artifacts  
- Loss of fine fracture details  

This project proposes **MedSRGAN**, a GAN-based architecture that enhances radiographic images while preserving critical anatomical structures for improved diagnosis.

---

## 🧠 Model Architecture

### 🔷 Framework Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/92ae2c5b-60a9-44d7-b463-76e9682ef20e" width="500"/>
</p>

### 🔹 Key Components

- **Generator (RRDB-based)**  
  Learns to reconstruct high-quality images from degraded inputs 

- **Discriminator**  
  Distinguishes between real and generated images  

- **Feature Extractor (VGG19)**  
  Used for perceptual loss to preserve structural details  

- **Loss Function**
  - Pixel Loss (MSE)
  - Perceptual Loss (VGG19)
  - Adversarial Loss

---

## 📊 Training Performance

<p align="center">
  <img src="https://github.com/user-attachments/assets/28341abf-a839-4246-a044-74d3f221cfb5" width="550"/>
</p>

✔ Stable GAN training  
✔ Increasing PSNR and SSIM over epochs  
✔ Reduced generator & discriminator loss  

---

## 🖼️ Results

### 🔍 Before vs After Enhancement

<p align="center">
  <img src="https://github.com/user-attachments/assets/cce82953-d8ce-43fc-a107-6e82ff044291" width="600"/>
  <img src="https://github.com/user-attachments/assets/2c149a2e-b1d8-49db-b684-98e5b8f78670" width="600"/>
  <img src="https://github.com/user-attachments/assets/ebf84719-f13f-4376-a97e-1365efe7cdde" width="600"/>
</p>

---

✔️ Enhanced fracture visibility  
✔️ Sharper bone edges  
✔️ Reduced noise  

---

## 🧪 Data Augmentation

<img width="1428" height="287" alt="image" src="https://github.com/user-attachments/assets/4e6cf26b-d1bb-4d5d-b13e-fba1f0dc6a56" />

To improve generalization, the dataset is augmented using:
- Gaussian noise  
- Rotation  
- Flipping  
- Brightness & contrast variation  

---

## 📈 Performance Comparison

| Model                  | PSNR (dB) | SSIM  |
|-----------------------|----------|-------|
| Bicubic               | 22.14    | 0.621 |
| ESRGAN                | 25.17    | 0.759 |
| **MedSRGAN**          | **28.49** | **0.871** |

✔ Significant improvement over baseline models  

---

## ⚙️ Tech Stack

- Python  
- PyTorch  
- OpenCV  
- NumPy  
- Matplotlib  

---

## 📁 Project Structure
```bash
MedSRGAN/
│
├── generator.py
├── discriminator.py
├── feature_extractor.py
├── dataset.py
├── add_noise.py
├── train.py
├── main.py
├── enhance_gui.py
├── utils.py
├── gpuc.py
├── requirements.txt
│
└── assets/
├── architecture.png
├── training_metrics.png
├── results.png
├── augmentation.png
```
---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
2️⃣ Train the model
```bash
python train.py
```
3️⃣ Run GUI for inference
```bash
python enhance_gui.py
```
---

## 📌 Key Features
- GAN-based medical image enhancement
- Preserves fine anatomical structures
- Improves diagnostic clarity
- High PSNR & SSIM performance
- Works on noisy and low-resolution X-rays

---
## 📌 Key Features
- Real-time deployment
- Mobile/edge optimization
- 3D medical image enhancement
- Clinical integration

---
