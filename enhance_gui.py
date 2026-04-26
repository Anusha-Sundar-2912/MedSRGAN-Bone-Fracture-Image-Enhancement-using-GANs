import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from generator import Generator
from utils import calculate_psnr, calculate_ssim, save_comparison

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import csv

# -------------------- Model Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# -------------------- GUI Setup --------------------
root = tk.Tk()
root.title("🦴 Bone Fracture Image Enhancer - MedSRGAN")
root.geometry("1100x720")
root.resizable(False, False)

bg_color = "#d6eaf8"  # Light blue
fg_color = "#000000"  # Black text

# -------------------- Theme Setup --------------------
def apply_theme(widget):
    try:
        widget.configure(bg=bg_color, fg=fg_color)
    except:
        pass
    for child in widget.winfo_children():
        apply_theme(child)

root.configure(bg=bg_color)

# -------------------- UI Layout --------------------
frame = tk.Frame(root, bg=bg_color)
frame.pack(expand=True, fill="both")

title = tk.Label(frame, text="Bone Fracture Image Enhancement using MedSRGAN", font=("Segoe UI", 16, "bold"))
title.pack(pady=10)

image_frame = tk.Frame(frame, bg=bg_color)
image_frame.pack(pady=10)

left_panel = tk.Label(image_frame, text="Original Noisy Image", font=("Segoe UI", 10))
left_panel.grid(row=0, column=0, padx=30)

right_panel = tk.Label(image_frame, text="Enhanced Image", font=("Segoe UI", 10))
right_panel.grid(row=0, column=1, padx=30)

metrics_label = tk.Label(frame, text="", font=("Segoe UI", 10))
metrics_label.pack(pady=10)

# -------------------- CSV Logging + Animated Popup --------------------
def log_metrics_to_csv(image_name, psnr_val, ssim_val, csv_file="output/enhancement_scores.csv"):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    header = ["Image", "PSNR", "SSIM"]
    write_header = not os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow([image_name, f"{psnr_val:.2f}", f"{ssim_val:.4f}"])

def show_animated_popup(psnr_val, ssim_val):
    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(['PSNR', 'SSIM'], [0, 0], color=['#5DADE2', '#58D68D'])
    ax.set_ylim(0, 50)
    ax.set_ylabel("Score")
    ax.set_title("Animated Enhancement Metrics")

    value_labels = [
        ax.text(bar.get_x() + bar.get_width()/2, 0, '', ha='center', va='bottom', fontsize=11, fontweight='bold')
        for bar in bars
    ]

    def animate(i):
        step = i / 10
        psnr_curr = min(psnr_val, step * psnr_val)
        ssim_curr = min(ssim_val * 50, step * ssim_val * 50)

        bars[0].set_height(psnr_curr)
        bars[1].set_height(ssim_curr)

        value_labels[0].set_text(f"{psnr_curr:.1f}")
        value_labels[0].set_y(psnr_curr + 1)

        value_labels[1].set_text(f"{ssim_curr/50:.2f}")
        value_labels[1].set_y(ssim_curr + 1)

    ani = animation.FuncAnimation(fig, animate, frames=11, repeat=False, interval=100)
    plt.tight_layout()
    plt.show()

# -------------------- Export CSV --------------------
def export_metrics_csv():
    source_path = "output/enhancement_scores.csv"
    if not os.path.exists(source_path):
        messagebox.showwarning("No Data", "No metrics file found to export.")
        return
    dest_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if dest_path:
        with open(source_path, 'r') as src, open(dest_path, 'w') as dst:
            dst.write(src.read())
        messagebox.showinfo("Exported", f"Metrics exported to:\n{dest_path}")

# -------------------- Enhance Image --------------------
def enhance_image():
    global enhanced_global, comparison_global
    global image_tensor_global, enhanced_tensor_global, noisy_tensor_global

    file_path = filedialog.askopenfilename(title="Select a Noisy X-ray Image")
    if not file_path:
        return

    try:
        image = Image.open(file_path).convert("RGB")
        image_resized = image.resize((256, 256))
        image_tensor = transform(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = generator(image_tensor).squeeze(0).cpu().clamp(0, 1)

        output_image = transforms.ToPILImage()(output_tensor)
        enhanced_global = output_image
        image_tensor_global = image_tensor
        enhanced_tensor_global = output_tensor.unsqueeze(0)
        noisy_tensor_global = image_tensor

        psnr_val = calculate_psnr(enhanced_tensor_global, image_tensor)
        ssim_val = calculate_ssim(enhanced_tensor_global, image_tensor)

        metrics_label.config(text=f"📊 PSNR: {psnr_val:.2f} dB    |    SSIM: {ssim_val:.4f}")

        original_img = ImageTk.PhotoImage(image_resized)
        left_panel.configure(image=original_img)
        left_panel.image = original_img

        enhanced_img = ImageTk.PhotoImage(output_image)
        right_panel.configure(image=enhanced_img)
        right_panel.image = enhanced_img

        comparison_global = (noisy_tensor_global, enhanced_tensor_global)

        image_name = os.path.basename(file_path)
        log_metrics_to_csv(image_name, psnr_val, ssim_val)
        show_animated_popup(psnr_val, ssim_val)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to enhance image.\n{str(e)}")

# -------------------- Save Functions --------------------
def save_enhanced():
    if not enhanced_global:
        messagebox.showwarning("No Image", "Please enhance an image first.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if path:
        enhanced_global.save(path)
        messagebox.showinfo("Saved", f"Enhanced image saved at:\n{path}")

def save_comparison_gui():
    if not comparison_global:
        messagebox.showwarning("No Comparison", "Please enhance an image first.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if path:
        noisy, enhanced = comparison_global
        save_comparison(noisy, enhanced, filename=os.path.basename(path))
        messagebox.showinfo("Saved", f"Comparison image saved:\n{path}")

# -------------------- Buttons --------------------
btn_frame = tk.Frame(frame, bg=bg_color)
btn_frame.pack(pady=15)

tk.Button(btn_frame, text="📂 Select Noisy Image", command=enhance_image, width=20).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="💾 Save Enhanced", command=save_enhanced, width=20).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="🖼 Save Comparison", command=save_comparison_gui, width=20).grid(row=0, column=2, padx=10)
tk.Button(btn_frame, text="📥 Export Metrics CSV", command=export_metrics_csv, width=25).grid(row=0, column=3, padx=10)

footer = tk.Label(root, text="© 2025 MedSRGAN - Developed by Anusha", font=("Segoe UI", 8), bg=bg_color)
footer.pack(side="bottom", pady=5)

apply_theme(root)
root.mainloop()
