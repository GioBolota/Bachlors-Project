"""Quick test on a single image"""
import sys
sys.path.append("src")

from PIL import Image
import torch
from model import load_checkpoint
from dataset import extract_fft_features, EVAL_TRANSFORM

device = "cpu"
model = load_checkpoint("checkpoints/best_model.pth", device=device)

def test_image(image_path):
    img = Image.open(image_path).convert("RGB")
    
    # High-res version
    img_tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(device)
    fft = torch.tensor(extract_fft_features(img, 256)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_high = model(img_tensor, fft).item()
    
    # CIFAR-10 style (32x32 then upscale back)
    width, height = img.size
    size = min(width, height)
    img_crop = img.crop(((width-size)//2, (height-size)//2, (width-size)//2+size, (height-size)//2+size))
    img_32 = img_crop.resize((32, 32))
    img_224 = img_32.resize((224, 224))
    
    img_tensor = EVAL_TRANSFORM(img_224).unsqueeze(0).to(device)
    fft = torch.tensor(extract_fft_features(img_224, 256)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_cifar = model(img_tensor, fft).item()
    
    print(f"High-Res:      {prob_high:.4f} → {'AI' if prob_high > 0.5 else 'Real'}")
    print(f"CIFAR-10 Style: {prob_cifar:.4f} → {'AI' if prob_cifar > 0.5 else 'Real'}")
    print(f"Difference: {prob_high - prob_cifar:+.4f}")

test_image("img\p (2).jpg") # path to the image which i want to process to cifar-10 style