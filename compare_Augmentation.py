import argparse
import torch
import lpips
from PIL import Image
from torchvision.transforms import transforms

import os
def compare_1():
    folder='SampledImgs/save_good/final/'
    image1="resizeaachen_000055_000019_leftImg8bit.png"
    image2='0.02aachen_000112_000019_leftImg8bitimg.jpg'
    #0.2aachen_000055_000019_leftImg8bitimg
    #0.02aachen_000112_000019_leftImg8bitimg.jpg
    #

    iA=folder+image1
    iB=folder+image2
    # Load and preprocess images
    img1 = Image.open(iA).convert('RGB')
    img2 = Image.open(iB).convert('RGB')

    # Use the same normalization as in the pretrained models
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)

    # Initialize LPIPS model (using AlexNet as the base model)
    lpips_model = lpips.LPIPS(net='alex')

    # Move images and LPIPS model to the same device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = img1.to(device)
    img2 = img2.to(device)
    lpips_model = lpips_model.to(device)

    # Compute the LPIPS distance between the images
    with torch.no_grad():
        distance = lpips_model(img1, img2).item()

    print(f'LPIPS distance between {image1} and {image2}: {distance:.4f}')

def compare_2():
    p=""
    imageA='SampledImgs/save_good/final/'+p+"resizeaachen_000055_000019_leftImg8bit.png"
    folder_path="data/compare/"
    # Load and preprocess image A
    imgA = Image.open(imageA).convert('RGB')

    # Use the same normalization as in the pretrained models
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    imgA = transform(imgA).unsqueeze(0)

    # Initialize LPIPS model (using AlexNet as the base model)
    lpips_model = lpips.LPIPS(net='alex')

    # Move images and LPIPS model to the same device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgA = imgA.to(device)
    lpips_model = lpips_model.to(device)

    # Compute the LPIPS distance between image A and all images in the folder
    folder = folder_path
    total_distance = 0
    num_images = 0

    with torch.no_grad():
        for file in os.listdir(folder):
            filepath = os.path.join(folder, file)
            if os.path.isfile(filepath) and file.lower().endswith(('png', 'jpg', 'jpeg')):
                img = Image.open(filepath).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                distance = lpips_model(imgA, img).item()
                total_distance += distance
                num_images += 1

    average_distance = total_distance / num_images

    print(
        f'Average LPIPS distance when p={p}: {average_distance:.4f}')


if __name__ == '__main__':
    compare_2()
