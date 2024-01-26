import os
import random
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CityscapesDataset(Dataset):

    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split

        # 读取文件列表
        self.image_files = []
        self.label_files = []
        image_dir = os.path.join(self.root_dir, 'leftImg8bit', self.split)
        label_dir = os.path.join(self.root_dir, 'gtFine', self.split)
        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_label_dir = os.path.join(label_dir, city)
            for file_name in os.listdir(city_image_dir):
                image_file = os.path.join(city_image_dir, file_name)
                label_file = os.path.join(city_label_dir, file_name.replace('_leftImg8bit.png', '_gtFine_labelIds.png'))
                self.image_files.append(image_file)
                self.label_files.append(label_file)
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        transform=transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        transform2=transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ])
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]

        # 读取图像和标注
        image = Image.open(image_file)
        label = Image.open(label_file)

        # 转换为 NumPy 数组
        # image = transform(image)
        # label = transform2(label)
        image = np.array(image)
        label = np.array(label)

        # 将类别 ID 转换为连续的整数，例如 0, 1, 2, ...
        label = label.astype(np.int32)
        label[label == 255] = -1
        classes = np.unique(label)
        for i, c in enumerate(classes):
            if c == -1:
                continue
            label[label == c] = i

        # 转换为 PyTorch 张量，并进行归一化
        image = torch.from_numpy(image).float()
        image /= 255.0

        label = torch.from_numpy(label).long()

        return image, label

def show_img(images,labels):
    image = images[0]
    image = image.detach().cpu().numpy()
    # 将数组转换为 PIL 图像
    image = np.transpose(image, (1, 2, 0)) * 255
    image = Image.fromarray(np.uint8(image))
    # 展示图像
    image.show()

    image = labels[0].detach().cpu().numpy()

    # 将数组转换为 PIL 图像
    image = np.transpose(image, (0, 1)) * 255
    image = Image.fromarray(np.uint8(image), mode='L')

    # 展示图像
    image.show()

if __name__ == '__main__':
    img_size=120
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size,img_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 数据集路径和批量大小
    dataset_dir = 'data/'
    batch_size = 12

    # 创建 DataLoader
    dataset = CityscapesDataset(dataset_dir, split='train')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 遍历数据集
    for i, (images, labels) in enumerate(dataloader):
        show_img(images, labels)
        print(f'Batch {i}, images shape: {images.shape}, labels shape: {labels.shape}')
