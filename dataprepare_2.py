import torch
import torchvision.transforms as transforms
from torchvision.datasets import Cityscapes
import numpy as np
import torchvision
from PIL import Image
import  matplotlib.pyplot as plt
def data_loader():
    # 设置数据集的路径
    cityscapes_root = 'data'

    # 定义预处理变换
    transform = transforms.Compose([
        transforms.Resize((120, 120)), # 可以调整为您需要的尺寸
        transforms.ToTensor()
    ])

    # 加载Cityscapes训练集
    train_dataset = Cityscapes(
        root=cityscapes_root,
        split='train',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=transform
    )

    # 加载Cityscapes验证集
    val_dataset = Cityscapes(
        root=cityscapes_root,
        split='val',
        mode='fine',
        target_type='semantic',
        transform=transform,
        target_transform=transform
    )

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    return train_loader
def s2(images,labels):
    image = images[0]
    image = image.detach().cpu().numpy()
    # 将数组转换为 PIL 图像
    image = np.transpose(image, (1, 2, 0)) * 255
    image = Image.fromarray(np.uint8(image))
    # 展示图像
    image.show()

    image = labels[0].detach().cpu().numpy()

    # 将数组转换为 PIL 图像
    image = np.transpose(image, (1, 2, 0)) * 255
    #image = Image.fromarray(np.uint8(image))
    # 展示图像
    image.show()

def show_img(images,labels):
    img_data=images
    #mask_data = labels.repeat(1, 3, 1, 1)
    #img_data = img_data * 0.5 + 0.5  # [0 ~ 1]
    #mask_data=torch.round(mask_data*255)
    #show_data = torch.cat((img_data, mask_data), dim=2)
    plt.rcParams['figure.dpi'] = 100
    plt.grid(False)
    plt.imshow(
        torchvision.utils.make_grid(img_data).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0),
        cmap=plt.cm.binary)
    plt.show()
    plt.pause(0.0001)
# 使用数据
def label_embedding(batch_size,img_size,mask,label_w2v):
    mask_i_o=mask*255
    mask_i=mask_i_o.view(-1)
    w2v_list = torch.tensor(list(label_w2v.values()))
    new_mask=torch.rand(mask_i.shape[0], w2v_list.shape[1])
    # print(torch.unique(mask_i))
    for i in list(range(int(mask_i.shape[0]))):
        new_mask[i]=w2v_list[int(mask_i[i])]
    new_mask=new_mask.view(batch_size,img_size, img_size, w2v_list.shape[1]).permute(0,3,1,2).clone()
    return new_mask

i=0
value_list=[0.0, 0.0039, 0.0078, 0.0118, 0.0157, 0.0196,
            0.0235, 0.0275, 0.0314, 0.0353, 0.0392, 0.0431,
            0.0471, 0.051, 0.0549, 0.0588, 0.0627, 0.0667,
            0.0706, 0.0745, 0.0784, 0.0824, 0.0863, 0.0902,
            0.0941, 0.098, 0.102, 0.1059, 0.1098, 0.1137,
            0.1176, 0.1216, 0.1255]

train_loader=data_loader()
label_w2v = torch.load('label_w2v_dict.pth')
for images, masks in train_loader:
    print(masks.shape)
    masks=label_embedding(8,120,masks,label_w2v)
    # print(images.shape)
    # print(masks.shape)
    # print(masks[0])
    # mask=masks*255
    # print(mask)
    show_img(images, masks)
    print(masks.shape)
    i=i+1
    # 在这里使用images和masks进行训练
    pass
