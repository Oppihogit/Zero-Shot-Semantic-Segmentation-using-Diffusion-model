import torch
import torch.nn as nn
from torchvision import transforms, datasets
import  matplotlib.pyplot as plt
import torchvision
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torch.optim as optim
from tqdm import tqdm
from torch.nn import init
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F
import os
import numpy as np
import cv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Diffusion.py
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Scheduler.py
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
class UNet(nn.Module):
    def __init__(self, dim,T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(dim, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, dim, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


def data_loader(batch_size,img_size):

    # Define data transforms to be applied to each image
    transform_mask = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
    ])

    transform_img = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
    ])


    # Define the root directory where the image files are stored
    mask_dir = 'Datasplit/'
    img_dir='Dataimg/'
    # Use the ImageFolder class from torchvision to load the images
    mask_dataset = datasets.ImageFolder(root=mask_dir, transform=transform_mask)
    img_dataset=datasets.ImageFolder(root=img_dir, transform=transform_img)
    img_w2v_embedding_dict=torch.load("Data\imgid_w2v_embedding_sorted.pth")
    img_w2v_embedding=img_w2v_embedding_dict.values()
    # Create a DataLoader to load the images in batches during training

    img_w2v_embedding_resize=[]

    for i in img_w2v_embedding:
        zero_tensor=torch.zeros(img_size,img_size)
        dim1=i.size(0)
        dim2=i.size(1)

        zero_tensor[:dim1,:dim2]=i
        resize_tensor = torch.unsqueeze(zero_tensor, 0)
        img_w2v_embedding_resize.append(resize_tensor)


    # torch.save(mask_dataset,'Data/DataLoader/mask_dataset.pth')
    # torch.save(img_dataset, 'Data/DataLoader/img_dataset.pth')
    # torch.save(img_w2v_embedding_resize, 'Data/DataLoader/w2v_dataset.pth')
    # print('w2v resize done')

    dataloader_mask = torch.utils.data.DataLoader(mask_dataset, batch_size=batch_size)
    dataloader_img=torch.utils.data.DataLoader(img_dataset, batch_size=batch_size)
    dataloader_w2v=torch.utils.data.DataLoader(img_w2v_embedding_resize, batch_size=batch_size)
    return dataloader_img,dataloader_mask,dataloader_w2v

#source from https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Diffusion.py
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T,Guidance_noise):

        x_t = x_T#total noise
        for time_step in reversed(range(self.T)):
            if time_step%50==0:
                print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mask_data = x_t[:, 3:, :, :]
            #img_data=x_t[:, :3, :, :]
            guidance=Guidance_noise[time_step]
            img_data=guidance[:,:3,:,:]

            mask_data=mask_data.to(device)
            img_data=img_data.to(device)
            x_t=torch.cat((img_data,mask_data),dim=1)

            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            print(noise.shape)
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t

        return torch.clip(x_0, -1, 1)

class NCGaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            if time_step%50==0:
                print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

def guidance_set(x,beta_1, beta_T, T):
    noise=torch.randn_like(x)
    alphas = 1. - torch.linspace(beta_1, beta_T, T).double()
    print(len(alphas))
    alphas_bar = torch.cumprod(alphas, dim=0)
    noise_list=[]
    for i in list(range(T)):
        t = torch.full((x.shape[0],),i)
        noise_i = (extract(torch.sqrt(alphas_bar), t, x.shape) * x + extract(torch.sqrt(1. - alphas_bar), t,x.shape) * noise)
        noise_list.append(noise_i)
    return  noise_list

def classifier(sample,w2v_dict):
    device = 'cuda'
    def show_img(images):
        label_colors ={
            0: (0, 0, 0),  # unlabeled
            1: (0, 0, 0),  # ego vehicle
            2: (0, 0, 0),  # rectification border
            3: (0, 0, 0),  # out of roi
            4: (0, 0, 0),  # static
            5: (111, 74, 0),  # dynamic
            6: (81, 0, 81),  # ground
            7: (128, 64, 128),  # road
            8: (244, 35, 232),  # sidewalk
            9: (250, 170, 160),  # parking
            10: (230, 150, 140),  # rail track
            11: (70, 70, 70),  # building
            12: (102, 102, 156),  # wall
            13: (190, 153, 153),  # fence
            14: (180, 165, 180),  # guard rail
            15: (150, 100, 100),  # bridge
            16: (150, 120, 90),  # tunnel
            17: (153, 153, 153),  # pole
            18: (153, 153, 153), # pole group
            19: (250, 170, 30),  # traffic light
            20: (220, 220, 0),  # traffic sign
            21: (107, 142, 35),  # vegetation
            22: (152, 251, 152),  # terrain
            23: (70, 130, 180),  # sky
            24: (220, 20, 60),  # person
            25: (255, 0, 0),  # rider
            26: (0, 0, 142),  # car
            27: (0, 0, 70),  # truck
            28: (0, 60, 100),  # bus
            29: (0, 0, 90),  # caravan
            30: (0, 0, 110),  # trailer
            31: (0, 80, 100),  # train
            32: (0, 0, 230),  # motorcycle
            33: (119, 11, 32),  # bicycle
        }
        image_tensor=images
        rgb_image_tensor = torch.zeros((sample.shape[0], 3, 128, 128), dtype=torch.uint8, device=device)
        for label_id, color in label_colors.items():
            mask = (image_tensor == label_id).to(torch.uint8)
            for channel, color_value in enumerate(color):
                rgb_image_tensor[:, channel] += (mask.squeeze(1) * color_value)

        # 将张量转换回CPU以便于显示
        rgb_image_tensor_cpu = rgb_image_tensor.cpu()
        print(rgb_image_tensor_cpu.shape)
        return rgb_image_tensor

    def find_closest(dict1, tensor2):
        # Compute absolute differences between each element in tensor2 and tensor1
        abs_diffs = torch.abs(tensor2.view(-1, 1) - dict1.view(1, -1))

        # Find the indices of the closest elements in tensor1 for each element in tensor2
        min_indices = torch.argmin(abs_diffs, dim=1)

        # Replace elements in tensor2 with the closest elements in tensor1
        result = dict1[min_indices]

        return result

    def v_to_c(tensor1,tensor2):
        # Use broadcasting to find where the rows in tensor1 equal the rows in tensor2
        matches = torch.eq(tensor1[:, None, :], tensor2)

        # Combine the matches along the last dimension
        combined_matches = matches.all(-1)

        # Find the non-zero indices
        match_indices = torch.nonzero(combined_matches).squeeze(-1).to('cuda')

        # Initialize tensor3 with all elements set to 1000
        tensor3 = torch.full((tensor1.shape[0],), 0, dtype=torch.long).to('cuda')

        # Set the corresponding indices in tensor3
        tensor3[match_indices[:, 0]] = match_indices[:, 1]
        return tensor3

    def w2v_to_class(w2v_data_flat,w2v_dict):
        dict1=torch.tensor([-1.0,-0.75,-0.5,-0.25,0.25,0.5,0.75,1]).to('cuda')

        d1=w2v_data_flat[:,0]
        d2=w2v_data_flat[:,1]
        d1_new = find_closest(dict1, d1)
        d2_new=find_closest(dict1, d2)
        mask_advance=torch.stack((d1_new, d2_new), dim=1)
        mask_class=v_to_c(mask_advance.to('cuda'),torch.tensor(list(w2v_dict.values())).to('cuda'))
        return mask_class

    def unseen_segementation(mask,batch_size,w2v_dict):
        mask_img=mask.view(batch_size,128,128,1)
        # image_upper_half = mask_img[:,:64, :, :]
        # print(mask_img[:,:64, :, :][mask_img[:,:64, :, :]==24])
        # mask_img[:,:64, :, :] = image_upper_half
        # print(mask_img[:,:64, :, :][mask_img[:,:64, :, :]==24])
        #
        # print(mask_img[:,:64, :, :][(mask_img[:,:64, :, :] == 24)])
        mask_img=mask_img.permute(0,3,1,2)
        return mask_img


    w2v_data=sample[:,3:,:,:]
    batch_size=sample.shape[0]
    #print(w2v_data.shape)
    w2v_data=w2v_data.permute(0,2,3,1)
    num_pix=w2v_data.shape[0]*w2v_data.shape[1]*w2v_data.shape[2]
    w2v_data_flat=w2v_data.reshape(num_pix,w2v_data.shape[-1])
    #print(w2v_data_flat)
    mask_class_flat=w2v_to_class(w2v_data_flat,w2v_dict)
    mask_class = unseen_segementation(mask_class_flat, batch_size, w2v_dict)
    rgb_image_tensor= show_img(mask_class)
    return rgb_image_tensor



    #print(w2v_data_flat.shape)
def preprocess(filename):
    def blur_image(image, kernel_size=5, sigma=1):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    def sharpen_image(image):
        # 使用拉普拉斯算子进行锐化
        laplacian = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(image, -1, laplacian)
        return sharpened_image

    image_path = filename
    image = cv2.imread(image_path)
    #image = sharpen_image(image)
    image = cv2.resize(image, (128, 128))

    image = blur_image(image, kernel_size=5, sigma=1)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用 Sobel 算子
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 将梯度幅值归一化到 [0, 1] 范围
    normalized_sobel_magnitude = cv2.normalize(sobel_magnitude, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)

    # 将原始彩色图像转换为浮点数并归一化
    normalized_color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    # 将原始图像与边缘增强图像相乘（需要将灰度图像扩展为3通道）
    combined_image = 1*normalized_color_image + 0.35*np.stack([normalized_sobel_magnitude] * 3, axis=-1)
    combined_tensor = torch.tensor(combined_image)
    # 显示原始图像、边缘增强图像和相乘后的图像
    # plt.figure(figsize=(18, 6))
    # plt.subplot(131), plt.imshow(normalized_color_image), plt.title('Original Image'), plt.axis('off')
    # plt.subplot(132), plt.imshow(normalized_sobel_magnitude, cmap='gray'), plt.title('Edge Enhanced Image'), plt.axis(
    #     'off')
    # plt.subplot(133), plt.imshow(combined_image), plt.title('Combined Image'), plt.axis('off')
    # plt.show()
    return combined_tensor
w2v_dict={0: (0.0, 0.0), 1: (-1.0, -1.0),
     2: (-1.0, -0.75), 3: (-1.0, -0.5),
     4: (-1.0, -0.25), 5: (-1.0, 0.25),
     6: (-1.0, 0.5), 7: (-0.75, -1.0),
     8: (-0.75, -0.75), 9: (-0.75, -0.5),
     10: (-0.75, -0.25), 11: (-0.5, -1.0),
     12: (-0.5, -0.75), 13: (-0.5, -0.5),
     14: (-0.5, -0.25), 15: (-0.5, 0.25),
     16: (-0.5, 0.5), 17: (-0.25, -1.0),
     18: (-0.25, -0.75), 19: (-0.25, -0.5),
     20: (-0.25, -0.25), 21: (0.25, -1.0),
     22: (0.25, -0.75), 23: (0.5, -1.0),
     24: (0.75, -1.0), 25: (0.75, -0.75),
     26: (1.0, -1.0), 27: (1.0, -0.75),
     28: (1.0, -0.5), 29: (1.0, -0.25),
     30: (1.0, 0.25), 31: (1.0, 0.5),
     32: (1.0, 0.75), 33: (1.0, 1.0)}
def sampler_img(modelConfig):
    w2v_dict = {0: (0.0, 0.0), 1: (-1.0, -1.0),
                2: (-1.0, -0.75), 3: (-1.0, -0.5),
                4: (-1.0, -0.25), 5: (-1.0, 0.25),
                6: (-1.0, 0.5), 7: (-0.75, -1.0),
                8: (-0.75, -0.75), 9: (-0.75, -0.5),
                10: (-0.75, -0.25), 11: (-0.5, -1.0),
                12: (-0.5, -0.75), 13: (-0.5, -0.5),
                14: (-0.5, -0.25), 15: (-0.5, 0.25),
                16: (-0.5, 0.5), 17: (-0.25, -1.0),
                18: (-0.25, -0.75), 19: (-0.25, -0.5),
                20: (-0.25, -0.25), 21: (0.25, -1.0),
                22: (0.25, -0.75), 23: (0.5, -1.0),
                24: (0.75, -1.0), 25: (0.75, -0.75),
                26: (1.0, -1.0), 27: (1.0, -0.75),
                28: (1.0, -0.5), 29: (1.0, -0.25),
                30: (1.0, 0.25), 31: (1.0, 0.5),
                32: (1.0, 0.75), 33: (1.0, 1.0)}
    device = torch.device(modelConfig["device"])
    # Set the device to run on

    # Initialize the UNet model
    net_model = UNet(
        dim=modelConfig["dim"],
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    # Load a pre-trained model if specified
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(
            torch.load(
                os.path.join(
                    modelConfig["save_weight_dir"],
                    modelConfig["training_load_weight"]
                ),
                map_location=device
            )
        )

    # Initialize the optimizer and schedulers
    optimizer = torch.optim.AdamW(
        net_model.parameters(),
        lr=modelConfig["lr"],
        weight_decay=1e-4
    )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=modelConfig["epoch"],
        eta_min=0,
        last_epoch=-1
    )
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10,
        after_scheduler=cosineScheduler
    )
    # start training

    for filename in range(10):
        with torch.no_grad():
            weight = torch.load("Checkpoints/city2/ckpt_499_.pt", map_location=device)
            #weight = torch.load("Checkpoints/ckpt_105_.pt", map_location=device)
            net_model.load_state_dict(weight['model_state_dict'])
            #net_model.load_state_dict(weight)


            sampler = GaussianDiffusionSampler(net_model, modelConfig["beta_1"], modelConfig["beta_T"],
                                               modelConfig["T"]).to(device)

            NCsampler = NCGaussianDiffusionSampler(net_model, modelConfig["beta_1"], modelConfig["beta_T"],
                                               modelConfig["T"]).to(device)

            noisyImage = torch.randn(
                size=[modelConfig["batch_size"], modelConfig["dim"], modelConfig["img_size"], modelConfig["img_size"]],
                device=device)

            # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            # guidance_batch = torch.load('classifier_data/batch200.pth')
            # guidance_selected = guidance_batch[29]
            #guidance_batch = torch.load('SampledImgs/adimg_35.pth')

            guidance_batch = preprocess('SampledImgs/0029.png')

            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            guidance_selected = guidance_batch.permute(2,0,1)
            guidance_selected = normalize(guidance_selected)
            guidance_selected = torch.unsqueeze(guidance_selected, 0)
            guidance = guidance_selected.repeat((modelConfig["batch_size"], 1, 1, 1))

                #---conditional sampler---
            # guidance_img = guidance[:, :3, :, :]
            # guidance=guidance_set(guidance_img,modelConfig["beta_1"],modelConfig["beta_T"],modelConfig["T"])
            # sampledImgs = sampler(noisyImage,guidance)

            #     #---unconditional sampler---
            noisyImage=noisyImage
            sampledImgs = NCsampler(noisyImage)
            torch.save(sampledImgs,'SampledImgs/sample01.pth')
            img_data = sampledImgs[:, :3, :, :]
            print(img_data.max(),'min',img_data.min())
            torch.save(sampledImgs,'classifier_data/train_class_1.pth')
            mask_data = classifier(sampledImgs, w2v_dict)
            mask_data=mask_data/255
            #img_data = img_data * 0.5 + 0.5  # [0 ~ 1]
            show_data = torch.cat((img_data.to(device), mask_data), dim=2)
            torch.save(show_data, 'SampledImgs/NC/' + 'NC'+ str(filename) + 'sample.pth')
            save_data = torchvision.utils.make_grid(show_data, nrow=8, padding=5)
            plt_file = torchvision.transforms.ToPILImage(mode='RGB')(save_data)
            plt_file.save('SampledImgs/NC/' +'NC'+ str(filename) + 'sample.jpg')
            # plt.rcParams['figure.dpi'] = 100
            # plt.grid(False)
            # plt.imshow(
            #     torchvision.utils.make_grid(show_data).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1,0),
            #     cmap=plt.cm.binary)
            # plt.show()
            # plt.pause(0.0001)


Parameter = {
    "epoch": 100,
    "batch_size": 8,
    "T": 1000,
    "channel": 128,
    "channel_mult": [1, 2, 3, 4],
    "attn": [2],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 1e-4,
    "multiplier": 2.,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    "dim":5,
    "img_size": 128,
    "grad_clip": 1.,
    "device": "cuda",
    "training_load_weight": None,
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "ckpt_000_.pt",
    "sampled_dir": "./SampledImgs/",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",
    "nrow": 8
}

# use this code can train the model
#train(Parameter)
#sampler_train_data_c(Parameter)
sampler_img(Parameter)


    # plt.rcParams['figure.dpi'] = 175
    # plt.grid(False)
    # plt.imshow(torchvision.utils.make_grid(mask_set[0]).cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0),
    #            cmap=plt.cm.binary)
    # plt.show()
