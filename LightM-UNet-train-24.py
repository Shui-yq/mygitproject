# -*- coding:utf-8 -*-
"""
作者：CS420InstallByWLF
日期：2021年09月06日
说明：本程序加载gz_data和mp_label中的数据，构建训练集与验证集，
     使用Attention U-Net进行异常体密度预测。程序中添加了重新初始化模型、
     优化器、数据加载器，以及清除缓存和检查预处理步骤的部分，确保更换数据集后不受上一次训练影响。
"""

from __future__ import annotations    
import math
import torch.nn as nn
import numpy as np
import torch
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from mamba_ssm import Mamba
from torchsummary import summary
from scipy.ndimage import zoom
from tqdm import tqdm
import os
import gc

# -------------------- 数据路径与归一化函数 --------------------
str1 = "../gz_data/"
str2 = "ob_gzz-"
str3 = "../mp_label/"
str4 = "mp-"
str5 = ".txt"

def MaxMinNormalization(x, Max, Min):
    # 线性归一化
    return (x - Min) / (Max - Min)

# -------------------- 设备设置 --------------------
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device3 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# -------------------- 数据加载 --------------------
num_train = 100
num_data = 576
num_label = 8000
data = np.zeros((num_train, num_data), dtype=float)
label = np.zeros((num_train, num_label), dtype=float)

for i in tqdm(range(num_train), desc="Loading data"):
    data_file = os.path.join(str1, str2 + str(i) + str5)
    label_file = os.path.join(str3, str4 + str(i) + str5)
    with open(data_file, "r") as f:
        lines = f.readlines()
        if len(lines) < num_data:
            print(f"Warning: 文件 {data_file} 行数不足 {num_data}")
        for j in range(num_data):
            try:
                # 每行假设有四个数字，取第四个数字
                _, _, _, d = map(float, lines[j].strip().split())
                data[i, j] = d
            except Exception as e:
                print(f"Error in file {data_file} at line {j}: {e}")
    max_val, min_val = np.max(data[i]), np.min(data[i])
    data[i, :] = MaxMinNormalization(data[i, :], max_val, min_val)
    with open(label_file, "r") as f_b:
        lines_b = f_b.readlines()
        if len(lines_b) < num_label:
            print(f"Warning: 文件 {label_file} 行数不足 {num_label}")
        for k in range(num_label):
            try:
                label[i, k] = float(lines_b[k].strip())
            except Exception as e:
                print(f"Error in file {label_file} at line {k}: {e}")

# 将数据转换为 4D 张量，reshape为 (num_train, 1, 24, 24)
data = data[:, np.newaxis, :]
dataa = data.reshape(num_train, 1, 24, 24)

# 划分训练集与验证集
train, val, train_label, val_label = train_test_split(dataa, label, test_size=0.2, random_state=42)

# -------------------- 数据集类 --------------------
class train_Dataset(Dataset):
    def __init__(self, data1, data2):
        self.x_data = torch.from_numpy(data1)
        self.y_data = torch.from_numpy(data2)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# 重建数据加载器函数
def build_data_loaders(train_data, train_labels, val_data, val_labels, batch_size=100):
    train_dataset = train_Dataset(train_data, train_labels)
    val_dataset = train_Dataset(val_data, val_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# 初始数据加载器
train_loader, val_loader = build_data_loaders(train, train_label, val, val_label, batch_size=100)

# -------------------- 轻量级 Mamba 网络相关定义 --------------------
def get_dwconv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = Convolution(
        spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels, 
        strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels
    )
    point_conv = Convolution(
        spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
        strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1
    )
    return torch.nn.Sequential(depth_conv, point_conv)

class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim  
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

def get_mamba_layer(
    spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims == 3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer

class ResMambaBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, norm: tuple | str,
                 kernel_size: int = 3, act: tuple | str = ("RELU", {"inplace": True})) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)
        self.conv2 = get_mamba_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)
    
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        x += identity        
        return x

class ResUpBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, norm: tuple | str,
                 kernel_size: int = 3, act: tuple | str = ("RELU", {"inplace": True})) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.skip_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)        
        return x

class LightMUNet(nn.Module):
    def __init__(self, spatial_dims: int = 2, init_filters: int = 8, in_channels: int = 1,
                 out_channels: int = 2, dropout_prob: float | None = None,
                 act: tuple | str = ("RELU", {"inplace": True}), norm: tuple | str = ("GROUP", {"num_groups": 8}),
                 norm_name: str = "", num_groups: int = 8, use_conv_final: bool = True,
                 blocks_down: tuple = (1, 2, 2, 4), blocks_up: tuple = (1, 1, 1),
                 upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_dwconv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)        
        self.out = nn.Linear(1152, 8000).to(device0)
        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = self.blocks_down, self.spatial_dims, self.init_filters, self.norm
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            downsample_mamba = (get_mamba_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                                if i > 0 else nn.Identity())
            down_layer = nn.Sequential(
                downsample_mamba, *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act)
                                     for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = self.upsample_mode, self.blocks_up, self.spatial_dims, self.init_filters, self.norm
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(*[ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                                for _ in range(blocks_up[i])])
            )
            up_samples.append(
                nn.Sequential(*[get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                                get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode)])
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_dwconv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []
        for down in self.down_layers:
            x = down(x)
            down_x.append(x)
        return x, down_x
         
    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)
        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()
        x = self.decode(x, down_x)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        output = self.out(x_flat)
        return output

# -------------------- 重置训练管线：重新初始化模型、优化器、数据加载器、清除缓存 --------------------
def reset_training_pipeline(model_class, optimizer_class, train_data, train_labels, val_data, val_labels, batch_size=100, lr=1e-4, device=device0):
    # 清空GPU缓存和Python垃圾回收
    torch.cuda.empty_cache()
    gc.collect()
    
    # 重新初始化数据加载器
    train_loader, val_loader = build_data_loaders(train_data, train_labels, val_data, val_labels, batch_size=batch_size)
    
    # 重新创建模型实例
    model = model_class().to(device)
    # (可选) 对模型权重进行初始化，若模型中未实现则调用reset_parameters()或apply(weight_reset)
    def weight_reset(m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
    model.apply(weight_reset)
    print("模型权重已重新初始化")
    
    # 重新创建优化器实例
    optimizer = optimizer_class(model.parameters(), lr=lr)
    
    print("训练管线已重置")
    return model, optimizer, train_loader, val_loader

# -------------------- 使用重置函数重新初始化模型与数据加载器 --------------------
# 当你更换数据集后，确保train, train_label, val, val_label为新的数据集，此处直接使用已有的
model, optimizer, train_loader, val_loader = reset_training_pipeline(
    model_class=lambda: LightMUNet(spatial_dims=2, in_channels=1, out_channels=2),
    optimizer_class=torch.optim.Adam,
    train_data=train,
    train_labels=train_label,
    val_data=val,
    val_labels=val_label,
    batch_size=100,
    lr=1e-4,
    device=device0
)

# -------------------- 训练循环 --------------------
loss_func = nn.MSELoss()
start_time = time.time()
epochs = 2
min_loss = float('inf')
record_train_loss = []
record_val_loss = []

for epoch in range(epochs):    
    model.train()
    mean_loss = 0.0    
    for step, (x, y) in enumerate(train_loader):
        b_x = x.to(device0, dtype=torch.float32)
        b_y = y.to(device0, dtype=torch.float32)
        optimizer.zero_grad()
        output = model(b_x)
        loss = loss_func(output, b_y)
        loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()
        mean_loss += loss.item()
    mean_loss /= len(train_loader)
    record_train_loss.append(mean_loss)
    
    model.eval()
    val_mean_loss = 0.0
    for step, (x, y) in enumerate(val_loader):
        val_x = x.to(device0, dtype=torch.float32)
        val_y = y.to(device0, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(val_x)            
            loss = loss_func(outputs, val_y) 
            val_mean_loss += loss.item()    
    val_mean_loss /= len(val_loader)
    record_val_loss.append(val_mean_loss)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {mean_loss:.4f}")
    print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_mean_loss:.4f}")
    
    if mean_loss < min_loss:
        min_loss = mean_loss
        save_path = os.path.join('model', f'min_loss{round(min_loss,0)}-LightM-UNet.pth')
        torch.save(model.state_dict(), save_path)
        print(f"模型权重已保存到 {save_path}")

end_time = time.time()
print('Finished Training')
print('Train time = ', round((end_time - start_time) / 60, 2), 'min')

save_path = os.path.join('model', f'min_loss{round(min_loss,0)}-LightM-UNet.pth')
torch.save(model, save_path)
print('模型已保存到', save_path)

np.savetxt('loss_train.txt', record_train_loss, fmt='%f')
np.savetxt('loss_val.txt', record_val_loss, fmt='%f')

plt.plot(range(epochs), record_train_loss, label='Train Loss')
plt.plot(range(epochs), record_val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('Finish this task!')
