import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=128):
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 加载数据集
    train_set = torchvision.datasets.FashionMNIST(
        root='../data/', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(
        root='../data/', train=False, download=True, transform=transform)
    
    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader