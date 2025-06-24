import os
import torch
import torch.optim as optim
import torch.nn as nn
from .model import SimpleLeNet
from .dataset import get_dataloaders
from .utils import save_training_plot, evaluate_model
from datetime import datetime

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(epochs=20, batch_size=128):
    # 创建数据目录
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(batch_size)
    
    # 模型初始化
    model = SimpleLeNet().to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    
    # 训练记录
    train_losses = []
    train_accs = []
    val_accs = []
    best_acc = 0.0
    best_model_path = ""
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # 训练阶段
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练指标
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(acc)
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f'../models/best_model_{timestamp}.pth'
            torch.save(model.state_dict(), best_model_path)
        
        # 输出epoch信息
        print(f'Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}%')
    
    print("Training completed!")
    
    # 保存训练结果图
    save_training_plot(train_losses, train_accs, val_accs)
    
    return best_model_path

if __name__ == "__main__":
    # 训练模型
    best_model_path = train_model()
    
    # 评估最佳模型
    print(f"Loading best model from {best_model_path} for evaluation...")
    model = SimpleLeNet().to(device)
    model.load_state_dict(torch.load(best_model_path))
    
    test_accuracy = evaluate_model(model, device)
    print(f"Test accuracy: {test_accuracy:.2f}%")