import matplotlib.pyplot as plt
import os
from datetime import datetime

def save_training_plot(train_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'r-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train')
    plt.plot(val_accs, 'g-', label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存结果
    os.makedirs('../results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'../results/training_results_{timestamp}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Training results saved to {plot_path}")
    
    return plot_path

def evaluate_model(model, device, batch_size=128):
    from .dataset import get_dataloaders
    
    _, test_loader = get_dataloaders(batch_size)
    
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy