import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_loader import train_loader, val_loader, device
from model_resnet import CustomResNet

## ----------------- Hyperparameters -----------------
NUM_EPOCHS = 40  
LEARNING_RATE = 0.001

# ----------------- Set Filename -----------------
model_num = 101
filename_weight = f"./weight/res4_360_{model_num}.pth"
plot_name = "res4_360_" + str(model_num)

# ----------------- Initialize model -----------------
model = CustomResNet(num_classes=100, pretrained=True).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: 1 if epoch < 10 else 0.3
) 
writer = SummaryWriter()

# ----------------- CutMix Implementation -----------------
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------- Training Function -----------------
def train_one_epoch_cutmix(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", leave=False)
    
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)
        images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0, device=device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        train_loader_tqdm.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

    return avg_loss, accuracy

# ----------------- Validation Function -----------------
def validate(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Accuracy/val", accuracy, epoch)
    
    return avg_loss, accuracy

# ----------------- Plotting Function -----------------
def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, plot_name):

    os.makedirs("plot", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12,5))
    
    ### Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label="Train Loss")
    plt.plot(epochs, val_losses, 's-', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    ### Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'o-', label="Train Accuracy")
    plt.plot(epochs, val_accuracies, 's-', label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    
    plt.suptitle(plot_name)
    filename_plot = os.path.join("plot", f"{plot_name}.png")
    plt.savefig(filename_plot)
    plt.close()
    print(f"Plot saved to {filename_plot}")

# ----------------- Main Training Program -----------------
best_acc = 0.0
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    train_loss, train_acc = train_one_epoch_cutmix(epoch)
    val_loss, val_acc = validate(epoch)
    
    scheduler.step() # Adjust learning rate
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), filename_weight)
        print(f"---------- Best model saved with accuracy: {best_acc:.2f}% ----------")

writer.close()
plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, plot_name)
print("Training Completed!")
