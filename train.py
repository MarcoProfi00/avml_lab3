import os
import torch
from torch import nn, optim
import wandb
from models.custom_net import CustomNet
from utils.training import train, validate
from dataset.loader import get_tiny_imagenet_loaders

# --- CONFIGURAZIONE DISPOSITIVO ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- INIZIALIZZAZIONE W&B ---
wandb.init(
    project="tiny-imagenet-lab2",
    config={
        "batch_size": 32,
        "epochs": 2,
        "optimizer": "SGD",
        "lr": 0.001,
        "momentum": 0.9
    }
)

# --- DATASET & MODEL ---
train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=32)
model = CustomNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0
num_epochs = 2

# --- TRAINING LOOP ---
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer, device)
    val_acc = validate(model, val_loader, criterion, device)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_acc": val_acc
    })

    best_acc = max(best_acc, val_acc)

# --- SALVATAGGIO MODELLO ---
print(f"Best validation accuracy: {best_acc:.2f}%")

os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/best_model.pth')

# --- CHIUSURA W&B ---
wandb.finish()
