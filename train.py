import torch
from torch import nn, optim
from models.custom_net import CustomNet
from utils.training import train, validate
from dataset.loader import get_tiny_imagenet_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=32)

model = CustomNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer, device)
    val_acc = validate(model, val_loader, criterion, device)
    best_acc = max(best_acc, val_acc)

print(f"Best validation accuracy: {best_acc:.2f}%")
torch.save(model.state_dict(), 'checkpoints/best_model.pth')
