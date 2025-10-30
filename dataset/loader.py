from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_tiny_imagenet_loaders(data_path='data/tiny-imagenet-200', batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(root=f'{data_path}/train', transform=transform)
    val_dataset = ImageFolder(root=f'{data_path}/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
