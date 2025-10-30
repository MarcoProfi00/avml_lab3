import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        #Pooling
        self.pool1 = nn.MaxPool2d(2, 2) # dimezza H e W (224 -> 112)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #Pooling
        self.pool2 = nn.MaxPool2d(2, 2) # dimezza H e W (112 -> 56)
        # Add more layers...

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # dimezza H e W (56 -> 28)

        # Layer fully connected
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 200)  # 200 classi TinyImageNet

    def forward(self, x):
        # Define forward pass
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        # B x 3 x 224 x 224
        # x = self.conv1(x).relu() # B x 64 x 224 x 224

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, features)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # output logits (non softmax)

        return x