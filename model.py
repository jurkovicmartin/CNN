import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected network (classification)
        self.fc1 = nn.Linear(64 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor):
        # Input 3x150x150
        x = self.relu(self.conv1(x)) # -> 32x75x75
        x = self.relu(self.conv2(x)) # -> 64x37x37
        x = self.relu(self.conv3(x)) # -> 64x37x37
        x = self.pool(x)             # -> 64x19x19
        x = self.relu(self.conv4(x)) # -> 64x19x19
        x = self.pool(x)             # -> 64x9x9

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
