import torch.nn as nn


class DummyEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DummyModel(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=8, num_classes=21):
        super().__init__()
        self.encoder = DummyEncoder(in_channels, hidden_channels)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


class DummyModelWithoutEncoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        logits = self.classifier(x)
        return logits
