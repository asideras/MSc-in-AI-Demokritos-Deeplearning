import torch
from torch import nn
from torchvision import  models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class myNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return x


class ResNet():
    def __init__(self, feature_extract,num_of_layers=18):
        if num_of_layers == 18:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.name = "ResNet 18"
        else:
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.name = "ResNet 50"
        self.__set_parameter_requires_grad(feature_extract)
        self.model.fc = nn.Linear(512, 5)

    def __set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False


