import torch
from torch import nn
from torchvision import  models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG11_Weights, AlexNet_Weights


class myNetwork(nn.Module):
    def __init__(self):
        super(myNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2)
        )


        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 64 * 64, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class ResNet():
    def __init__(self, feature_extract,num_of_layers=18):
        if num_of_layers == 18:
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.name = "ResNet 18"
            self.model.fc = nn.Linear(512, 5)
        else:
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.name = "ResNet 50"
            self.model.fc = nn.Linear(2048, 5)
        self.__set_parameter_requires_grad(feature_extract)


    def __set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

class VGG11():
    def __init__(self):
        self.name = "VGG11"
        self.model = models.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[-1].in_features
        output = 5
        new_output = nn.Linear(in_features,output)
        self.model.classifier[-1] = new_output

class ALEXNET():
    def __init__(self):
        self.name = "ALEXNET"
        self.model = models.alexnet(weights = AlexNet_Weights.IMAGENET1K_V1)
        in_features = self.model.classifier[-1].in_features
        new_οutput = nn.Linear(in_features,5)
        self.model.classifier[-1] = new_οutput


# input = torch.randn((1,3,256,256))
# print(input.size())
# model = ResNet(feature_extract=False,num_of_layers=50).model
# output = model(input)
# print(output.size())
#
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# print(f"Number of trainable parameters: {params}\n")