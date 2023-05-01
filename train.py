import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from Model.network import ResNet
from Model.network import myNetwork
from Model.data_loader import Data_loader
import csv


def sample_outputs():
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)
    with open(f'{RESULTS_DIR}\\results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x_min', 'y_min', 'x_max', 'y_max'])

        with torch.no_grad():
            for inputs, targets in train_dataloader:
                inputs.to(device)
                targets.to(device)
                outputs = model(inputs)

                for output in outputs:
                    x_min, y_min, x_max, y_max = output.tolist()
                    writer.writerow([x_min, y_min, x_max, y_max])


if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    IMG_DIR = data['IMG_DIR']
    ANNOTATIONS_FILE = data['ANNOTATIONS_FILE']
    NUM_EPOCHS = data['NUM_EPOCHS']
    BATCH_SIZE = data['BATCH_SIZE']
    LEARNING_RATE = data['LEARNING_RATE']
    RESULTS_DIR = data['RESULTS_DIR']

    criterion = nn.MSELoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    resnet = ResNet(feature_extract=True)
    model = resnet.model

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # print(resnet.model)

    # for name, param in resnet.model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    training_data = Data_loader(annotations_file=ANNOTATIONS_FILE, img_dir=IMG_DIR,
                                transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)

        model.train()

        for inputs, targets in train_dataloader:
            inputs.to(device)
            targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss}")

    sample_outputs()
