import torch
import yaml
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from Model.network import ResNet
from Model.network import myNetwork
from Model.data_loader import Data_loader
import csv
from Model.Losses import IoCLoss


def sample_outputs(training_data):
    samples_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)
    with open('training_samples.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id','x_min', 'y_min', 'x_max', 'y_max'])
        res = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(samples_dataloader):
                if batch_idx == 3:
                    break

                ids , inputs, _ = batch_data
                inputs.to(device)
                output = model(inputs)
                output = output.cpu()
                temp_list = [[i] + row.tolist() for i, row in zip(ids, output)]

                for sample in temp_list:
                    res.append(sample)
            writer.writerows(res)

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    IMG_DIR = data['IMG_DIR_TRAINING']
    ANNOTATIONS_FILE = data['ANNOTATIONS_FILE_TRAINING']
    NUM_EPOCHS = data['NUM_EPOCHS']
    BATCH_SIZE = data['BATCH_SIZE']
    LEARNING_RATE = data['LEARNING_RATE']
    RESULTS_DIR = data['RESULTS_DIR']

    criterion_mse = MSELoss()
    criterion_ioc = IoCLoss()
    criterion = criterion_mse + 1000*criterion_ioc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    resnet = ResNet(feature_extract=False)
    model = resnet.model

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # print(resnet.model)

    # for name, param in resnet.model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    training_data = Data_loader(annotations_file=ANNOTATIONS_FILE, img_dir=IMG_DIR,
                                transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training set size: {len(training_data)}")
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)

        model.train()

        for _, inputs, targets in train_dataloader:
            inputs.to(device)
            targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            print(f"Loss: {loss}")
            loss.backward()
            optimizer.step()

    sample_outputs(training_data)
    torch.save(model.state_dict(), 'model.pth')
