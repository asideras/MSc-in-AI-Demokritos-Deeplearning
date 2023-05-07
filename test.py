import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from Model.data_loader import Data_loader
from Model.network import ResNet
import csv

if __name__ == '__main__':

    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    ANNOTATIONS_FILE = data['ANNOTATIONS_FILE_VALIDATION']
    IMG_DIR = data['IMG_DIR_VALIDATION']
    RESULTS_DIR = data['VALIDATION_RESULTS_DIR']

    BATCH_SIZE = data['BATCH_SIZE']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet(feature_extract=False)
    model.load_state_dict(torch.load('model.pth', map_location=device))

    testing_data = Data_loader(annotations_file=ANNOTATIONS_FILE, img_dir=IMG_DIR,
                               transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    test_dataloader = DataLoader(testing_data, batch_size=BATCH_SIZE, shuffle=False)

    outputs = []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs.to(device)
            output = model(inputs)
            output = output.cpu()
            outputs.append(output)

    csv_file = f'{RESULTS_DIR}\\.csv'

    # Write the outputs to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Output'])
        writer.writerows(outputs)
