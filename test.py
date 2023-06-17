import os
import  pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from Model.data_loader import InpaintedDataset
from torchvision import transforms
from Model.network import ResNet, VGG11, ALEXNET
from train import sample_outputs,accuracy

if __name__ == '__main__':
    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    IMG_DIR_INPAINTED = data['IMG_DIR_INPAINTED']
    IMG_DIR_ORIGINAL = data['IMG_DIR_ORIGINAL']

    ANNOTATIONS_FILE = data['ANNOTATIONS_FILE']
    RESULTS_DIR = data['RESULTS_DIR']

    NUM_TRAINING_SAMPLES = data['TRAIN_VAL_TEST_SPLIT']['num_training_samples']
    NUM_VALIDATION_SAMPLES = data['TRAIN_VAL_TEST_SPLIT']['num_validation_samples']
    NUM_TESTING_SAMPLES = data['TRAIN_VAL_TEST_SPLIT']['num_testing_samples']

    BATCH_SIZE = data['BATCH_SIZE']

    CHECKPOINT_PATH = data['CHECKPOINT_PATH']

    checkpoint = torch.load(CHECKPOINT_PATH)

    model_type = checkpoint['type']

    print(f"Final model is a : {model_type}")

    if model_type == "ResNet 18":
        network = ResNet(feature_extract=False, num_of_layers=18)
    elif model_type == "ResNet 50":
        network = ResNet(feature_extract=False, num_of_layers=50)
    elif model_type == "VGG11":
        network = VGG11()
    elif model_type == "ALEXNET":
        network = ALEXNET()
    else:
        raise ValueError("Wrong network option")

    model = network.model
    model.load_state_dict(checkpoint['model_state_dict'])

    test_data = InpaintedDataset(annotations_file=ANNOTATIONS_FILE,
                                 img_dir_inpainted=IMG_DIR_INPAINTED,
                                 img_dir_original=IMG_DIR_ORIGINAL,
                                 transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                 samples_type='Test',
                                 num_training_samples=NUM_TRAINING_SAMPLES,
                                 num_validation_samples=NUM_VALIDATION_SAMPLES,
                                 num_testing_samples=NUM_TESTING_SAMPLES
                                 )

    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set size: {len(test_data)}\n")

    test_results_file = os.path.join(RESULTS_DIR, "test_results.csv")

    sample_outputs(model, test_dataloader, test_results_file)
    print("\nTEST SET METRICS:")
    try:
        accuracy(pd.read_csv(ANNOTATIONS_FILE), pd.read_csv(f"{RESULTS_DIR}\\test_results.csv"))
    except:
        accuracy(pd.read_csv(ANNOTATIONS_FILE), pd.read_csv(f"{RESULTS_DIR}/test_results.csv"))