import argparse
import os

import numpy as np
import torch
import yaml
from torchvision.ops import generalized_box_iou_loss
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from Model.network import ResNet
from Model.data_loader import InpaintedDataset
import csv
import math
from Model.Losses import IoCLoss


def sample_outputs(model, data_loader, file):
    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'x_min', 'y_min', 'x_max', 'y_max'])
        res = []
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):

                ids, inputs, _ = batch_data
                inputs.to(device)
                output = model(inputs)
                output = output.cpu()
                temp_list = [[i] + row.tolist() for i, row in zip(ids, output)]

                for sample in temp_list:
                    res.append(sample)
            writer.writerows(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Load checkpoint or not""")
    parser.add_argument('--load_checkpoint', action='store_true', default=False,
                        help='Set this flag to load a training checkpoint')

    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    IMG_DIR_INPAINTED = data['IMG_DIR_INPAINTED']
    IMG_DIR_ORIGINAL = data['IMG_DIR_ORIGINAL']

    ANNOTATIONS_FILE = data['ANNOTATIONS_FILE']
    RESULTS_DIR = data['RESULTS_DIR']

    NUM_EPOCHS = data['NUM_EPOCHS']
    BATCH_SIZE = data['BATCH_SIZE']
    LEARNING_RATE = data['LEARNING_RATE']

    NUM_TRAINING_SAMPLES = data['TRAIN_VAL_TEST_SPLIT']['num_training_samples']
    NUM_VALIDATION_SAMPLES = data['TRAIN_VAL_TEST_SPLIT']['num_validation_samples']
    NUM_TESTING_SAMPLES = data['TRAIN_VAL_TEST_SPLIT']['num_testing_samples']

    CHECKPOINT_PATH = data['CHECKPOINT_PATH']
    if args.load_checkpoint:
        checkpoint = torch.load(CHECKPOINT_PATH)

    criterion = IoCLoss()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    resnet = ResNet(feature_extract=False)

    model = resnet.model
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if args.load_checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

    # print(resnet.model)

    # for name, param in resnet.model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    training_data = InpaintedDataset(annotations_file=ANNOTATIONS_FILE,
                                     img_dir_inpainted=IMG_DIR_INPAINTED,
                                     img_dir_original=IMG_DIR_ORIGINAL,
                                     transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                     samples_type='Train',
                                     num_training_samples=NUM_TRAINING_SAMPLES,
                                     num_validation_samples=NUM_VALIDATION_SAMPLES,
                                     num_testing_samples=NUM_TESTING_SAMPLES)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training set size: {len(training_data)}")

    validation_data = InpaintedDataset(annotations_file=ANNOTATIONS_FILE,
                                       img_dir_inpainted=IMG_DIR_INPAINTED,
                                       img_dir_original=IMG_DIR_ORIGINAL,
                                       transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                       samples_type='Validation',
                                       num_training_samples=NUM_TRAINING_SAMPLES,
                                       num_validation_samples=NUM_VALIDATION_SAMPLES,
                                       num_testing_samples=NUM_TESTING_SAMPLES
                                       )
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Validation set size: {len(validation_data)}")

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
    print(f"Test set size: {len(test_data)}")

    training_accuracy = []
    validation_accuracy = []
    avg_batch_loss_train = []
    avg_batch_loss_val = []
    batches_train = math.ceil(len(training_data) / BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)

        model.train()

        avg_batch_loss_train.clear()
        batch_counter = 1
        for id, inputs, targets in train_dataloader:
            inputs.to(device)
            targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss =  criterion(outputs,targets)

            avg_batch_loss_train.append(loss.item())

            print(f"Loss: {loss}, Batch:{batch_counter}/{batches_train}")
            loss.backward()
            optimizer.step()
            batch_counter += 1

        training_accuracy.append(np.mean(avg_batch_loss_train))

        with torch.no_grad():
            model.eval()
            avg_batch_loss_val.clear()
            for _, inputs, targets in validation_dataloader:
                inputs.to(device)
                targets.to(device)
                outputs = model(inputs)
                loss =  criterion(outputs,targets)
                avg_batch_loss_val.append(loss.item())
            mean_val_loss = np.mean(avg_batch_loss_val)
            print(f"Mean Validation loss : {mean_val_loss}")
            validation_accuracy.append(mean_val_loss)

            if min(validation_accuracy) == mean_val_loss and (epoch + 1) > 1:
                # We found the best model until so far
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_val_loss,
                }, CHECKPOINT_PATH)

    training_results_file = os.path.join(RESULTS_DIR, "training_results.csv")
    validation_results_file = os.path.join(RESULTS_DIR, "validation_results.csv")
    test_results_file = os.path.join(RESULTS_DIR, "test_results.csv")

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)

    sample_outputs(model, train_dataloader, training_results_file)
    sample_outputs(model, validation_dataloader, validation_results_file)
    sample_outputs(model, test_dataloader, test_results_file)
