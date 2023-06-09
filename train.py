import argparse
import os
import pandas as pd
import torch
import yaml
from torchvision.ops import generalized_box_iou_loss
from Model.Losses import L2Loss, giou_loss
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from Model.network import ResNet, VGG11, ALEXNET, myNetwork
from Model.data_loader import InpaintedDataset
import csv
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve
import numpy as np
import time
import seaborn as sns
import torch.nn.functional as F


def calculate_iou(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_min_intersection = max(x_min1, x_min2)
    y_min_intersection = max(y_min1, y_min2)
    x_max_intersection = min(x_max1, x_max2)
    y_max_intersection = min(y_max1, y_max2)

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x_max_intersection - x_min_intersection + 1) * max(0,
                                                                                  y_max_intersection - y_min_intersection + 1)

    # Calculate the areas of the bounding boxes
    box1_area = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
    box2_area = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)

    # Calculate the Union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou


def accuracy(ground_truth_df, test_preds_df):
    merged_df = pd.merge(ground_truth_df, test_preds_df, on='id', how='inner', suffixes=("_ground_truth", "_pred"))

    threshold = 0.5
    actual_labels = merged_df['fake_label_ground_truth']
    predicted_values = merged_df['fake_label_pred']
    predicted_labels = np.where(predicted_values >= threshold, 1, 0)

    # Accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)

    # Precision
    precision = precision_score(actual_labels, predicted_labels)

    # Recall
    recall = recall_score(actual_labels, predicted_labels)

    # F1 score
    f1 = f1_score(actual_labels, predicted_labels)

    # Confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels)

    # ROC AUC
    roc_auc = roc_auc_score(actual_labels, predicted_values)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(cm)
    print("ROC AUC:", roc_auc)

    # Plotting the confusion matrix
    labels = ['Real', 'Fake']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Plotting the ROC AUC curve
    fpr, tpr, thresholds = roc_curve(actual_labels, predicted_values)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve")
    plt.show()

    ground_truth_boxes = merged_df[
        ['x_min_ground_truth', 'y_min_ground_truth', 'x_max_ground_truth', 'y_max_ground_truth']]
    predicted_boxes = merged_df[['x_min_pred', 'y_min_pred', 'x_max_pred', 'y_max_pred']]

    # Calculate IoU for each pair of ground truth and predicted bounding boxes
    iou_scores = []
    for i in range(len(merged_df)):
        iou = calculate_iou(ground_truth_boxes.iloc[i], predicted_boxes.iloc[i])
        iou_scores.append(iou)

    # Add the IoU scores as a new column in the dataframe
    merged_df['IoU'] = iou_scores

    contains_artificial = merged_df[merged_df['fake_label_ground_truth'] == 1]
    contains_artificial = contains_artificial[contains_artificial['fake_label_pred'] > threshold]
    # Print the dataframe with the IoU scores
    print(f"Mean IoU: {contains_artificial.IoU.mean()} ")


def sample_outputs(model, data_loader, file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    with open(file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'fake_label', 'x_min', 'y_min', 'x_max', 'y_max'])
        res = []
        with torch.no_grad():

            model.eval()
            for batch_idx, batch_data in enumerate(data_loader):

                ids, inputs, _ = batch_data
                ids = list(ids)
                for i in range(len(ids)):
                    to_fill = 8 - len(ids[i])
                    ids[i] = '0' * to_fill + ids[i]

                inputs = inputs.to(device)
                output = model(inputs).to(device)
                classification_neuron = torch.sigmoid(output[:, 0])
                output = torch.cat((classification_neuron.unsqueeze(1), output[:, 1:]), dim=1)
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

    ALPHA = data['ALPHA']
    BETA = data['BETA']

    LOCALIZATION_LOSS = data['LOCALIZATION_LOSS']

    CHECKPOINT_PATH = data['CHECKPOINT_PATH']

    MODEL = data['MODEL']

    if MODEL == "ResNet18":
        network = ResNet(feature_extract=False, num_of_layers=18)
    elif MODEL == "ResNet50":
        network = ResNet(feature_extract=False, num_of_layers=50)
    elif MODEL == "VGG11":
        network = VGG11()
    elif MODEL == "ALEXNET":
        network = ALEXNET()
    elif MODEL == "myNetwork":
        network = myNetwork()
    else:
        raise ValueError("Wrong network option")

    if args.load_checkpoint:
        checkpoint = torch.load(CHECKPOINT_PATH)

    localization_loss_options = {"MSE_lOSS": nn.MSELoss(),
                                 "SMOOTH_L1_LOSS": nn.SmoothL1Loss(),
                                 "GIoU": giou_loss,
                                 "L2_LOSS": L2Loss()}

    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = localization_loss_options[LOCALIZATION_LOSS]

    print(f"localization loss: {criterion2}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device\n")

    print(f"Using {network.name} model\n")
    model = network.model
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {params}\n")

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
    print(f"Training set size: {len(training_data)}\n")

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
    print(f"Validation set size: {len(validation_data)}\n")

    # test_data = InpaintedDataset(annotations_file=ANNOTATIONS_FILE,
    #                              img_dir_inpainted=IMG_DIR_INPAINTED,
    #                              img_dir_original=IMG_DIR_ORIGINAL,
    #                              transform=transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #                              samples_type='Test',
    #                              num_training_samples=NUM_TRAINING_SAMPLES,
    #                              num_validation_samples=NUM_VALIDATION_SAMPLES,
    #                              num_testing_samples=NUM_TESTING_SAMPLES
    #                              )
    #
    # test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    # print(f"Test set size: {len(test_data)}\n")

    assert len(set(training_data.img_labels.id.unique()).intersection(
        validation_data.img_labels.id.unique())) == 0, "Something went wrong with the datasets. An image id is in both training and validation sets"

    training_accuracy = []
    validation_accuracy = []
    avg_batch_loss_train = []
    avg_batch_loss_val = []
    batches_train = math.ceil(len(training_data) / BATCH_SIZE)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)

        model.train()

        avg_batch_loss_train.clear()
        batch_counter = 1
        for id, inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs).to(device)

            bce_loss = criterion1(outputs[:, 0], targets[:, 0])
            if criterion2 == giou_loss:
                non_zero_rows = torch.all(targets[:, 1:] != 0, dim=1)
                filtered_boxes1 = outputs[:, 1:][non_zero_rows]
                filtered_boxes2 = targets[:, 1:][non_zero_rows]

                filtered_boxes1 = F.relu(filtered_boxes1)

                loc_loss = criterion2(filtered_boxes1, filtered_boxes2)

                if loc_loss < 0 :
                    print("")

            else:
                loc_loss = criterion2(outputs[:, 1:], targets[:, 1:])

            loss = ALPHA * bce_loss + BETA * loc_loss

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
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs).to(device)

                bce_loss = criterion1(outputs[:, 0], targets[:, 0])
                if criterion2 == giou_loss:
                    non_zero_rows = torch.all(targets[:, 1:] != 0, dim=1)
                    filtered_boxes1 = outputs[:, 1:][non_zero_rows]
                    filtered_boxes2 = targets[:, 1:][non_zero_rows]

                    filtered_boxes1 = F.relu(filtered_boxes1)

                    loc_loss = criterion2(filtered_boxes1, filtered_boxes2)
                else:
                    loc_loss = criterion2(outputs[:, 1:], targets[:, 1:])
                loss = ALPHA * bce_loss + BETA * loc_loss

                avg_batch_loss_val.append(loss.item())
            mean_val_loss = np.mean(avg_batch_loss_val)
            print(f"Mean Validation loss : {mean_val_loss}")
            validation_accuracy.append(mean_val_loss)

            if min(validation_accuracy) == mean_val_loss and (epoch + 1) > 1:
                # We found the best model until so far

                print(f"New best model found! (based on validation set performance)")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_val_loss,
                    'type': network.name
                }, CHECKPOINT_PATH)

    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time

    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int((running_time % 3600) % 60)

    # Print the running time
    print("Training Time: {} hours, {} minutes, {} seconds\n".format(hours, minutes, seconds))

    training_results_file = os.path.join(RESULTS_DIR, "training_results.csv")
    validation_results_file = os.path.join(RESULTS_DIR, "validation_results.csv")
    test_results_file = os.path.join(RESULTS_DIR, "test_results.csv")

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)

    sample_outputs(model, train_dataloader, training_results_file)
    sample_outputs(model, validation_dataloader, validation_results_file)
    # sample_outputs(model, test_dataloader, test_results_file)

    # Get the length of the lists
    length = len(training_accuracy)

    # Create x-axis values
    x = range(1, length + 1)

    # Plot the training and validation accuracies
    plt.plot(x, training_accuracy, label='Training Loss')
    plt.plot(x, validation_accuracy, label='Validation Loss')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel(f'Loss')
    plt.title('Training and Validation Losses')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()

    print("\nTRAINING SET METRICS:")
    try:
        accuracy(pd.read_csv(ANNOTATIONS_FILE), pd.read_csv(f"{RESULTS_DIR}\\training_results.csv"))
    except:
        accuracy(pd.read_csv(ANNOTATIONS_FILE), pd.read_csv(f"{RESULTS_DIR}/training_results.csv"))

    print("-" * 10)
    print("\nVALIDATION SET METRICS:")
    try:
        accuracy(pd.read_csv(ANNOTATIONS_FILE), pd.read_csv(f"{RESULTS_DIR}\\validation_results.csv"))
    except:
        accuracy(pd.read_csv(ANNOTATIONS_FILE), pd.read_csv(f"{RESULTS_DIR}/validation_results.csv"))
    # print("-" * 10)
    # print("\nTEST SET METRICS:")
    # accuracy(pd.read_csv(ANNOTATIONS_FILE), pd.read_csv(f"{RESULTS_DIR}\\test_results.csv"))
