import cv2
import pandas as pd
import torch
import yaml
from Model.network import ResNet
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.io import read_image
from torchvision import transforms
import torchvision
import cv2


def demonstrate_result(img_id):
    to_fill = 8 - len(img_id)
    img_id_2 = '0' * to_fill + img_id

    ground_truths = pd.read_csv("C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Data\\anns_class_local.csv")

    selected_row = ground_truths.loc[ground_truths['id'] == int(img_id)]

    fake_label = selected_row.fake_label.item()
    print(f"fake: {fake_label}")

    original = f"C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Data\\original\\Places365_val_{img_id_2}.jpg"
    original_masked = f"C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Data\\original_masked\\{img_id_2}.png"

    if fake_label:
        inpainted_path = f"C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Data\\inpainted\\{img_id}_second_out.jpg"
    else:
        inpainted_path = original

    checkpoint = torch.load(
        "C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Model Checkpoint\\model.pt",
    map_location=torch.device('cpu'))

    print(f"Image path: {inpainted_path}")

    network = ResNet(feature_extract=False)
    #print(f"Using {network.name} model\n")
    model = network.model
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" checkpoint loss: {checkpoint['loss']}")
    model.eval()
    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    inpainted  = read_image(inpainted_path).float()
    new_image = transform(inpainted)

    # Prepare the image tensor for inference
    new_image = new_image.unsqueeze(0)


    # Forward pass through the model
    with torch.no_grad():
        output = model(new_image)
        class_unit = torch.sigmoid(output[0, 0]).unsqueeze(0)
        model_output = torch.cat((class_unit.unsqueeze(1), output[:, 1:]), dim=1)[0]
    output = model_output.tolist()
    print(f"Model's confidence about whether the image contains an artificial part: {output[0]}")
    print("Model output: ", model_output)

    # Artificial Part (if applicable)
    if output[0] > 0.5 and fake_label:
        # Load and plot the images
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Original Image
        original_image = Image.open(original)
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")

        # Masked Image
        masked_image = Image.open(original_masked)
        axes[0, 1].imshow(masked_image)
        axes[0, 1].set_title("Masked Image")

        # Inpainted Image
        inpainted_image = Image.open(inpainted_path)
        axes[1, 0].imshow(inpainted_image)
        axes[1, 0].set_title("Inpainted Image")

        output = [int(pt) for pt in output]
        coordinates = tuple(output[1:])

        # Read the image with OpenCV
        inpainted_img = cv2.imread(inpainted_path)

        # Convert the color channel ordering from BGR to RGB
        inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)

        # Draw the rectangle on the image
        bboxed = cv2.rectangle(inpainted_img, (coordinates[1],coordinates[0],coordinates[3],coordinates[2]), (0, 255, 0), 2)

        # Display the image using Matplotlib
        axes[1, 1].imshow(bboxed)
        axes[1, 1].set_title("Artificial Part")

        # Remove axis ticks
        for ax in axes.flatten():
            ax.axis("off")

        # Adjust layout
        plt.tight_layout()
        plt.show()
    elif output[0] > 0.5 and not fake_label:
        print("The model has been wrong. The image does not contain an artificial part")
    elif output[0] < 0.5 and not fake_label:
        print("The model correctly said that the image does not contain an artificial part")
    elif output[0] < 0.5 and fake_label:
        print("The model has been wrong. The image contains an artificial part")
        # Load and plot the images
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Original Image
        original_image = Image.open(original)
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title("Original Image")

        # Masked Image
        masked_image = Image.open(original_masked)
        axes[0, 1].imshow(masked_image)
        axes[0, 1].set_title("Masked Image")

        # Inpainted Image
        inpainted_image = Image.open(inpainted_path)
        axes[1, 0].imshow(inpainted_image)
        axes[1, 0].set_title("Inpainted Image")

        # Remove axis ticks
        for ax in axes.flatten():
            ax.axis("off")

        # Adjust layout
        plt.tight_layout()
        plt.show()


def draw_bbox(image_path, x_min, y_min, x_max, y_max):
    # Load the image
    img = cv2.imread(image_path)
    # Draw the bounding box
    cv2.rectangle(img, (int(y_min), int(x_min)), (int(y_max), int(x_max)), (0, 255, 0), 2)
    # Display the image
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':

    # img_id = "27092"
    # demonstrate_result(img_id)
    # exit()

    with open('../config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    input_path = data['IMG_DIR_INPAINTED']
    results_dir = data['RESULTS_DIR']
    results = pd.read_csv(f"{results_dir}\\test_results.csv")
    #results = pd.read_csv(
       # "C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Data\\anns_class_local.csv")

    for index, row in results.iterrows():
        id = str(int(row.id))
        img_path = f"{input_path}\\{id}_second_out.jpg"
        fake_label, x_min, y_min, x_max, y_max = row['fake_label'], row['x_min'], row['y_min'], row['x_max'], row['y_max']
        if fake_label:
            draw_bbox(img_path, x_min, y_min, x_max, y_max)
        else:
            print(f"Image with id: {id} does not contain artificial part")
        print("---")


