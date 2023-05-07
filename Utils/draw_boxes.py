import cv2
import pandas as pd
import yaml


def draw_bbox(image_path, x_min, y_min, x_max, y_max):
    # Load the image
    img = cv2.imread(image_path)
    # Draw the bounding box
    cv2.rectangle(img, (y_min, x_min), (y_max, x_max), (0, 255, 0), 2)
    # Display the image
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


with open('../config.yaml', 'r') as file:
    data = yaml.safe_load(file)

input_path = data['IMG_DIR_VALIDATION']
results_dir = data['RESULTS_DIR']
results = pd.read_csv(f"{results_dir}\\validation_results.csv")

counter = 1
for index, row in results.iterrows():
    img_path = f"{input_path}\\{counter}_second_out.jpg"
    x_min, y_min, x_max, y_max = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
    draw_bbox(img_path, x_min, y_min, x_max, y_max)
    counter += 1
