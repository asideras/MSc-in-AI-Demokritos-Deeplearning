import argparse
import os
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import csv


def make_mask(image_path, masks_dir, original_masked_dir):
    # Load the image
    original_name, extension = os.path.splitext(os.path.basename(image_path))
    img = Image.open(image_path).convert('RGB')

    x = random.randint(30, 225)
    y = random.randint(30, 225)

    height = random.uniform(55, 105)
    width = random.uniform(55, 105)

    file_name = original_name.split('_')[-1]  # Get the file name from the original file name
    # x counts columns
    # y counts rows
    x_min = max(0, int(x - (width / 2)))
    y_min = max(0, int(y - (height / 2)))
    x_max = min(img.size[0], int(x + (width / 2)))
    y_max = min(img.size[1], int(y + (height / 2)))

    annotations = [file_name, 1, round(x / img.size[0], 2), round(y / img.size[1], 2), \
                   round(width / img.size[0], 2), round(height / img.size[1], 2), \
                   x_min, y_min, x_max, y_max]

    # Make the region white
    img_array = np.array(img)
    img_array[x_min:x_max, y_min:y_max, :] = 255
    img = Image.fromarray(img_array)

    # Create the binary mask
    mask_array = np.zeros(img.size[::-1], dtype=np.uint8)
    mask_array[x_min:x_max, y_min:y_max] = 255
    mask = Image.fromarray(mask_array, mode='L')

    # Save the masked image and mask
    masked_file_name = f'{file_name}.png'
    mask_file_name = f'{file_name}.png'
    img.save(os.path.join(original_masked_dir, masked_file_name))
    mask.save(os.path.join(masks_dir, mask_file_name))

    return annotations


current_dir = os.getcwd()

original_dir = os.path.join(current_dir, "../Data/original")
masks_dir = os.path.join(current_dir, "../Data/masks")
original_masked_dir = os.path.join(current_dir, "../Data/original_masked")
os.makedirs(masks_dir, exist_ok=True)
os.makedirs(original_masked_dir, exist_ok=True)
os.makedirs(original_dir, exist_ok=True)

header = ["id", "label", "x", "y", "width", "height", "x_min", 'y_min', 'x_max', 'y_max']
rows = []

with open("../Data/annotations_file.csv", 'w', newline='') as annotations_file:
    writer = csv.writer(annotations_file)
    writer.writerow(header)
    for filename in tqdm(sorted(os.listdir(original_dir)), desc='Processing files'):
        rows.append(make_mask(os.path.join(original_dir, filename), masks_dir, original_masked_dir))

    print(rows)
    for row in rows:
        writer.writerow(row)
