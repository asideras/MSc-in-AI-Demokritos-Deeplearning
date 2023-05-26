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

input_path = data['IMG_DIR_INPAINTED']
results_dir = data['RESULTS_DIR']
#results = pd.read_csv(f"{results_dir}\\test_results.csv")
results = pd.read_csv("C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Data\\anns_class_local.csv")



for index, row in results.iterrows():
    id = str(int(row.id))
    img_path = f"{input_path}\\{id}_second_out.jpg"
    fake_label ,x_min, y_min, x_max, y_max = row['fake_label'], int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
    if fake_label:
       draw_bbox(img_path, x_min, y_min, x_max, y_max)
    else:
        print(f"Image with id: {id} does not contain artificial part")
    print("---")