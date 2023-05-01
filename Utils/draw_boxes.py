import cv2
import pandas as pd

def draw_bbox(image_path, x_min, y_min, x_max, y_max):
    # Load the image
    img = cv2.imread(image_path)
    # Draw the bounding box
    cv2.rectangle(img, (y_min, x_min), (y_max, x_max), (0, 255, 0), 2)
    # Display the image
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


input_path = "C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\\Deep Learning\\Classification and Localization of Inpainted Regions\\DeepFillv2_Pytorch\\results"
#file = pd.read_csv("C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Data\\annotations_file.csv")
# x_min, y_min, x_max, y_max = int(file.iloc[0,6]), int(file.iloc[0,7]), int(file.iloc[0,8]), int(file.iloc[0,9])
#
# print(x_min)
# print(y_min)
# print(x_max)
# print(y_max)


results = pd.read_csv("C:\\Users\\ANDRE\\OneDrive\\Desktop\\Andreas_Sideras\\Demokritos\\Msc in AI\\2nd Semester\Deep Learning\\Classification and Localization of Inpainted Regions\\Classification and Localization\\Results\\results.csv")

counter=1
for index, row in results.iterrows():
    img_path = f"{input_path}\\{counter}_second_out.jpg"
    x_min, y_min, x_max, y_max = int(row['x_min']), int(row['y_min']), int(row['x_max']), int(row['y_max'])
    draw_bbox(img_path, x_min, y_min, x_max, y_max)
    counter+=1

