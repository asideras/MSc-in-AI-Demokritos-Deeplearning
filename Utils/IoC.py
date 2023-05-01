import numpy as np
import yaml
import pandas as pd


def iou():
    """
    Calculate IoU between two bounding boxes represented as (xmin, ymin, xmax, ymax).
    """
    with open('../config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    annotations_file = data['ANNOTATIONS_FILE']
    results_file = f"{data['RESULTS_DIR']}\\COLAB_RESULTS.csv"

    annotations = pd.read_csv(annotations_file)
    results = pd.read_csv(results_file)
    res = []
    for index, row in results.iterrows():
        x_min, y_min, x_max, y_max = int(results.loc[index]['x_min']), int(results.loc[index]['y_min']), int(
            results.loc[index]['x_max']), int(results.loc[index]['y_max'])
        bbox_pred = [x_min, y_min, x_max, y_max]

        x_min, y_min, x_max, y_max = int(annotations.loc[index]['x_min']), int(annotations.loc[index]['y_min']), int(
            annotations.loc[index]['x_max']), int(annotations.loc[index]['y_max'])
        bbox_true = [x_min, y_min, x_max, y_max]

        xA = np.maximum(bbox_true[0], bbox_pred[0])
        yA = np.maximum(bbox_true[1], bbox_pred[1])
        xB = np.minimum(bbox_true[2], bbox_pred[2])
        yB = np.minimum(bbox_true[3], bbox_pred[3])
        intersection = max(0, xB - xA) * max(0, yB - yA)
        area_true = (bbox_true[2] - bbox_true[0]) * (bbox_true[3] - bbox_true[1])
        area_pred = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1])
        union = area_true + area_pred - intersection
        iou = intersection / union

        res.append(iou)
    return np.mean(res)


iou_score = iou()
print(iou_score)  # Output: 0.17
