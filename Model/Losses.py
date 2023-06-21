import torch
from torch import nn
from torch.nn import MSELoss
from torchvision.ops.boxes import _box_inter_union

def giou_loss(input_boxes, target_boxes, eps=1e-7):
    """
    Args:
        input_boxes: Tensor of shape (N, 4) or (4,).
        target_boxes: Tensor of shape (N, 4) or (4,).
        eps (float): small number to prevent division by zero
    """
    inter, union = _box_inter_union(input_boxes, target_boxes)
    iou = inter / union

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

    giou = iou - ((area_c - union) / (area_c + eps))

    loss = 1 - giou

    return loss.sum()

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        return torch.mean((torch.sum(torch.pow(input - target, 2), dim=1)))


class IoCLoss(nn.Module):
    def __init__(self):
        super(IoCLoss, self).__init__()

    def forward(self, input, target):
        """
        Calculate Intersection over Union (IoU) for a batch of bounding boxes.

        Arguments:
        Returns:
        iou -- Intersection over Union (IoU) values, shape (num_of_batch,)
        """
        # Calculate the intersection coordinates
        x1 = torch.max(input[:, 0], target[:, 0])
        y1 = torch.max(input[:, 1], target[:, 1])
        x2 = torch.min(input[:, 2], target[:, 2])
        y2 = torch.min(input[:, 3], target[:, 3])

        # Calculate the intersection area
        intersection_area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)

        # Calculate the union area
        bbox1_area = (input[:, 2] - input[:, 0] + 1) * (input[:, 3] - input[:, 1] + 1)
        bbox2_area = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)

        union_area = bbox1_area + bbox2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / (union_area + 1e-6)

        iou = 1 - iou

        return torch.mean(iou)


# bboxes1 = torch.tensor([[10,10, 15, 15],[10,10,60,70]]).float()
# bboxes2 = torch.tensor([[20, 20, 30, 30],[100,100, 150, 150]]).float()
#
# criterion1 = L2Loss()
# print("Mean L2 distance: ",criterion1(bboxes1, bboxes2))
#
# criterion2 = MSELoss()
# print("MSELoss: ",criterion2(bboxes1, bboxes2))
#
# criterion3 = IoCLoss()
# print("Mean IoC loss: ",criterion3(bboxes1,bboxes2))
