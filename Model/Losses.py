import torch
from torch import nn


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


# bboxes1 = torch.tensor([[0, 0, 5, 5]])
# bboxes2 = torch.tensor([[10, 10, 15, 15]])
#
# criterion = IoC()
# print(criterion(bboxes1, bboxes2))
