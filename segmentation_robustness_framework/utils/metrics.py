from typing import Optional

import numpy as np
import torch


class SegmentationMetric:
    """Implements metrics to evaluate the quality of multiclass semantic segmentation models.

    Attributes:
        targets (torch.Tensor): Ground truth segmentation mask [C, H, W], where each pixel value is the true class.
        preds (torch.Tensor): Predicted segmentation mask [C, H, W], where each pixel value is the predicted class.
        num_classes (int): The number of classes.
    """

    def __init__(self, targets: torch.Tensor, preds: torch.Tensor, num_classes: int, ignore_index: int = 255) -> None:
        """Segmentation metric initialization.

        Raises:
            TypeError: If `targets` is not a `torch.Tensor` or `numpy.ndarray`.
            TypeError: If `preds` is not a `torch.Tensor` or `numpy.ndarray`.
            TypeError: If `num_classes` is not an integer.
            ValueError: If `num_classes` is less than 2.
        """
        if isinstance(targets, torch.Tensor):
            self.true_mask = targets.cpu().detach().numpy()
        elif isinstance(targets, np.ndarray):
            self.true_mask = targets
        else:
            raise TypeError("`targets` must be of type `torch.Tensor` or `numpy.ndarray`")

        if isinstance(preds, torch.Tensor):
            self.pred_mask = preds.cpu().detach().numpy()
        elif isinstance(preds, np.ndarray):
            self.pred_mask = preds
        else:
            raise TypeError("`preds` must be of type `torch.Tensor` or `numpy.ndarray`")

        if not isinstance(num_classes, int):
            raise TypeError("The number of classes must be integer")
        if num_classes < 2:
            raise ValueError("The number of classes must be 2 or more")
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def iou(self) -> list[Optional[float]]:
        """Compute the Intersection over Union (IoU) per class for semantic segmentation.

        Returns:
            list[Optional[float]]: List of IoUs for each class.
        """
        iou_scores = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred = (self.pred_mask == cls).astype(np.int32)
            true = (self.true_mask == cls).astype(np.int32)

            intersection = np.sum(pred * true)
            union = np.sum(pred) + np.sum(true) - intersection

            if np.sum(true) == 0 and np.sum(pred) == 0:
                iou = np.nan
            else:
                iou = intersection / union if union > 0 else 0.0
            iou_scores.append(round(iou, 3))

        return iou_scores

    def mean_iou(self) -> float:
        """Compute the mean Intersection over Union (IoU) for semantic segmentation.

        Returns:
            float: Mean IoU score.
        """
        iou_per_class = self.iou()
        mean_iou = np.nanmean(iou_per_class)
        return round(mean_iou, 3)

    def pixel_accuracy(self) -> float:
        """Compute pixel accuracy for semantic segmentation.

        Returns:
            float: Pixel accuracy score.
        """
        correct_pixels = (self.pred_mask == self.true_mask).sum()
        total_pixels = self.true_mask.size
        return round(correct_pixels / total_pixels, 3)

    def precision_recall_f1(self) -> tuple[dict[str, float]]:
        """Compute precision, recall, and F1 score for multiclass segmentation.

        Returns:
            tuple[dict[str, float]]: Precision, recall, F1 score for each class and overall.
        """
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        f1_score = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_class = self.pred_mask == cls
            true_class = self.true_mask == cls

            true_positive = np.logical_and(pred_class, true_class).sum()
            false_positive = np.logical_and(pred_class, np.logical_not(true_class)).sum()
            false_negative = np.logical_and(np.logical_not(pred_class), true_class).sum()

            if true_positive + false_positive == 0:
                precision[cls] = np.nan
            else:
                precision[cls] = true_positive / (true_positive + false_positive)

            if true_positive + false_negative == 0:
                recall[cls] = np.nan
            else:
                recall[cls] = true_positive / (true_positive + false_negative)

            if np.isnan(precision[cls]) or np.isnan(recall[cls]) or (precision[cls] + recall[cls]) == 0:
                f1_score[cls] = np.nan
            else:
                f1_score[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])

        # Macro Average
        if np.all(np.isnan(precision)):
            macro_precision = 0.0
        else:
            macro_precision = np.nanmean(precision)

        if np.all(np.isnan(recall)):
            macro_recall = 0.0
        else:
            macro_recall = np.nanmean(recall)

        if np.all(np.isnan(f1_score)):
            macro_f1 = 0.0
        else:
            macro_f1 = np.nanmean(f1_score)

        # Micro Average
        total_tp = (np.logical_and(self.pred_mask == self.true_mask, self.true_mask != -1)).sum()
        total_fp = (np.logical_and(self.pred_mask != self.true_mask, self.pred_mask != -1)).sum()
        total_fn = (np.logical_and(self.pred_mask != self.true_mask, self.true_mask != -1)).sum()

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        precision = {"macro": round(macro_precision, 3), "micro": round(micro_precision, 3)}
        recall = {"macro": round(macro_recall, 3), "micro": round(micro_recall, 3)}
        f1_score = {"macro": round(macro_f1, 3), "micro": round(micro_f1, 3)}

        return precision, recall, f1_score

    def dice_coefficient(self) -> dict[str, float]:
        """Compute Dice coefficient for multiclass segmentation.

        Returns:
            dict[str, float]: Dice coefficient for each class.
        """
        dice_scores = np.zeros(self.num_classes)
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_class = self.pred_mask == cls
            true_class = self.true_mask == cls

            intersection = np.logical_and(pred_class, true_class).sum()
            pred_sum = pred_class.sum()
            true_sum = true_class.sum()

            if pred_sum + true_sum == 0:
                dice_scores[cls] = np.nan
            else:
                dice_scores[cls] = 2 * intersection / (pred_sum + true_sum)

        # Macro Average
        if np.all(np.isnan(dice_scores)):
            macro_dice = 0.0
        else:
            macro_dice = np.nanmean(dice_scores)

        # Micro Average
        total_intersection = np.logical_and(self.pred_mask == self.true_mask, self.true_mask != -1).sum()
        total_pred_sum = (self.pred_mask != -1).sum()
        total_true_sum = (self.true_mask != -1).sum()

        micro_dice = (
            2 * total_intersection / (total_pred_sum + total_true_sum) if (total_pred_sum + total_true_sum) > 0 else 0.0
        )

        return {"macro": round(macro_dice, 3), "micro": round(micro_dice, 3)}
