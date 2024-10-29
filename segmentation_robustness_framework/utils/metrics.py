from typing import Optional

import numpy as np
import torch


class MetricsCollection:
    """Implements metrics to evaluate the quality of multiclass semantic segmentation models.

    Attributes:
        targets (torch.Tensor): Ground truth segmentation mask [C, H, W], where each pixel value is the true class.
        preds (torch.Tensor): Predicted segmentation mask [C, H, W], where each pixel value is the predicted class.
        num_classes (int): The number of classes.
    """

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        """Segmentation metric initialization.

        Raises:
            TypeError: If `num_classes` is not an integer.
            ValueError: If `num_classes` is less than 2.
        """
        if not isinstance(num_classes, int):
            raise TypeError("The number of classes must be integer")
        if num_classes < 2:
            raise ValueError("The number of classes must be 2 or more")

        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def _preprocess_input_data(self, targets: torch.Tensor, preds: torch.Tensor) -> tuple[torch.Tensor]:
        """Processes input ground-truth and predicted masks.

        Args:
            targets (torch.Tensor): Ground-truth segmentation masks.
            preds (torch.Tensor): Predicted segmentation masks.

        Raises:
            TypeError: If `targets` is not a `torch.Tensor` or `numpy.ndarray`.
            TypeError: If `preds` is not a `torch.Tensor` or `numpy.ndarray`.

        Returns:
            tuple[torch.Tensor]: Ground-truth and prediction mask tensors.
        """
        if isinstance(targets, torch.Tensor):
            true_mask = targets.cpu().detach().numpy()
        elif isinstance(targets, np.ndarray):
            true_mask = targets
        else:
            raise TypeError("`targets` must be of type `torch.Tensor` or `numpy.ndarray`")

        if isinstance(preds, torch.Tensor):
            pred_mask = preds.cpu().detach().numpy()
        elif isinstance(preds, np.ndarray):
            pred_mask = preds
        else:
            raise TypeError("`preds` must be of type `torch.Tensor` or `numpy.ndarray`")

        return true_mask, pred_mask

    def mean_iou(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute mean Intersection over Union metric.

        Args:
            targets (torch.Tensor): Ground-truth segmentation masks.
            preds (torch.Tensor): Predicted segmentation masks.
            average (str): Type of averaging to use: "macro" or "micro". Defaults to "macro"

        Returns:
            float: mean IoU.
        """
        assert average in ["macro", "micro"]

        true_mask, pred_mask = self._preprocess_input_data(targets, preds)

        # Sums for micro average
        total_intersection = 0
        total_union = 0

        iou_scores = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            pred = (pred_mask == cls).astype(np.int32)
            true = (true_mask == cls).astype(np.int32)

            intersection = np.sum(pred * true)
            union = np.sum(pred) + np.sum(true) - intersection

            if np.sum(true) == 0 and np.sum(pred) == 0:
                iou = np.nan
            else:
                iou = intersection / union if union > 0 else 0.0
            iou_scores.append(round(iou, 3))

            total_intersection += intersection
            total_union += union

        if average == "macro":
            return round(np.nanmean(iou_scores), 3)
        if average == "micro":
            return round(total_intersection / total_union, 3) if total_union > 0 else 0.0

    def pixel_accuracy(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        """Compute pixel accuracy metric.

        Args:
            targets (torch.Tensor): Ground-truth segmentation masks.
            preds (torch.Tensor): Predicted segmentation masks.

        Returns:
            float: Pixel accuracy.
        """
        true_mask, pred_mask = self._preprocess_input_data(targets, preds)

        correct_pixels = (pred_mask == true_mask).sum()
        total_pixels = true_mask.size
        return round(correct_pixels / total_pixels, 3) if total_pixels > 0 else 0.0

    def precision(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute precision metric.

        Args:
            targets (torch.Tensor): Ground-truth segmentation masks.
            preds (torch.Tensor): Predicted segmentation masks.
            average (str): Type of averaging to use: "macro" or "micro". Defaults to "macro"

        Returns:
            float: Precision metric.
        """
        assert average in ["macro", "micro"]

        true_mask, pred_mask = self._preprocess_input_data(targets, preds)
        precision = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            pred_class = pred_mask == cls
            true_class = true_mask == cls

            true_positive = np.logical_and(pred_class, true_class).sum()
            false_positive = np.logical_and(pred_class, np.logical_not(true_class)).sum()

            if true_positive + false_positive == 0:
                precision[cls] = np.nan
            else:
                precision[cls] = true_positive / (true_positive + false_positive)

        if average == "macro":
            macro_precision = np.nanmean(precision) if not np.all(np.isnan(precision)) else 0.0
            return round(macro_precision, 3)

        if average == "micro":
            total_tp = (np.logical_and(pred_mask == true_mask, true_mask != -1)).sum()
            total_fp = (np.logical_and(pred_mask != true_mask, pred_mask != -1)).sum()
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            return round(micro_precision, 3)

    def recall(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute recall metric.

        Args:
            targets (torch.Tensor): Ground-truth segmentation masks.
            preds (torch.Tensor): Predicted segmentation masks.
            average (str): Type of averaging to use: "macro" or "micro". Defaults to "macro"

        Returns:
            float: Recall metric.
        """
        assert average in ["macro", "micro"]
        true_mask, pred_mask = self._preprocess_input_data(targets, preds)
        recall = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_class = pred_mask == cls
            true_class = true_mask == cls

            true_positive = np.logical_and(pred_class, true_class).sum()
            false_negative = np.logical_and(np.logical_not(pred_class), true_class).sum()

            if true_positive + false_negative == 0:
                recall[cls] = np.nan
            else:
                recall[cls] = true_positive / (true_positive + false_negative)

        if average == "macro":
            macro_recall = np.nanmean(recall) if not np.all(np.isnan(recall)) else 0.0
            return round(macro_recall, 3)

        if average == "micro":
            total_tp = (np.logical_and(pred_mask == true_mask, true_mask != -1)).sum()
            total_fn = (np.logical_and(pred_mask != true_mask, true_mask != -1)).sum()
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            return round(micro_recall, 3)

    def dice_score(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute dice score.

        Args:
            targets (torch.Tensor): Ground-truth segmentation masks.
            preds (torch.Tensor): Predicted segmentation masks.
            average (str): Type of averaging to use: "macro" or "micro". Defaults to "macro"

        Returns:
            float: Dice score.
        """
        assert average in ["macro", "micro"]

        true_mask, pred_mask = self._preprocess_input_data(targets, preds)
        dice_scores = np.zeros(self.num_classes)

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            pred_class = pred_mask == cls
            true_class = true_mask == cls

            intersection = np.logical_and(pred_class, true_class).sum()
            pred_sum = pred_class.sum()
            true_sum = true_class.sum()

            if pred_sum + true_sum == 0:
                dice_scores[cls] = np.nan
            else:
                dice_scores[cls] = 2 * intersection / (pred_sum + true_sum)

        if average == "macro":
            return round(np.nanmean(dice_scores), 3) if not np.all(np.isnan(dice_scores)) else 0.0

        if average == "micro":
            total_intersection = np.logical_and(pred_mask == true_mask, true_mask != -1).sum()
            total_pred_sum = (pred_mask != -1).sum()
            total_true_sum = (true_mask != -1).sum()

            return (
                round(2 * total_intersection / (total_pred_sum + total_true_sum), 3)
                if (total_pred_sum + total_true_sum) > 0
                else 0.0
            )
