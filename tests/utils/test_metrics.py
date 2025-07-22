import numpy as np
import pytest
import torch
from segmentation_robustness_framework.utils.metrics import MetricsCollection


def test_metrics_collection_initialization():
    metrics = MetricsCollection(num_classes=3)

    assert metrics.num_classes == 3
    assert metrics.ignore_index == 255


def test_metrics_collection_initialization_custom_ignore_index():
    metrics = MetricsCollection(num_classes=5, ignore_index=100)

    assert metrics.num_classes == 5
    assert metrics.ignore_index == 100


def test_metrics_collection_invalid_num_classes_type():
    with pytest.raises(TypeError, match="The number of classes must be integer"):
        MetricsCollection(num_classes="3")


def test_metrics_collection_invalid_num_classes_value():
    with pytest.raises(ValueError, match="The number of classes must be 2 or more"):
        MetricsCollection(num_classes=1)


def test_metrics_collection_invalid_num_classes_zero():
    with pytest.raises(ValueError, match="The number of classes must be 2 or more"):
        MetricsCollection(num_classes=0)


def test_metrics_collection_invalid_num_classes_negative():
    with pytest.raises(ValueError, match="The number of classes must be 2 or more"):
        MetricsCollection(num_classes=-1)


def test_preprocess_input_data_torch_tensors():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)

    true_mask, pred_mask = metrics._preprocess_input_data(targets, preds)

    assert isinstance(true_mask, np.ndarray)
    assert isinstance(pred_mask, np.ndarray)
    assert np.array_equal(true_mask, np.array([[0, 1], [1, 0]]))
    assert np.array_equal(pred_mask, np.array([[0, 1], [1, 1]]))


def test_preprocess_input_data_numpy_arrays():
    metrics = MetricsCollection(num_classes=3)

    targets = np.array([[0, 1], [1, 0]], dtype=np.int64)
    preds = np.array([[0, 1], [1, 1]], dtype=np.int64)

    true_mask, pred_mask = metrics._preprocess_input_data(targets, preds)

    assert isinstance(true_mask, np.ndarray)
    assert isinstance(pred_mask, np.ndarray)
    assert np.array_equal(true_mask, targets)
    assert np.array_equal(pred_mask, preds)


def test_preprocess_input_data_mixed_types():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = np.array([[0, 1], [1, 1]], dtype=np.int64)

    true_mask, pred_mask = metrics._preprocess_input_data(targets, preds)

    assert isinstance(true_mask, np.ndarray)
    assert isinstance(pred_mask, np.ndarray)
    assert np.array_equal(true_mask, np.array([[0, 1], [1, 0]]))
    assert np.array_equal(pred_mask, np.array([[0, 1], [1, 1]]))


def test_preprocess_input_data_invalid_targets_type():
    metrics = MetricsCollection(num_classes=3)

    targets = "invalid"
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    with pytest.raises(TypeError, match="`targets` must be of type `torch.Tensor` or `numpy.ndarray`"):
        metrics._preprocess_input_data(targets, preds)


def test_preprocess_input_data_invalid_preds_type():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = "invalid"

    with pytest.raises(TypeError, match="`preds` must be of type `torch.Tensor` or `numpy.ndarray`"):
        metrics._preprocess_input_data(targets, preds)


def test_mean_iou_perfect_prediction():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")

    assert iou == 1.0


def test_mean_iou_perfect_prediction_micro():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="micro")

    assert iou == 1.0


def test_mean_iou_no_overlap():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")

    assert iou == 0.0


def test_mean_iou_partial_overlap():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")

    # Class 0: intersection=2, union=4, IoU=2/4=0.5
    # Class 1: intersection=0, union=2, IoU=0
    # Class 2: both empty, ignored
    # Macro average: (0.5 + 0.0) / 2 = 0.25
    assert iou == 0.25


def test_mean_iou_with_ignore_index():
    metrics = MetricsCollection(num_classes=3, ignore_index=255)

    targets = torch.tensor([[0, 255], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")

    # Only classes 0 and 1 are considered (255 is ignored)
    # Class 0: intersection=1, union=2, IoU=0.5
    # Class 1: intersection=1, union=2, IoU=0.5
    # Macro average: (0.5 + 0.5) / 2 = 0.75
    assert iou == 0.75


def test_mean_iou_invalid_average():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    with pytest.raises(AssertionError):
        metrics.mean_iou(targets, preds, average="invalid")


def test_pixel_accuracy_perfect_prediction():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    accuracy = metrics.pixel_accuracy(targets, preds)

    assert accuracy == 1.0


def test_pixel_accuracy_no_correct_pixels():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)

    accuracy = metrics.pixel_accuracy(targets, preds)

    assert accuracy == 0.0


def test_pixel_accuracy_partial_correct():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 1]], dtype=torch.long)

    accuracy = metrics.pixel_accuracy(targets, preds)

    # 3 correct out of 4 pixels = 0.75
    assert accuracy == 0.75


def test_pixel_accuracy_empty_mask():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([], dtype=torch.long)
    preds = torch.tensor([], dtype=torch.long)

    accuracy = metrics.pixel_accuracy(targets, preds)

    assert accuracy == 0.0


def test_precision_perfect_prediction():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    precision = metrics.precision(targets, preds, average="macro")

    assert precision == 1.0


def test_precision_perfect_prediction_micro():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    precision = metrics.precision(targets, preds, average="micro")

    assert precision == 1.0


def test_precision_false_positives():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    precision = metrics.precision(targets, preds, average="macro")

    # Class 0: TP=2, FP=0, Precision=2/(2+0)=1.0
    # Class 1: TP=0, FP=2, Precision=0/(0+2)=0.0
    # Class 2: TP=0, FP=0, Precision=nan (ignored)
    # Macro average: (1.0 + 0.0) / 2 = 0.5
    assert precision == 0.5


def test_precision_with_ignore_index():
    metrics = MetricsCollection(num_classes=3, ignore_index=255)

    targets = torch.tensor([[0, 255], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    precision = metrics.precision(targets, preds, average="macro")

    # Only classes 0 and 1 are considered (255 is ignored)
    # Class 0: TP=1, FP=0, Precision=1/(1+0)=1.0
    # Class 1: TP=1, FP=0, Precision=1/(1+0)=1.0
    # Macro average: (1.0 + 1.0) / 2 = 0.75
    assert precision == 0.75


def test_precision_invalid_average():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    with pytest.raises(AssertionError):
        metrics.precision(targets, preds, average="invalid")


def test_recall_perfect_prediction():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    recall = metrics.recall(targets, preds, average="macro")

    assert recall == 1.0


def test_recall_perfect_prediction_micro():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    recall = metrics.recall(targets, preds, average="micro")

    assert recall == 1.0


def test_recall_false_negatives():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)

    recall = metrics.recall(targets, preds, average="macro")

    # Class 0: TP=2, FN=0, Recall=2/(2+0)=1.0
    # Class 1: TP=0, FN=2, Recall=0/(0+2)=0.0
    # Class 2: TP=0, FN=0, Recall=nan (ignored)
    # Macro average: (1.0 + 0.0) / 2 = 0.5
    assert recall == 0.5


def test_recall_with_ignore_index():
    metrics = MetricsCollection(num_classes=3, ignore_index=255)

    targets = torch.tensor([[0, 255], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)

    recall = metrics.recall(targets, preds, average="macro")

    # Only classes 0 and 1 are considered (255 is ignored)
    # Class 0: TP=1, FN=0, Recall=1/(1+0)=1.0
    # Class 1: TP=0, FN=1, Recall=0/(0+1)=0
    # Macro average: (1.0 + 0) / 2 = 0.5
    assert recall == 0.5


def test_recall_invalid_average():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    with pytest.raises(AssertionError):
        metrics.recall(targets, preds, average="invalid")


def test_dice_score_perfect_prediction():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    dice = metrics.dice_score(targets, preds, average="macro")

    assert dice == 1.0


def test_dice_score_perfect_prediction_micro():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    dice = metrics.dice_score(targets, preds, average="micro")

    assert dice == 1.0


def test_dice_score_no_overlap():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)

    dice = metrics.dice_score(targets, preds, average="macro")

    assert dice == 0.0


def test_dice_score_partial_overlap():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    dice = metrics.dice_score(targets, preds, average="macro")

    # Class 0: intersection=2, pred_sum=2, true_sum=4, Dice=2*2/(2+4)=0.667
    # Class 1: intersection=0, pred_sum=2, true_sum=0, Dice=2*0/(2+0)=0
    # Class 2: intersection=0, pred_sum=0, true_sum=0, Dice=nan (ignored)
    # Macro average: (0.667 + 0) / 2 = 0.333
    assert dice == 0.333


def test_dice_score_with_ignore_index():
    metrics = MetricsCollection(num_classes=3, ignore_index=255)

    targets = torch.tensor([[0, 255], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    dice = metrics.dice_score(targets, preds, average="macro")

    # Only classes 0 and 1 are considered (255 is ignored)
    # Class 0: intersection=1, pred_sum=1, true_sum=1, Dice=2*1/(1+1)=1.0
    # Class 1: intersection=1, pred_sum=1, true_sum=1, Dice=2*1/(1+1)=1.0
    # Macro average: (1.0 + 1.0) / 2 = 0.833
    assert dice == 0.833


def test_dice_score_invalid_average():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    with pytest.raises(AssertionError):
        metrics.dice_score(targets, preds, average="invalid")


def test_get_metric_with_averaging_mean_iou():
    metrics = MetricsCollection(num_classes=3)

    metric_func = metrics.get_metric_with_averaging("mean_iou", "macro")

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    result = metric_func(targets, preds)

    assert result == 1.0


def test_get_metric_with_averaging_precision():
    metrics = MetricsCollection(num_classes=3)

    metric_func = metrics.get_metric_with_averaging("precision", "micro")

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    result = metric_func(targets, preds)

    assert result == 1.0


def test_get_metric_with_averaging_recall():
    metrics = MetricsCollection(num_classes=3)

    metric_func = metrics.get_metric_with_averaging("recall", "macro")

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    result = metric_func(targets, preds)

    assert result == 1.0


def test_get_metric_with_averaging_dice_score():
    metrics = MetricsCollection(num_classes=3)

    metric_func = metrics.get_metric_with_averaging("dice_score", "micro")

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    result = metric_func(targets, preds)

    assert result == 1.0


def test_get_metric_with_averaging_pixel_accuracy():
    metrics = MetricsCollection(num_classes=3)

    metric_func = metrics.get_metric_with_averaging("pixel_accuracy")

    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    result = metric_func(targets, preds)

    assert result == 1.0


def test_get_metric_with_averaging_invalid_metric():
    metrics = MetricsCollection(num_classes=3)

    with pytest.raises(ValueError, match="Unsupported metric"):
        metrics.get_metric_with_averaging("invalid_metric", "macro")


def test_get_metric_with_averaging_invalid_average():
    metrics = MetricsCollection(num_classes=3)

    with pytest.raises(ValueError, match="average must be 'macro' or 'micro'"):
        metrics.get_metric_with_averaging("mean_iou", "invalid")


def test_get_all_metrics_with_averaging_include_pixel_accuracy():
    metrics = MetricsCollection(num_classes=3)

    metrics_list, metric_names = metrics.get_all_metrics_with_averaging(include_pixel_accuracy=True)

    expected_names = [
        "mean_iou_macro",
        "precision_macro",
        "recall_macro",
        "dice_score_macro",
        "pixel_accuracy",
        "mean_iou_micro",
        "precision_micro",
        "recall_micro",
        "dice_score_micro",
    ]

    assert len(metrics_list) == 9
    assert len(metric_names) == 9
    assert metric_names == expected_names

    # Test that all metrics are callable
    targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    for metric_func in metrics_list:
        result = metric_func(targets, preds)
        assert isinstance(result, float)


def test_get_all_metrics_with_averaging_exclude_pixel_accuracy():
    metrics = MetricsCollection(num_classes=3)

    metrics_list, metric_names = metrics.get_all_metrics_with_averaging(include_pixel_accuracy=False)

    expected_names = [
        "mean_iou_macro",
        "precision_macro",
        "recall_macro",
        "dice_score_macro",
        "mean_iou_micro",
        "precision_micro",
        "recall_micro",
        "dice_score_micro",
    ]

    assert len(metrics_list) == 8
    assert len(metric_names) == 8
    assert metric_names == expected_names


def test_metrics_with_large_masks():
    metrics = MetricsCollection(num_classes=5)

    targets = torch.randint(0, 5, (100, 100), dtype=torch.long)
    preds = torch.randint(0, 5, (100, 100), dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")
    accuracy = metrics.pixel_accuracy(targets, preds)
    precision = metrics.precision(targets, preds, average="macro")
    recall = metrics.recall(targets, preds, average="macro")
    dice = metrics.dice_score(targets, preds, average="macro")

    assert 0.0 <= iou <= 1.0
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= dice <= 1.0


def test_metrics_with_single_class():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.zeros((10, 10), dtype=torch.long)
    preds = torch.zeros((10, 10), dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")
    accuracy = metrics.pixel_accuracy(targets, preds)
    precision = metrics.precision(targets, preds, average="macro")
    recall = metrics.recall(targets, preds, average="macro")
    dice = metrics.dice_score(targets, preds, average="macro")

    assert iou == 1.0
    assert accuracy == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert dice == 1.0


def test_metrics_with_all_ignore_index():
    metrics = MetricsCollection(num_classes=3, ignore_index=255)

    targets = torch.full((10, 10), 255, dtype=torch.long)
    preds = torch.full((10, 10), 255, dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")
    accuracy = metrics.pixel_accuracy(targets, preds)
    precision = metrics.precision(targets, preds, average="macro")
    recall = metrics.recall(targets, preds, average="macro")
    dice = metrics.dice_score(targets, preds, average="macro")

    # When all pixels are ignored:
    # - IoU becomes nan (no valid classes to average)
    # - Pixel accuracy is 1.0 (all pixels match)
    # - Precision, recall, and dice become 0.0 (no valid predictions)
    assert np.isnan(iou)
    assert accuracy == 1.0  # Pixel accuracy considers all pixels as correct
    assert precision == 0.0
    assert recall == 0.0
    assert dice == 0.0


def test_metrics_edge_cases():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0]], dtype=torch.long)
    preds = torch.tensor([[1]], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")
    accuracy = metrics.pixel_accuracy(targets, preds)

    assert iou == 0.0
    assert accuracy == 0.0

    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    preds = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")
    accuracy = metrics.pixel_accuracy(targets, preds)

    assert 0.0 <= iou <= 1.0
    assert accuracy == 0.5


def test_metrics_rounding():
    metrics = MetricsCollection(num_classes=3)

    targets = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.long)
    preds = torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.long)

    iou = metrics.mean_iou(targets, preds, average="macro")
    accuracy = metrics.pixel_accuracy(targets, preds)
    precision = metrics.precision(targets, preds, average="macro")
    recall = metrics.recall(targets, preds, average="macro")
    dice = metrics.dice_score(targets, preds, average="macro")

    assert isinstance(iou, float)
    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(dice, float)

    assert round(iou, 3) == iou
    assert round(accuracy, 3) == accuracy
    assert round(precision, 3) == precision
    assert round(recall, 3) == recall
    assert round(dice, 3) == dice
