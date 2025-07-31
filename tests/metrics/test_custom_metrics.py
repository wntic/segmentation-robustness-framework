from unittest.mock import patch

import pytest
from segmentation_robustness_framework.metrics.custom_metrics import (
    CUSTOM_METRICS_REGISTRY,
    get_custom_metric,
    list_custom_metrics,
    register_custom_metric,
)


def test_register_custom_metric_success():
    @register_custom_metric("test_metric")
    def test_metric(targets, preds):
        return 0.5

    assert "test_metric" in CUSTOM_METRICS_REGISTRY
    assert CUSTOM_METRICS_REGISTRY["test_metric"] == test_metric


def test_register_custom_metric_overwrites_existing():
    @register_custom_metric("overwrite_metric")
    def first_metric(targets, preds):
        return 0.3

    @register_custom_metric("overwrite_metric")
    def second_metric(targets, preds):
        return 0.7

    assert "overwrite_metric" in CUSTOM_METRICS_REGISTRY
    assert CUSTOM_METRICS_REGISTRY["overwrite_metric"] == second_metric


def test_register_custom_metric_decorator_returns_function():
    @register_custom_metric("return_test")
    def test_func(targets, preds):
        return 0.8

    result = test_func([1, 2], [1, 2])
    assert result == 0.8


def test_get_custom_metric_success():
    @register_custom_metric("retrieve_test")
    def retrieve_metric(targets, preds):
        return 0.9

    metric_func = get_custom_metric("retrieve_test")
    assert metric_func == retrieve_metric
    assert metric_func([1, 2], [1, 2]) == 0.9


def test_get_custom_metric_raises_keyerror():
    with pytest.raises(KeyError, match="Custom metric 'non_existent' not found"):
        get_custom_metric("non_existent")


def test_get_custom_metric_error_message_includes_available_metrics():
    CUSTOM_METRICS_REGISTRY.clear()

    @register_custom_metric("available_metric")
    def available_metric(targets, preds):
        return 0.5

    with pytest.raises(KeyError, match="Available metrics: \\['available_metric'\\]"):
        get_custom_metric("missing_metric")


def test_list_custom_metrics_empty():
    CUSTOM_METRICS_REGISTRY.clear()

    metrics = list_custom_metrics()
    assert metrics == []


def test_list_custom_metrics_with_registered():
    CUSTOM_METRICS_REGISTRY.clear()

    @register_custom_metric("metric1")
    def metric1(targets, preds):
        return 0.1

    @register_custom_metric("metric2")
    def metric2(targets, preds):
        return 0.2

    metrics = list_custom_metrics()
    assert set(metrics) == {"metric1", "metric2"}


def test_list_custom_metrics_returns_copy():
    CUSTOM_METRICS_REGISTRY.clear()

    @register_custom_metric("copy_test")
    def copy_test_metric(targets, preds):
        return 0.5

    metrics_list = list_custom_metrics()
    metrics_list.append("should_not_affect_registry")

    registry_keys = list(CUSTOM_METRICS_REGISTRY.keys())
    assert "should_not_affect_registry" not in registry_keys


def test_register_custom_metric_logs_info():
    with patch("segmentation_robustness_framework.metrics.custom_metrics.logger") as mock_logger:

        @register_custom_metric("logging_test")
        def logging_test_metric(targets, preds):
            return 0.6

        mock_logger.info.assert_called_with("Registered custom metric: logging_test")


def test_custom_metric_function_signature():
    @register_custom_metric("simple_metric")
    def simple_metric(targets, preds):
        return len(targets)

    @register_custom_metric("complex_metric")
    def complex_metric(targets, preds, threshold=0.5, weight=1.0):
        return threshold * weight

    simple_result = get_custom_metric("simple_metric")([1, 2, 3], [1, 2, 3])
    complex_result = get_custom_metric("complex_metric")([1, 2], [1, 2], threshold=0.8, weight=2.0)

    assert simple_result == 3
    assert complex_result == 1.6


def test_custom_metric_with_different_return_types():
    @register_custom_metric("float_metric")
    def float_metric(targets, preds):
        return 0.75

    @register_custom_metric("int_metric")
    def int_metric(targets, preds):
        return 42

    @register_custom_metric("bool_metric")
    def bool_metric(targets, preds):
        return True

    float_result = get_custom_metric("float_metric")([1], [1])
    int_result = get_custom_metric("int_metric")([1], [1])
    bool_result = get_custom_metric("bool_metric")([1], [1])

    assert isinstance(float_result, float)
    assert isinstance(int_result, int)
    assert isinstance(bool_result, bool)


def test_register_custom_metric_with_empty_name():
    @register_custom_metric("")
    def empty_name_metric(targets, preds):
        return 0.0

    assert "" in CUSTOM_METRICS_REGISTRY
    assert CUSTOM_METRICS_REGISTRY[""] == empty_name_metric


def test_register_custom_metric_with_special_characters():
    @register_custom_metric("metric_with_underscores")
    def underscore_metric(targets, preds):
        return 0.1

    @register_custom_metric("metric-with-dashes")
    def dash_metric(targets, preds):
        return 0.2

    @register_custom_metric("metric123")
    def number_metric(targets, preds):
        return 0.3

    assert "metric_with_underscores" in CUSTOM_METRICS_REGISTRY
    assert "metric-with-dashes" in CUSTOM_METRICS_REGISTRY
    assert "metric123" in CUSTOM_METRICS_REGISTRY


def test_get_custom_metric_case_sensitive():
    @register_custom_metric("CaseSensitive")
    def case_sensitive_metric(targets, preds):
        return 0.5

    metric = get_custom_metric("CaseSensitive")
    assert metric == case_sensitive_metric

    with pytest.raises(KeyError, match="Custom metric 'casesensitive' not found"):
        get_custom_metric("casesensitive")


def test_custom_metric_registry_is_global():
    CUSTOM_METRICS_REGISTRY.clear()

    @register_custom_metric("global_test")
    def global_test_metric(targets, preds):
        return 0.8

    from segmentation_robustness_framework.metrics.custom_metrics import CUSTOM_METRICS_REGISTRY as imported_registry

    assert "global_test" in imported_registry
    assert imported_registry["global_test"] == global_test_metric


def test_register_custom_metric_with_lambda():
    lambda_metric = lambda targets, preds: sum(targets) / len(targets) if targets else 0.0  # noqa: E731

    CUSTOM_METRICS_REGISTRY["lambda_test"] = lambda_metric

    retrieved_metric = get_custom_metric("lambda_test")
    assert retrieved_metric == lambda_metric

    result = retrieved_metric([1, 2, 3], [1, 2, 3])
    assert result == 2.0


def test_custom_metric_with_exception():
    @register_custom_metric("exception_metric")
    def exception_metric(targets, preds):
        if len(targets) == 0:
            raise ValueError("Empty targets")
        return 0.5

    metric_func = get_custom_metric("exception_metric")

    result = metric_func([1, 2], [1, 2])
    assert result == 0.5

    with pytest.raises(ValueError, match="Empty targets"):
        metric_func([], [])


def test_register_custom_metric_preserves_function_metadata():
    @register_custom_metric("metadata_test")
    def metadata_test_metric(targets, preds):
        return 0.5

    registered_func = get_custom_metric("metadata_test")

    assert registered_func.__name__ == "metadata_test_metric"
    assert registered_func.__doc__ == "Test function with docstring."


def test_multiple_registrations_same_function():
    def shared_metric(targets, preds):
        return 0.9

    CUSTOM_METRICS_REGISTRY["name1"] = shared_metric
    CUSTOM_METRICS_REGISTRY["name2"] = shared_metric

    metric1 = get_custom_metric("name1")
    metric2 = get_custom_metric("name2")

    assert metric1 == metric2 == shared_metric
    assert metric1([1], [1]) == 0.9
    assert metric2([1], [1]) == 0.9
