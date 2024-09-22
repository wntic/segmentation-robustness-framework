import pytest
import torch
from segmentation_robustness_framework import models


class TestBaseModel:
    def test_base_model(self):
        base = models.base_model.SegmentationModel(encoder_name="resnet50", encoder_weights="imagenet", num_classes=21)

        assert base.model is None
        assert base.encoder_name == "resnet50"
        assert base.encoder_weights == "imagenet"
        assert base.num_classes == 21

    def test_forward(self):
        base = models.base_model.SegmentationModel(encoder_name="resnet50", encoder_weights="imagenet", num_classes=21)
        base.eval()

        with pytest.raises(NotImplementedError):
            x = torch.rand(1, 3, 5, 5)
            base(x)


class TestFCN:
    def test_resnet50_encoder(self, mocker):
        mock_fcn_resnet50 = mocker.patch("torchvision.models.segmentation.fcn_resnet50")
        mock_fcn_resnet50.return_value = mocker.Mock()

        fcn = models.fcn.FCN(encoder_name="resnet50")

        mock_fcn_resnet50.assert_called_once_with(weights=models.fcn.FCN_ENCODERS["resnet50"], num_classes=21)
        assert fcn.model == mock_fcn_resnet50.return_value

    def test_resnet101_encoder(self, mocker):
        mock_fcn_resnet101 = mocker.patch("torchvision.models.segmentation.fcn_resnet101")
        mock_fcn_resnet101.return_value = mocker.Mock()

        fcn = models.fcn.FCN(encoder_name="resnet101")

        mock_fcn_resnet101.assert_called_once_with(weights=models.fcn.FCN_ENCODERS["resnet101"], num_classes=21)
        assert fcn.model == mock_fcn_resnet101.return_value

    def test_invalid_encoder(self):
        with pytest.raises(ValueError, match='Encoder "invalid_encoder" is not supported for the FCN.'):
            model = models.fcn.FCN(encoder_name="invalid_encoder")

    def test_forward(self, mocker):
        mock_fcn_resnet50 = mocker.patch("torchvision.models.segmentation.fcn_resnet50")
        mock_model_output = {"out": torch.randn(1, 21, 224, 224)}
        mock_fcn_resnet50.return_value = mocker.Mock(return_value=mock_model_output)

        model = models.fcn.FCN(encoder_name="resnet50")
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model.forward(input_tensor)

        mock_fcn_resnet50.return_value.assert_called_once_with(input_tensor)
        assert torch.equal(output, mock_model_output["out"])


class TestDeepLabV3:
    def test_resnet50_encoder(self, mocker):
        mock_deeplabv3_resnet50 = mocker.patch("torchvision.models.segmentation.deeplabv3_resnet50")
        mock_deeplabv3_resnet50.return_value = mocker.Mock()

        deeplabv3 = models.deeplab.DeepLabV3(encoder_name="resnet50")

        mock_deeplabv3_resnet50.assert_called_once_with(
            weights=models.deeplab.DEEPLABV3_ENCODERS["resnet50"], num_classes=21
        )
        assert deeplabv3.model == mock_deeplabv3_resnet50.return_value

    def test_resnet101_encoder(self, mocker):
        mock_deeplabv3_resnet101 = mocker.patch("torchvision.models.segmentation.deeplabv3_resnet101")
        mock_deeplabv3_resnet101.return_value = mocker.Mock()

        deeplabv3 = models.deeplab.DeepLabV3(encoder_name="resnet101")

        mock_deeplabv3_resnet101.assert_called_once_with(
            weights=models.deeplab.DEEPLABV3_ENCODERS["resnet101"], num_classes=21
        )
        assert deeplabv3.model == mock_deeplabv3_resnet101.return_value

    def test_mobilenetv3_encoder(self, mocker):
        mock_deeplabv3_mobilenetv3 = mocker.patch("torchvision.models.segmentation.deeplabv3_mobilenet_v3_large")
        mock_deeplabv3_mobilenetv3.return_value = mocker.Mock()

        deeplabv3 = models.deeplab.DeepLabV3(encoder_name="mobilenet_v3_large")

        mock_deeplabv3_mobilenetv3.assert_called_once_with(
            weights=models.deeplab.DEEPLABV3_ENCODERS["mobilenet_v3_large"], num_classes=21
        )
        assert deeplabv3.model == mock_deeplabv3_mobilenetv3.return_value

    def test_invalid_encoder(self):
        with pytest.raises(ValueError, match='Encoder "invalid_encoder" is not supported for the DeepLabV3.'):
            model = models.deeplab.DeepLabV3(encoder_name="invalid_encoder")

    def test_forward(self, mocker):
        mock_deeplabv3_resnet50 = mocker.patch("torchvision.models.segmentation.deeplabv3_resnet50")
        mock_model_output = {"out": torch.randn(1, 21, 224, 224)}
        mock_deeplabv3_resnet50.return_value = mocker.Mock(return_value=mock_model_output)

        model = models.deeplab.DeepLabV3(encoder_name="resnet50")
        input_tensor = torch.randn(1, 3, 224, 224)
        output = model.forward(input_tensor)

        mock_deeplabv3_resnet50.return_value.assert_called_once_with(input_tensor)
        assert torch.equal(output, mock_model_output["out"])
