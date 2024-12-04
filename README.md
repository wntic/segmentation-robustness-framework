
# Segmentation Robustness Framework

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue) ![PyTorch Version](https://img.shields.io/badge/PyTorch-2.4.0+-ee4c2c?logo=pytorch)
 [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) ![GitHub License](https://img.shields.io/github/license/wntic/segmentation-robustness-framework)

**Segmentation Robustness Framework** (SRF) is a tool for testing the robustness of semantic segmentation models to digital adversarial attacks.

## Installation

### Requirements:

* Python >= 3.12
* PyTorch >= 2.4.0
* Torchvision >= 0.19.0

```bash
git clone https://github.com/wntic/segmentation-robustness-framework.git
cd segmentation-robustness-framework
pip install -r requirements.txt
```

## Models

SRF supports loading models from three sources. Segmentation models from `torchvision` and `segmentation_model.pytorch`[[GitHub]](https://github.com/qubvel-org/segmentation_models.pytorch/) libraries are integrated into the framework. To determine the source of model loading, the `origin` field (`torchvision` or `smp` respectively) must be specified in the configuration file. You can also test your own models with custom architectures. For this purpose, the `origin` field is omitted (or you can specify `origin: None`), and the model will be loaded from the `segmentation_robustness_framework.models` module.

Two models with corresponding encoders are integrated from torchvision:

- **Fully Convolutional Network (FCN)** [[paper]](https://arxiv.org/abs/1605.06211v1)
	- Supported encoders: resnet50, resnet101.
- **DeepLabV3** [[paper]](https://arxiv.org/abs/1706.05587v3)
	- Supported encoders: resnet50, resnet101, mobilenet_v3_large.

The `segmentation_model.pytorch` supports 10 different segmentation model architectures and over 500 encoders. You can select any architecture and encoder, train the model and test it with SRF by specifying the path to the model weights in the configuration file.

## Attack Methods

Supported attack methods for generating adversarial perturbations:

- **Fast Gradient Sign Method** (FGSM) [[paper]](https://arxiv.org/abs/1412.6572)
- **Projected Gradient Descent** (PGD) [[paper]](https://arxiv.org/abs/1706.06083)
- **Random Fast Gradient Sign Method** (R+FGSM) [[paper]](https://arxiv.org/abs/1705.07204)
- **PGD based on KL-Divergence loss** (TPGD) [[paper]](https://arxiv.org/abs/1901.08573)

## Usage

Easy to use in 3 lines of code. Just prepare the configuration file and start the attack evaluation process!

```python
from segmentation_robustness_framework.engine import RobustEngine

engine = RobustnessEvaluation**(config_path="configs/sample_config.yaml")
engine.run(save=True, show=True, metrics=["mean_iou", "recall_macro"])
```

## Configuring SRF

This section provides details on how to configure the tool for performing adversarial attacks on image segmentation models. The configuration is done using a YAML file. Basic configuration files are located in the `configs/` directory. Below is an example of a configuration file and a detailed description of its parameters:

```yaml
model:
  origin: torchvision
  name: DeepLabV3
  encoder: resnet101
  weights: default
  num_classes: 21
  device: cuda

attacks:
  - name: FGSM
    epsilon: [0.05, 0.125, 0.25, 0.5, 0.66]

  - name: PGD
    epsilon: [0.05, 0.37, 0.5]
    alpha: [0.003, 0.007]
    steps: 40
    targeted: true
    target_label: 15

dataset:
  name: VOC
  root: path/to/VOCdevkit/VOC2012/
  split: val
  image_shape: [512, 376]
  max_images: 100
```

### Model Configuration

This section specifies the segmentation model to use, including the model origin, name, encoder, weights, number of classes, and the device (e.g., cpu or cuda) for processing.

1. **origin**: Source of model download (`torchvision`, `smp` or `None`).
2. **name**: The name of the segmentation model.
3. **encoder**: The encoder backbone to be used within the model.
4. **weights**: Pre-trained weights to be loaded into the model.
5. **num_classes**: The number of classes, including the background.
6. **device**: The device on which the model will run (`cpu` or `cuda`).


### Attacks Configuration

This section lists the adversarial attacks to be applied to the model, along with their respective parameters.

You can use multiple attacks at the same time, just put them in the config file.

1. **name**: The name of the adversarial attack method.
	- Available Attack Methods: FGSM, R+FGSM, PGD, TPGD.
2. **epsilon**: The magnitude of the perturbation for the attack. This can be a list of values to test different perturbation strengths.
	- Values are specified as a list, for example `[0.1]` or `[0.125, 0.25, 0.5]`
3. **alpha**: The step size for iterative attacks like PGD.
	- Values are specified as a list, for example `[0.07]` or `[0.02, 0.07]`
4. **iters**: The number of iterations for iterative attacks like PGD.
5. **targeted**: A flag to indicate whether the attack is targeted or untargeted.
6. **target_label**: (Required if `targeted` is true) The target class label for a targeted attack.

### Dataset Configuration

This section specifies the dataset to be used for the segmentation task, including the dataset name, root directory, data split, image shape, and the maximum number of images to process.

There are four datasets available for selection, each with specific configurations.

Available datasets: ADE20K, StanfordBackground, VOC, Cityscapes.

When configuring a dataset, you can specify the size of the images to which they will be resized: `image_shape: [512, 256]`.

You can also specify the maximum number of images that will be processed: `max_images: 100`.

#### Pascal VOC

Pascal VOC contains 20 object categories including vehicles, household, animals, and other: aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, TV/monitor, bird, cat, cow, dog, horse, sheep, and person.

Configuration parameters:
1. **root**: Path to VOC2012 directory, for example `datasets/VOCdevkit/VOC2012/`
2. **split**: The data split to use.
	- Available Splits: `train`, `val`, `trainval`

#### Stanford Background Dataset

The Stanford Background dataset contains 715 RGB images and the corresponding label images. Images are approximately 240×320 pixels in size and pixels are classified into eight different categories:  sky, tree, road, grass, water, building, mountain, foreground, unknown.

Configuration parameters:
1. **root**: Path to dataset (direcoty with `images` and `labels_colored` directories).
2. **split**: The dataset does not have split sets of images. Do it yourself, for example with `random_split`, or use the entire dataset.

#### ADE20K

The ADE20K semantic segmentation dataset contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed.

Configuration parameters:
1. **root**: Path to dataset, for example `datasets/ADEChallengeData2016/`
2. **split**: The data split to use.
	- Available Splits: `train`, `val`

#### Cityscapes

Cityscapes is a large-scale database which focuses on semantic understanding of urban street scenes. It provides semantic, instance-wise, and dense pixel annotations for 30 classes grouped into 8 categories (flat surfaces, humans, vehicles, constructions, objects, nature, sky, and void). The dataset consists of around 5000 `fine` annotated images and 20000 `coarse` annotated ones.

Configuration parameters:

1. **root**: Path to Cityscapes dataset.
2. **split**: The data split to use.
	- Available Splits: `train`, `val`, `test`, `train_extra`
3. **mode**: The annotation quality mode.
	- Available Modes: `fine`, `coarse`
4. **target_type**: The type of target annotation.
	- Available Target Types: `instance`, `semantic`, `color`, `polygon`
	- Note: You can use one or more annotation types:
		- One target type: `target_type="semantic"`
		- Several target types: `target_type=["semantic", "instance"]`

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
