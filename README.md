
# Segmentation Robustness Framework

![GitHub License](https://img.shields.io/github/license/wntic/segmentation-robustness-framework) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Segmentation Robustness Framework (SRF) is a tool for testing the robustness of segmentation models to digital adversarial attacks.

## Installation

### Requirements:
1. Python 3.9+
2. torch==2.4.0
3. torchvision==0.19.0

```bash
git clone https://github.com/wntic/segmentation-robustness-framework.git
cd segmentation-robustness-framework
# Create venv if needed
pip install -r requirements.txt
```

## Models

- **Fully Convolutional Network (FCN)** [[paper]](https://arxiv.org/abs/1605.06211v1)
	- Supported encoders: resnet50, resnet101.
- **DeepLabV3** [[paper]](https://arxiv.org/abs/1706.05587v3)
	- Supported encoders: resnet50, resnet101, mobilenet_v3_large.

## Attack Methods

### FGSM

FGSM stands for Fast Gradient Sign Method, a simple and widely used method for generating adversarial examples.

$$\textbf{adv}_{x} = x + \epsilon*\textbf{sign}(\nabla_xJ(\theta, x, y))$$

**Attack configuration**:
- **epsilon**: The magnitude of the perturbation for the attack.

#### PGD

The Projected Gradient Descent (PGD) attack is an iterative and more powerful variant of the FGSM.


$$X_{t+1} = X_{t} = \alpha * \text{sign}(\nabla_xJ(\theta, X_{t}, y))$$

$$X_{t+1} = \textbf{clip} (X_{t+1}, X_{0} - \epsilon, X_{0} + \epsilon)$$

## Usage

Easy to use in 3 lines of code. Just prepare the configuration file and start the process!

```python
from segmentation-robustness-framework import RobustnessEvaluation


config_path = "configs/config.yaml"
srf = RobustnessEvaluation(config_path=config_path)

srf.run()
```

## Configuring SRF
This section provides details on how to configure the tool for performing adversarial attacks on image segmentation models. The configuration is done using a YAML file. Basic configuration files are located in the `configs/` directory. Below is an example of a configuration file and a detailed description of its parameters:
```yaml
model:
  name: "FCN"
  encoder: "resnet101"
  weights: "coco_with_voc_labels"
  num_classes: 21
  device: "cuda"

attacks:
  - name: "FGSM"
    epsilon: [0.25]
  - name: "PGD"
    epsilon: [0.05, 0.125, 0.25]
    alpha: [0.007]
    steps: 40
    targeted: true
    target_label: 12

dataset:
  name: "PascalVOC"
  root: "path/to/datasets/VOCdevkit/VOC2012/"
  split: "val"
  image_shape: [512, 256]
  max_images: 500

output:
  save_dir: "./input/"
  save_images: true
  save_log: true
```

### Model Configuration

This section specifies the segmentation model to use, including the model name, encoder, weights, number of classes, and the device (e.g., CPU or GPU) for processing.

1. **name**: The name of the segmentation model.
	- Available Models: `FCN`, `DeepLabV3`
2. **encoder**: The encoder backbone to be used within the model.
	- Available Encoders for FCN: `resnet50`, `resnet101`
	- Available Encoders for DeepLabV3: `resnet50`, `resnet101`, `mobilenet_v3_large`
3. **weights**: Pre-trained weights to be loaded into the model.
	- *Note:* For FCN and DeepLabV3 only `coco_with_voc_labels` weights are supported.
4. **num_classes**: The number of classes, including the background.
5. **device**: The device on which the model will run (`cpu` or `cuda`).


### Attacks Configuration

This section lists the adversarial attacks to be applied to the model, along with their respective parameters. Two attack methods are available: FGSM and PGD. Each attack has its own set of parameters (see section [Attack Methods](#Attack Methods))

You can use multiple attacks at the same time, just put them in the config file.

1. **name**: The name of the adversarial attack method.
	- Available Attack Methods: FGSM, PGD
2. **epsilon**: The magnitude of the perturbation for the attack. This can be a list of values to test different perturbation strengths.
	- Values are specified as a list, for example `[0.1]` or `[0.125, 0.25, 0.5]`
3. **alpha**: The step size for iterative attacks like PGD.
	- Values are specified as a list, for example `[0.07]` or `[0.02, 0.07]`
4. **steps**: The number of iterations for iterative attacks like PGD.
5. **targeted**: A flag to indicate whether the attack is targeted or untargeted.
6. **target_label**: (Required if `targeted` is true) The target class label for a targeted attack.

### Dataset Configuration

This section specifies the dataset to be used for the segmentation task, including the dataset name, root directory, data split, image shape, and the maximum number of images to process.

There are four datasets available for selection, each with specific configurations.

When configuring a dataset, you can specify the size of the images to which they will be resized: `image_shape: [512, 256]`.

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
