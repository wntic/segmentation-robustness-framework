import matplotlib.pyplot as plt
from torch import Tensor, argmax
import numpy as np


def visualize_segmentation(output: Tensor):
    pred_labels = argmax(output, dim=1)
    predicted_labels_np = pred_labels.squeeze().cpu().numpy()

    plt.imshow(predicted_labels_np)
    plt.title("Predicted Segmentation Map")
    plt.axis("off")
    plt.show()


def denormalize(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean
    return image


def show_image(original_image: Tensor, segmentation_mask: Tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
    image = denormalize(image, mean, std)
    image = np.clip(image, 0, 1)
    mask = segmentation_mask.cpu().detach().numpy()

    print(image.shape)
    print(mask.shape)

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask, cmap="tab20")
    plt.show()
