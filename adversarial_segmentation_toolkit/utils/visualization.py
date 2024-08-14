import matplotlib.pyplot as plt
from torch import Tensor, argmax


def visualize_segmentation(output: Tensor, output_file_name: str):
    pred_labels = argmax(output, dim=1)
    predicted_labels_np = pred_labels.squeeze().cpu().numpy()

    plt.imshow(predicted_labels_np)
    plt.title("Predicted Segmentation Map")
    plt.axis("off")
    plt.savefig(f"{output_file_name}.jpg")
    plt.show()
