import argparse

import torch
import torchattacks
from adversarial_segmentation_toolkit import attacks, models, preprocess_image, visualize_segmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="path to input image/video")
    args = vars(parser.parse_args())

    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.DeepLabV3(encoder_name="resnet50")
    model.eval().to(device)

    image = preprocess_image(args["input"]).to(device)

    output = model(image)
    labels = output.argmax(dim=1)
    target = torch.full(image.size()[2:4], 12, dtype=torch.long, device=device).unsqueeze(0)  # [batch_size, h, w]

    eps = 0.1
    output_file_name = f"{args['input'].split('/')[-1].split('.')[0]}"

    atk = torchattacks.PGD(model, eps, alpha=2 / 255, steps=10, random_start=True)
    # atk.set_mode_default()
    atk.set_mode_targeted_by_label(quiet=True)
    adv_image_torchattacks = atk(image, target)
    adv_output_torchattacks = model(adv_image_torchattacks)
    visualize_segmentation(adv_output_torchattacks, output_file_name + "_adv_output_torchattacks")

    atk = attacks.PGD(model, eps, targeted=True)
    adv_image = atk.attack(image, target)
    adv_output = model(adv_image)
    visualize_segmentation(adv_output, output_file_name + "_adv_output")
