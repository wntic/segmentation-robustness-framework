from PIL import Image
from torchvision.transforms import transforms


def preprocess_image(image_path: str):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)

    return image
