from PIL import Image
from torch import Tensor
from torchvision.transforms import transforms


def preprocess_image(image_path: str) -> Tensor:
    if not isinstance(image_path, str):
        raise TypeError(f"Expected type str, but got {type(image_path).__name__}")

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    h = (h // 8 + 1) * 8 if h % 8 != 0 else h
    w = (w // 8 + 1) * 8 if w % 8 != 0 else w

    preprocess = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image)
    image = image.unsqueeze(0)
    return image
