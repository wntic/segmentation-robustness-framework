import matplotlib.pyplot as plt

# Pascal VOC classes
voc_classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

# tab20 colormap
tab20_cmap = plt.get_cmap("tab20")
tab20_colors = tab20_cmap.colors[: len(voc_classes)]
