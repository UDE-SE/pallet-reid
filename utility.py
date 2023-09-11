import torchvision.transforms as T
from torch import nn as nn


def img_transform(img, is_eval=False):
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]

    transforms = []
    transforms += [T.ToTensor()]
    transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    if not is_eval:
        transforms += [T.RandomHorizontalFlip()]
        transforms += [T.GaussianBlur(kernel_size=3)]
        transforms += [T.RandomPerspective()]
    preprocess = T.Compose(transforms)
    return preprocess(img)


class EmbeddingHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.normalize(x, p=2)

    
class ClassifierHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc1(x)