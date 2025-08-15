from torchvision import transforms
import torch
from torchvision.transforms.functional import to_pil_image

__all__ = [
    "encoder_image_transform_index",
    "captioner_image_transform_index",
    "mean",
    "std",
    "to_rgb",
]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

encoder_train_image_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

captioner_train_image_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

val_image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

encoder_image_transform_index = {
    "train": encoder_train_image_transform,
    "val": val_image_transform,
}

captioner_image_transform_index = {
    "train": captioner_train_image_transform,
    "val": val_image_transform,
}


def to_rgb(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    img = img.clamp(0, 1)  # ensure values are within [0,1]
    img = (img * 255).byte()
    img = to_pil_image(img)
    return img
