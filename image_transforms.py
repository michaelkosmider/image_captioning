from torchvision import transforms

__all__ = ["encoder_image_transform_index", "captioner_image_transform_index"]

encoder_train_image_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

captioner_train_image_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_image_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
