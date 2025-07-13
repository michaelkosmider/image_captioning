import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from paths import paths
from functools import partial
import os

__all__ = ["get_coco_loader"]


class CocoDataset(Dataset):

    def __init__(self, images_path, annotations_path, image_transform):
        super().__init__()

        self.images_path = images_path
        self.annotations = torch.load(annotations_path, weights_only=False)

        self.image_transform = image_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        caption = self.annotations[idx]["caption"]
        caption_tensor = torch.tensor(caption, dtype=torch.long)

        image_id = self.annotations[idx]["image_id"]
        image_path = os.path.join(self.images_path, f"{image_id:012}.jpg")
        image = Image.open(image_path).convert("RGB")
        image_transformed = self.image_transform(image)

        return image_transformed, caption_tensor


def coco_collate_fn(batch, pad_idx):

    images = []
    captions = []

    for image, caption in batch:
        images.append(image)
        captions.append(caption)

    images_batch = torch.stack(images, dim=0)
    captions_batch = pad_sequence(captions, batch_first=True, padding_value=pad_idx)

    return images_batch, captions_batch


def get_coco_loader(split, batch_size, transform, pad_idx, num_workers):
    dataset = CocoDataset(
        paths["images"][split],
        paths["captions_tokenized"][split],
        transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(True if split == "train" else False),
        collate_fn=partial(coco_collate_fn, pad_idx=pad_idx),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader
