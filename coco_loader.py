import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from paths import paths
from functools import partial
import os

__all__ = ["get_coco_loader"]


class CaptFirstDataset(Dataset):

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


class ImgFirstDataset(Dataset):

    def __init__(self, images_path, im2capts_path, image_transform):
        super().__init__()

        self.images_path = images_path
        self.im2capts = torch.load(im2capts_path, weights_only=False)

        self.image_transform = image_transform

    def __len__(self):
        return len(self.im2capts)

    def __getitem__(self, idx):

        image_id, captions = self.im2capts[idx]
        image_path = os.path.join(self.images_path, f"{image_id:012}.jpg")
        image = Image.open(image_path).convert("RGB")
        image_transformed = self.image_transform(image)

        return image_transformed, captions


class ImageOnlyDataset(Dataset):

    def __init__(self, images_path, image_ids_path, image_transform):
        super().__init__()

        self.images_path = images_path
        self.image_ids = torch.load(image_ids_path, weights_only=False)

        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_path, f"{image_id:012}.jpg")
        image = Image.open(image_path).convert("RGB")
        image_transformed = self.image_transform(image)

        return image_transformed


def coco_collate_fn(batch, pad_idx, context_size):

    images = []
    captions = []

    for image, caption in batch:
        images.append(image)
        captions.append(caption[:context_size])  # Truncate if needed

    images_batch = torch.stack(images, dim=0)
    captions_batch = pad_sequence(captions, batch_first=True, padding_value=pad_idx)

    return images_batch, captions_batch


# mode can be one of: "caption_first", "image_first", "image_only"
# pass in context_size and pad_idx only if mode is "caption_first"
def get_coco_loader(
    split,
    batch_size,
    transform,
    num_workers,
    context_size=None,
    pad_idx=None,
    drop_last=False,
    small=False,
    small_frac=0.01,
    mode="caption_first",
):

    dataset = (
        CaptFirstDataset(
            paths["images"][split],
            paths["captions_tokenized"][split],
            transform,
        )
        if mode == "caption_first"
        else (
            ImgFirstDataset(
                paths["images"][split], paths["image_to_captions"][split], transform
            )
            if mode == "image_first"
            else ImageOnlyDataset(
                paths["images"][split], paths["image_ids"][split], transform
            )
        )
    )

    # Prune dataset if small is True
    if small:
        num_samples = int(small_frac * len(dataset))
        inds = torch.randperm(len(dataset))[:num_samples]
        dataset = Subset(dataset, inds)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(True if split == "train" else False),
        collate_fn=(
            partial(coco_collate_fn, pad_idx=pad_idx, context_size=context_size)
            if mode == "caption_first"
            else None
        ),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return dataloader
