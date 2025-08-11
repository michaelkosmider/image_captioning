import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from paths import paths
from functools import partial
import os
import sys

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

        return image_transformed, captions, image_id


class ImageOnlyDataset(Dataset):

    def __init__(
        self,
        images_path,
        image_ids_path,
        image_transform,
        unlaleled_images_path=None,
        unlabeled_image_ids_path=None,
    ):
        super().__init__()

        self.images_path = images_path
        self.image_ids = torch.load(image_ids_path, weights_only=False)

        if unlabeled_image_ids_path is not None:
            self.unlabeled_start_idx = len(self.image_ids)
            self.image_ids.extend(
                torch.load(unlabeled_image_ids_path, weights_only=False)
            )
            self.unlabeled_images_path = unlaleled_images_path

        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        if hasattr(self, "unlabeled_images_path") and idx >= self.unlabeled_start_idx:
            image_path = os.path.join(self.unlabeled_images_path, f"{image_id:012}.jpg")
        else:
            image_path = os.path.join(self.images_path, f"{image_id:012}.jpg")

        image = Image.open(image_path).convert("RGB")
        image_transformed = self.image_transform(image)

        return image_transformed


def image_first_collate_fn(batch):
    images, captions, image_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(captions), list(image_ids)


def caption_first_collate_fn(batch, pad_idx, context_size):

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
    include_unlabeled=False,
    pin_memory=True,
):

    # Get the specified dataset.
    if mode == "caption_first":
        dataset = CaptFirstDataset(
            paths["images"][split],
            paths["captions_tokenized"][split],
            transform,
        )
    elif mode == "image_first":
        dataset = ImgFirstDataset(
            paths["images"][split], paths["image_to_captions"][split], transform
        )
    elif mode == "image_only" and not include_unlabeled:
        dataset = ImageOnlyDataset(
            paths["images"][split], paths["image_ids"][split], transform
        )
    elif mode == "image_only" and include_unlabeled:
        dataset = ImageOnlyDataset(
            paths["images"][split],
            paths["image_ids"][split],
            transform,
            paths["images"]["unlabeled"],
            paths["image_ids"]["unlabeled"],
        )
    else:
        dataset = None

    if not dataset:
        print(f"Invalid dataset specification:")
        print(f"mode: {mode}")
        print(f"include_unlabeled: {include_unlabeled}")
        sys.exit(1)

    # Prune dataset if small is True
    if small:
        num_samples = max(int(small_frac * len(dataset)), batch_size * 2)
        inds = torch.randperm(len(dataset))[:num_samples]
        dataset = Subset(dataset, inds)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(True if split == "train" else False),
        collate_fn=(
            partial(
                caption_first_collate_fn, pad_idx=pad_idx, context_size=context_size
            )
            if mode == "caption_first"
            else image_first_collate_fn if mode == "image_first" else None
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


def decode_predictions(preds, tokenizer):

    clean_preds = []
    for pred in preds:
        tokens = []
        for token in pred:
            if token == tokenizer.token_to_id("<EOS>"):
                break
            if token != tokenizer.token_to_id("<PAD>"):
                tokens.append(token.item())
        clean_preds.append(tokens)

    generated_texts = tokenizer.decode_batch(clean_preds, skip_special_tokens=True)
    return generated_texts
