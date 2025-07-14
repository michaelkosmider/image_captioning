# This is an object obtaining the paths of various files. Peek inside paths.py for usage.
from paths import paths

import os
import torch
from torch.optim import Adam
import pickle
import yaml
from torchvision import transforms
from coco_loader import get_coco_loader
from tqdm import tqdm
import argparse

# Model Definition (my very own, in-house transformer implementation!!)
from transformer_components import (
    TransformerEncoderDecoder,
    get_causal_mask,
)
from image_captioner import ImageEncoder, CaptionDecoder

# =============================================================================
# Section 1: Read in model and training configuration.
# =============================================================================


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to checkpoint to resume training")
args = parser.parse_args()

tokenizer_info = torch.load(paths["tokenizer_info"], weights_only=False)
with open(paths["config"], "r") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = 64
NUM_WORKERS = config["num_workers"]
CONTEXT_SIZE = config["context_size"]
PATCH_SIZE = config["patch_size"]
IMAGE_SIZE = config["image_size"]
PAD_IDX = tokenizer_info["<PAD>"]
VOCAB_SIZE = tokenizer_info["vocab_size"]
transformer_encoder_config = config["transformer_encoder_config"]
transformer_decoder_config = config["transformer_decoder_config"]

# Set device.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"You are using {device}.")

# =============================================================================
# Section 2: Define images transforms and get dataloaders.
# =============================================================================

train_image_transform = transforms.Compose(
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

image_transform_index = {"train": train_image_transform, "val": val_image_transform}

# Get the dataloaders for train and val.
coco_loaders = {}

for split in ["train", "val"]:

    coco_loaders[split] = get_coco_loader(
        split,
        BATCH_SIZE,
        CONTEXT_SIZE,
        image_transform_index[split],
        PAD_IDX,
        NUM_WORKERS,
    )

# =============================================================================
# Section 3: Main training loop.
# =============================================================================

# Initialize model.
model = TransformerEncoderDecoder(
    ImageEncoder(IMAGE_SIZE, PATCH_SIZE, transformer_encoder_config),
    CaptionDecoder(VOCAB_SIZE, CONTEXT_SIZE, transformer_decoder_config),
).to(device)

# Initialize optimizer and loss.
optimizer = Adam(model.parameters(), 0.0001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Track progress.
train_losses = []
val_losses = []
epochs_completed = 0

if args.checkpoint is not None:
    checkpoint_path = os.path.join(paths["checkpoint"], args.checkpoint)
    checkpoint = torch.load(, pickle_module=pickle)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epochs_completed = checkpoint["epochs_completed"]
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
else:
    os.makedirs(paths["checkpoint"], exist_ok=True)

epochs = 1


for epoch in range(epochs_completed, epochs_completed + epochs):

    # Train
    train_batches = tqdm(
        coco_loaders["train"], desc=f"Training epoch {epoch+1}:", leave=True
    )
    train_loss = 0
    train_token_count = 0
    model.train()
    # Note: img and caption are batches, not single instances.
    for img, caption in train_batches:

        optimizer.zero_grad()

        img = img.to(device)
        caption = caption.to(device)

        labels = caption[:, 1:]
        caption_in = caption[:, :-1]

        logits = model(
            caption_in,
            img,
            tgt_mask=get_causal_mask(caption_in.shape[1], device=device),
            tgt_key_padding_mask=(caption_in == PAD_IDX),
            src_key_padding_mask=None,
        )

        loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

        batch_token_count = torch.sum(labels != PAD_IDX).item()
        train_token_count += batch_token_count
        train_loss += loss.item() * batch_token_count

        loss.backward()
        optimizer.step()

        train_batches.set_postfix({"loss": loss.item()})

    train_losses.append(train_loss / train_token_count)

    # Validate
    with torch.no_grad():
        val_batches = tqdm(
            coco_loaders["val"], desc=f"Validation epoch {epoch+1}:", leave=True
        )
        val_loss = 0
        val_token_count = 0

        model.eval()
        for img, caption in val_batches:

            img = img.to(device)
            caption = caption.to(device)

            labels = caption[:, 1:]
            caption_in = caption[:, :-1]

            logits = model(
                caption_in,
                img,
                tgt_mask=get_causal_mask(caption_in.shape[1], device=device),
                tgt_key_padding_mask=(caption_in == PAD_IDX),
                src_key_padding_mask=None,
            )

            loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

            batch_token_count = torch.sum(labels != PAD_IDX).item()
            val_token_count += batch_token_count
            val_loss += loss.item() * batch_token_count

            val_batches.set_postfix({"loss": loss.item()})

        val_losses.append(val_loss / val_token_count)

    # Checkpoint
    epochs_completed += 1
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs_completed": epochs_completed,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    checkpoint_path = os.path.join(paths["checkpoint"], f"checkpoint{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
