import argparse
import os
import sys
import pickle
import yaml
from tqdm import tqdm
from paths import paths
import torch
from torch.optim import Adam
from image_transforms import image_transform_index
from tokenizers import Tokenizer
from coco_loader import get_coco_loader

# My very own, in-house transformer implementation!!
from transformer_components import (
    TransformerEncoderDecoder,
    get_causal_mask,
)
from image_captioner import ImageEncoder, CaptionDecoder

# Parse arguments (just the checkpoint for now).
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint to resume training",
)
args = parser.parse_args()

# =============================================================================
# Section 1: Read in model and training configuration.
# =============================================================================

# Get checkpoint if passed in.
os.makedirs(paths["captioner_checkpoint"], exist_ok=True)
checkpoint = None
if args.checkpoint is not None:
    checkpoint_path = os.path.join(paths["captioner_checkpoint"], args.checkpoint)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, pickle_module=pickle)
    else:
        print(f"The path {checkpoint_path} does not exist!")
        sys.exit(1)

# Get hyper-parameters.
tokenizer = Tokenizer.from_file(paths["tokenizer"])
with open(paths["config"], "r") as f:
    config = yaml.safe_load(f)

PAD_IDX = tokenizer.token_to_id("<PAD>")

VOCAB_SIZE = config["vocab_size"]
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
CONTEXT_SIZE = config["context_size"]
PATCH_SIZE = config["patch_size"]
IMAGE_SIZE = config["image_size"]
transformer_encoder_config = config["transformer_encoder"]
transformer_decoder_config = config["transformer_decoder"]

# Set device.
if "device" in config:
    device = config["device"]
else:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
print(f"You are using {device}.")

# =============================================================================
# Section 2: Initialize training loop elements: model, optimizer, criterion,
#            training history, and data loaders.
# =============================================================================

# Initialize model.
model = TransformerEncoderDecoder(
    ImageEncoder(IMAGE_SIZE, PATCH_SIZE, transformer_encoder_config),
    CaptionDecoder(VOCAB_SIZE, CONTEXT_SIZE, transformer_decoder_config),
).to(device)

# Initialize optimizer and loss.
optimizer = Adam(model.parameters(), 0.0001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Initialize training history.
train_losses = []
val_losses = []
epochs_completed = 0

# Load all of the above (except criterion) from checkpoint if a checkpoint was passed in.
if checkpoint is not None:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epochs_completed = checkpoint["epochs_completed"]
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

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

for epoch in range(epochs_completed, epochs_completed + EPOCHS):

    # Train
    train_batches = tqdm(
        coco_loaders["train"], desc=f"Training epoch {epoch+1}:", leave=True
    )
    train_loss = 0
    train_token_count = 0

    model.train()
    cap = 30
    for img, caption in train_batches:
        cap -= 1
        if cap == 0:
            break

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
    checkpoint_path = os.path.join(
        paths["captioner_checkpoint"], f"checkpoint{epochs_completed}.pt"
    )
    torch.save(checkpoint, checkpoint_path)
