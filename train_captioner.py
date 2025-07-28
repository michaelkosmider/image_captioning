import argparse
import os
import sys
import pickle
import yaml
from tqdm import tqdm
from paths import paths
import torch
from torch.optim import AdamW
from image_transforms import image_transform_index
from tokenizers import Tokenizer
from coco_loader import get_coco_loader
from torch.amp import autocast, GradScaler

# My very own, in-house transformer implementation!!
from transformer_components import (
    TransformerEncoderDecoder,
    get_causal_mask,
)
from image_captioner import ImageEncoder, CaptionDecoder

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint to resume training",
)
parser.add_argument(
    "-p",
    "--pretrained",
    type=str,
    default=None,
    help="Path to pretrained ViT encoder checkpoint",
)
args = parser.parse_args()
if (args.checkpoint is not None) and (args.pretrained is not None):
    print("Cannot train from checkpoint when pretrained is also passed in.")
    sys.exit(1)


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

# Get image encoder state dict if passed in.
image_encoder_state_dict = None
if args.pretrained is not None:
    pretrained_path = os.path.join(paths["encoder_checkpoint"], args.pretrained)

    if os.path.exists(pretrained_path):
        image_encoder_state_dict = torch.load(pretrained_path, pickle_module=pickle)[
            "image_encoder_state_dict"
        ]
    else:
        print(f"The path {pretrained_path} does not exist!")
        sys.exit(1)

# Get hyper-parameters.
tokenizer = Tokenizer.from_file(paths["tokenizer"])
with open(paths["config"], "r") as f:
    config = yaml.safe_load(f)

PAD_IDX = tokenizer.token_to_id("<PAD>")

VOCAB_SIZE = config["vocab_size"]
EPOCHS = int(config["captioner_epochs"])
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
CONTEXT_SIZE = config["context_size"]
PATCH_SIZE = config["patch_size"]
IMAGE_SIZE = config["image_size"]
LEARNING_RATE = float(config["captioner_lr"])
WEIGHT_DECAY = float(config["captioner_wd"])
image_encoder_config = config["image_encoder"]
caption_decoder_config = config["caption_decoder"]

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
# Section 2: Initialize training loop elements: model, optimizer, loss,
#            training history, and data loaders.
# =============================================================================

# Initialize model.
model = TransformerEncoderDecoder(
    ImageEncoder(IMAGE_SIZE, PATCH_SIZE, image_encoder_config),
    CaptionDecoder(VOCAB_SIZE, CONTEXT_SIZE, caption_decoder_config),
).to(device)

# Initialize optimizer.
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Initialize training history.
train_losses = []
val_losses = []
epochs_completed = 0

# Load all of the above from checkpoint if a checkpoint was passed in.
if checkpoint is not None:
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    epochs_completed = checkpoint["epochs_completed"]

# If checkpoint was not passed in but encoder weights were, then load those into encoder.
if image_encoder_state_dict is not None:
    model.encoder.load_state_dict(image_encoder_state_dict)
    print(f"Loading in encoder weights from checkpoint at {pretrained_path}")

# Initialize loss.
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

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

scaler = GradScaler(device=device)  # tmp
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
    for img, caption in train_batches:
        optimizer.zero_grad()

        img = img.to(device)
        caption = caption.to(device)

        labels = caption[:, 1:]
        caption_in = caption[:, :-1]

        with autocast(device_type=device):
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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
