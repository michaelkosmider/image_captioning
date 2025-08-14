from image_captioner import ImageEncoder, ImageDecoder, ImageAutoEncoder
from coco_loader import get_coco_loader
from paths import paths
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from image_transforms import encoder_image_transform_index
from tqdm import tqdm
import os
import argparse
import pickle
import sys
from torch.amp import autocast, GradScaler

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )
    parser.add_argument(
        "-s",
        "--small",
        type=float,
        default=None,
        help="Fraction of dataset to use.",
    )
    args = parser.parse_args()

    # =============================================================================
    # Section 1: Read in model and training configuration.
    # =============================================================================

    # Get checkpoint if passed in.
    os.makedirs(paths["encoder_checkpoint"], exist_ok=True)
    checkpoint = None
    if args.checkpoint is not None:

        checkpoint_path = os.path.join(paths["encoder_checkpoint"], args.checkpoint)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, pickle_module=pickle)
        else:
            print(f"The path {checkpoint_path} does not exist!")
            sys.exit(1)

    # Get hyper-parameters and training parameters.
    with open(paths["encoder_config"]) as f:
        config = yaml.safe_load(f)

    IMAGE_SIZE = config["image_size"]
    PATCH_SIZE = config["patch_size"]
    MASKING_RATIO = config["masking_ratio"]
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    NUM_MASKED_PATCHES = int(NUM_PATCHES * MASKING_RATIO)
    NUM_WORKERS = config["num_workers"]
    BATCH_SIZE = config["batch_size"]
    TOTAL_EPOCHS = int(config["total_epochs"])
    CUR_EPOCHS = int(config["cur_epochs"])
    WARMUP_EPOCHS = int(config["warmup_epochs"])
    LEARNING_RATE = float(config["lr"])
    START_FACTOR = float(config["start_factor"])
    ETA_MIN = float(config["eta_min"])
    WEIGHT_DECAY = float(config["weight_decay"])

    image_encoder_config = config["image_encoder"]
    image_decoder_config = config["image_decoder"]

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
    # Section 2: Initialize training loop elements: model, optimizer, scheduler,
    #            scaler, training history, loss, and data loaders.
    # =============================================================================

    # Initialize model.
    model = ImageAutoEncoder(
        ImageEncoder(IMAGE_SIZE, PATCH_SIZE, image_encoder_config),
        ImageDecoder(PATCH_SIZE, NUM_PATCHES, image_decoder_config),
    ).to(device)

    # Initialize optimizer and scheduler.
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    cosine = CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=ETA_MIN
    )
    linear = LinearLR(
        optimizer,
        start_factor=START_FACTOR,
        end_factor=1,
        total_iters=WARMUP_EPOCHS,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[linear, cosine], milestones=[WARMUP_EPOCHS]
    )

    # Initialize grad scaler (helpful as we are using mixed precision training).
    scaler = GradScaler(device=device)

    # Initialize training history.
    history = {"train_losses": [], "val_losses": [], "epochs_completed": 0}

    # Load all of the above from checkpoint if a checkpoint was passed in.
    if checkpoint is not None:
        model.image_encoder.load_state_dict(checkpoint["image_encoder_state_dict"])
        model.image_decoder.load_state_dict(checkpoint["image_decoder_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Initialize loss.
    criterion = nn.MSELoss()
    patch_extracter = nn.Unfold(kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    # Get the dataloaders for train and val.
    coco_loaders = {}

    for split in ["train", "val"]:
        coco_loaders[split] = get_coco_loader(
            split,
            BATCH_SIZE,
            encoder_image_transform_index[split],
            NUM_WORKERS,
            drop_last=True,
            mode="image_only",
            include_unlabeled=(
                True if split == "train" and args.small is None else False
            ),
            pin_memory=True if device == "cuda" else False,
            small=True if args.small is not None else False,
            small_frac=args.small if args.small is not None else None,
        )

    # =============================================================================
    # Section 3: Main training loop.
    # =============================================================================

    for epoch in range(
        history["epochs_completed"], history["epochs_completed"] + CUR_EPOCHS
    ):

        # Train
        train_batches = tqdm(
            coco_loaders["train"], desc=f"Training epoch {epoch+1}:", leave=True
        )
        train_loss = 0

        model.train()
        for image in train_batches:
            optimizer.zero_grad()

            image = image.to(device)
            positions = torch.randint(
                0,
                NUM_PATCHES,
                (
                    image.shape[0],
                    NUM_PATCHES,
                ),
                device=image.device,
            )
            masked_positions = positions[:, :NUM_MASKED_PATCHES]
            unmasked_positions = positions[:, NUM_MASKED_PATCHES:]

            with autocast(device_type=device):
                # Create labels
                image_patches = patch_extracter(image).transpose(-1, -2)
                ground_inds = masked_positions.unsqueeze(-1).expand(
                    -1, -1, image_patches.shape[-1]
                )
                ground_masked_patches = torch.gather(
                    image_patches, dim=1, index=ground_inds
                )

                # Get predictions
                reconstructed_image_patches = model(image, unmasked_positions)
                pred_masked_patches = torch.gather(
                    reconstructed_image_patches, dim=1, index=ground_inds
                )

                # Compute loss and update weights
                loss = criterion(pred_masked_patches, ground_masked_patches)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_batches.set_postfix(
                {"loss": loss.item(), "LR": scheduler.get_last_lr()}
            )

        history["train_losses"].append(train_loss / len(train_batches))
        scheduler.step()

        # Validate
        with torch.no_grad():
            val_batches = tqdm(
                coco_loaders["val"], desc=f"Validation epoch {epoch+1}:", leave=True
            )
            val_loss = 0

            model.eval()
            for image in val_batches:
                image = image.to(device)
                positions = torch.randint(
                    0,
                    NUM_PATCHES,
                    (
                        image.shape[0],
                        NUM_PATCHES,
                    ),
                    device=image.device,
                )
                masked_positions = positions[:, :NUM_MASKED_PATCHES]
                unmasked_positions = positions[:, NUM_MASKED_PATCHES:]

                # Create labels
                image_patches = patch_extracter(image).transpose(-1, -2)
                ground_inds = masked_positions.unsqueeze(-1).expand(
                    -1, -1, image_patches.shape[-1]
                )
                ground_masked_patches = torch.gather(
                    image_patches, dim=1, index=ground_inds
                )

                # Get predictions
                reconstructed_image_patches = model(image, unmasked_positions)
                pred_masked_patches = torch.gather(
                    reconstructed_image_patches, dim=1, index=ground_inds
                )

                # Compute loss and update weights
                loss = criterion(pred_masked_patches, ground_masked_patches)

                val_loss += loss.item()
                val_batches.set_postfix({"loss": loss.item()})

            history["val_losses"].append(val_loss / len(val_batches))

        # Checkpoint
        history["epochs_completed"] += 1
        checkpoint = {
            "image_encoder_state_dict": model.image_encoder.state_dict(),
            "image_decoder_state_dict": model.image_decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
        }
        epochs_completed = history[
            "epochs_completed"
        ]  # Trouble putting this directly in the f string.
        checkpoint_path = os.path.join(
            paths["encoder_checkpoint"], f"checkpoint{epochs_completed}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
