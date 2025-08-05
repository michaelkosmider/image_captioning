import argparse
import os
import sys
import pickle
import yaml
from tqdm import tqdm
from paths import paths
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from image_transforms import image_transform_index
from tokenizers import Tokenizer
from coco_loader import get_coco_loader, decode_predictions
from torch.amp import autocast, GradScaler
import evaluate

# My very own, in-house transformer implementation!!
from transformer_components import (
    TransformerEncoderDecoder,
    get_causal_mask,
)
from image_captioner import ImageEncoder, CaptionDecoder

# TODO
# unfreeze encoder
# val metrics


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
parser.add_argument(
    "-s",
    "--small",
    type=float,
    default=None,
    help="Fraction of dataset to use.",
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

# Get image encoder state dict if passed in and above not passed in.
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

SOS_IDX = tokenizer.token_to_id("<SOS>")
EOS_IDX = tokenizer.token_to_id("<EOS>")
PAD_IDX = tokenizer.token_to_id("<PAD>")

BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]

VOCAB_SIZE = config["vocab_size"]
CONTEXT_SIZE = config["context_size"]
PATCH_SIZE = config["patch_size"]
IMAGE_SIZE = config["image_size"]

EVAL_FREQ = config["eval_freq"]
TOTAL_EPOCHS = config["total_captioner_epochs"]
CUR_EPOCHS = config["cur_captioner_epochs"]
WARMUP_EPOCHS = int(config["captioner_warmup_epochs"])

START_FACTOR = float(config["captioner_start_factor"])
LEARNING_RATE = float(config["captioner_lr"])
ETA_MIN = float(config["captioner_eta_min"])
WEIGHT_DECAY = float(config["captioner_wd"])
LABEL_SMOOTHING = float(config["label_smoothing"])

LENGTH_ALPHA = float(config["length_alpha"])
NUM_BEAMS = config["num_beams"]

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

# Initialize optimizer and scheduler.
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

warmup = LinearLR(
    optimizer, start_factor=START_FACTOR, end_factor=1, total_iters=WARMUP_EPOCHS
)
cosine = CosineAnnealingLR(
    optimizer, T_max=TOTAL_EPOCHS - WARMUP_EPOCHS, eta_min=ETA_MIN
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[WARMUP_EPOCHS],
)

# Initialize grad scaler (helpful as we are using mixed precision training).
scaler = GradScaler(device=device)

# Initialize training history.
history = {
    "train_losses": [],
    "val_losses": [],
    "epochs_completed": 0,
    "bleu_scores": {},
    "cider_scores": {},
}

# Load all of the above from checkpoint if a checkpoint was passed in.
if checkpoint is not None:
    print(f"Starting from checkpoint at {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    history = checkpoint["history"]

# If checkpoint was not passed in but encoder weights were, then load those into encoder.
elif image_encoder_state_dict is not None:
    model.encoder.load_state_dict(image_encoder_state_dict)
    print(f"Loading in encoder weights from checkpoint at {pretrained_path}")

else:
    print(
        "Starting training from scratch. Did you forget to pass in a checkpoint or backbone?"
    )

# Initialize loss.
criterion = torch.nn.CrossEntropyLoss(
    ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING
)

# Get the dataloaders for train and val.
coco_loaders = {}

for split in ["train", "val"]:

    coco_loaders[split] = get_coco_loader(
        split,
        BATCH_SIZE,
        image_transform_index[split],
        NUM_WORKERS,
        CONTEXT_SIZE,
        PAD_IDX,
        small=True if args.small is not None else False,
        small_frac=args.small if args.small is not None else None,
    )

loader_for_metrics = get_coco_loader(
    "val",
    BATCH_SIZE,
    image_transform_index["val"],
    NUM_WORKERS,
    mode="image_first",
    drop_last=True,
)

# Get metrics for evaluation.
bleu = evaluate.load("bleu")

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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * batch_token_count
        train_batches.set_postfix(
            {"loss": loss.item(), "LR": scheduler.get_last_lr()[0]}
        )

    scheduler.step()

    history["train_losses"].append(train_loss / train_token_count)

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

        history["val_losses"].append(val_loss / val_token_count)

    # # Evaluate metrics
    # if epoch % EVAL_FREQ == 0:
    #     model.eval()
    #     with torch.no_grad():
    #         metric_batches = tqdm(
    #             loader_for_metrics, desc=f"Metrics for epoch {epoch+1}:", leave=True
    #         )
    #         all_preds, all_refs = [], []
    #         for img, references in metric_batches:
    #             img = img.to(device)

    #             # Get and decode predictions.
    #             predictions = model.generate(
    #                 img,
    #                 None,
    #                 NUM_BEAMS,
    #                 CONTEXT_SIZE,
    #                 LENGTH_ALPHA,
    #                 SOS_IDX,
    #                 PAD_IDX,
    #                 EOS_IDX,
    #             )
    #             decoded_predictions = decode_predictions(predictions, tokenizer)
    #             all_preds.extend(decoded_predictions)

    #             # Decode references.
    #             for ref_group in references:
    #                 decoded = tokenizer.decode_batch(
    #                     ref_group, skip_special_tokens=True
    #                 )
    #                 all_refs.append(decoded)

    #         cleaned_preds, cleaned_refs = [], []
    #         for pred, refs in zip(all_preds, all_refs):
    #             if not pred.strip():
    #                 continue
    #             cleaned_preds.append(pred)
    #             cleaned_refs.append(refs)
    #         # Compute BLEU
    #         bleu_score = bleu.compute(
    #             predictions=cleaned_preds, references=cleaned_refs
    #         )
    #         history["bleu_scores"][epoch] = bleu_score

    # Checkpoint
    history["epochs_completed"] += 1
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": history,
    }
    checkpoint_path = os.path.join(
        paths["captioner_checkpoint"], f"checkpoint{history["epochs_completed"]}.pt"
    )
    torch.save(checkpoint, checkpoint_path)
