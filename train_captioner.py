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
from image_transforms import captioner_image_transform_index
from tokenizers import Tokenizer
from coco_loader import get_coco_loader, decode_predictions
from torch.amp import autocast, GradScaler
import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# My very own, in-house transformer implementation!!
from transformer_components_michaelkosmider import (
    TransformerEncoderDecoder,
    get_causal_mask,
)
from image_captioner import ImageEncoder, CaptionDecoder

if __name__ == "__main__":

    # TODO
    # unhardcode freeze blocks
    # unfreeze encoder

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
            image_encoder_state_dict = torch.load(
                pretrained_path, pickle_module=pickle
            )["image_encoder_state_dict"]
        else:
            print(f"The path {pretrained_path} does not exist!")
            sys.exit(1)

    # Get hyper-parameters.
    tokenizer = Tokenizer.from_file(paths["tokenizer"])
    SOS_IDX = tokenizer.token_to_id("<SOS>")
    EOS_IDX = tokenizer.token_to_id("<EOS>")
    PAD_IDX = tokenizer.token_to_id("<PAD>")

    with open(paths["encoder_config"], "r") as f:
        MAE_config = yaml.safe_load(f)

    with open(paths["captioner_config"], "r") as f:
        captioner_config = yaml.safe_load(f)

    BATCH_SIZE = captioner_config["batch_size"]
    NUM_WORKERS = captioner_config["num_workers"]

    VOCAB_SIZE = captioner_config["vocab_size"]
    CONTEXT_SIZE = captioner_config["context_size"]
    PATCH_SIZE = MAE_config["patch_size"]
    IMAGE_SIZE = MAE_config["image_size"]

    EVAL_FREQ = captioner_config["eval_freq"]
    TOTAL_EPOCHS = captioner_config["total_epochs"]
    CUR_EPOCHS = captioner_config["cur_epochs"]
    WARMUP_EPOCHS = captioner_config["warmup_epochs"]

    START_FACTOR = float(captioner_config["start_factor"])
    LEARNING_RATE = float(captioner_config["lr"])
    UNFREEZE_LR_FACTOR = float(captioner_config["unfreeze_lr_factor"])
    UNFREEZE_START_EPOCH = captioner_config["unfreeze_start_epoch"]
    ETA_MIN = float(captioner_config["eta_min"])
    WEIGHT_DECAY = float(captioner_config["weight_decay"])
    LABEL_SMOOTHING = float(captioner_config["label_smoothing"])

    LENGTH_ALPHA = float(captioner_config["length_alpha"])
    NUM_BEAMS = captioner_config["num_beams"]

    STACK_SIZE = captioner_config["image_encoder"]["stack_size"]

    image_encoder_config = MAE_config["image_encoder"]
    caption_decoder_config = captioner_config["caption_decoder"]

    # Set device.
    if "device" in captioner_config:
        device = captioner_config["device"]
    else:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    print(f"You are using {device}.")

    # =============================================================================
    # Section 2: Initialize training loop elements: model, optimizer, loss,
    #            training history, and data loaders. Also metrics (BLEU).
    # =============================================================================

    # Initialize model.
    model = TransformerEncoderDecoder(
        ImageEncoder(IMAGE_SIZE, PATCH_SIZE, image_encoder_config),
        CaptionDecoder(VOCAB_SIZE, CONTEXT_SIZE, caption_decoder_config),
    ).to(device)

    unfreeze_schedule = {}
    for i in range(STACK_SIZE):
        unfreeze_schedule[i + UNFREEZE_START_EPOCH] = {
            "name": f"Encoder Layer {STACK_SIZE - i - 1}",
            "sub_module": model.encoder.transformer_encoder.encoder_stack[
                STACK_SIZE - i - 1
            ],
        }
    unfreeze_schedule[STACK_SIZE + UNFREEZE_START_EPOCH] = {
        "name": "Patcher Embedder",
        "sub_module": model.encoder.patch_embedder,
    }

    # Initialize optimizer and scheduler.
    param_groups = (
        [
            {
                "params": model.encoder.patch_embedder.parameters(),
                "lr": UNFREEZE_LR_FACTOR ** (STACK_SIZE + 1) * LEARNING_RATE,
            }
        ]
        + [
            {
                "params": model.encoder.transformer_encoder.encoder_stack[
                    i
                ].parameters(),
                "lr": UNFREEZE_LR_FACTOR ** (STACK_SIZE - i) * LEARNING_RATE,
            }
            for i in range(STACK_SIZE)
        ]
        + [
            {
                "params": model.decoder.parameters(),
                "lr": LEARNING_RATE,
            }
        ]
    )
    optimizer = AdamW(
        param_groups,
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
    history = {
        "train_losses": [],
        "val_losses": [],
        "epochs_completed": 0,
        "bleu_scores": {},
        "learning_rates": [],
    }

    # Load all of the above in from checkpoint, if provided.
    # Alternatively, load just the encoder weights.
    if checkpoint is not None:
        print(f"Starting from checkpoint at {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint["history"]

    else:
        print(f"Loading in encoder weights from checkpoint at {pretrained_path}")
        model.encoder.load_state_dict(image_encoder_state_dict)

    # Feeze the layers that should still be frozen.
    for unfreeze_epoch, sub_module in unfreeze_schedule.items():
        if unfreeze_epoch < history["epochs_completed"]:
            # Don't freeze
            pass
        else:
            # Freeze
            name = sub_module["name"]
            print(f"Freezing module: {name}")
            for param in sub_module["sub_module"].parameters():
                param.requires_grad = False

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
            captioner_image_transform_index[split],
            NUM_WORKERS,
            CONTEXT_SIZE,
            PAD_IDX,
            pin_memory=True if device == "cuda" else False,
            small=True if args.small is not None else False,
            small_frac=args.small if args.small is not None else None,
        )

    loader_for_metrics = get_coco_loader(
        "val",
        BATCH_SIZE,
        captioner_image_transform_index["val"],
        NUM_WORKERS,
        mode="image_first",
    )

    # Get metrics for evaluation.
    bleu = evaluate.load("bleu")

    # =============================================================================
    # Section 3: Main training loop.
    # =============================================================================

    for epoch in range(
        history["epochs_completed"], history["epochs_completed"] + CUR_EPOCHS
    ):

        if epoch in unfreeze_schedule:
            sub_module = unfreeze_schedule[epoch]
            name = sub_module["name"]
            print(f"Unfreezing module: {name}")
            for param in sub_module["sub_module"].parameters():
                param.requires_grad = True

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
            train_batches.set_postfix({"loss": loss.item()})

        history["learning_rates"].append(list(scheduler.get_last_lr()))
        history["train_losses"].append(train_loss / train_token_count)

        scheduler.step()

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
        #     metric_batches = tqdm(
        #         loader_for_metrics,
        #         desc=f"Metrics for epoch {epoch+1}:",
        #         leave=True,
        #     )
        #     with torch.no_grad():
        #         decoded_preds = []
        #         decoded_refs = []
        #         model.eval()
        #         for img, references, _ in metric_batches:
        #             img = img.to(device)
        #             pred = model.generate(
        #                 img,
        #                 None,
        #                 NUM_BEAMS,
        #                 CONTEXT_SIZE,
        #                 LENGTH_ALPHA,
        #                 SOS_IDX,
        #                 PAD_IDX,
        #                 EOS_IDX,
        #             )
        #             decoded_preds.extend(decode_predictions(pred, tokenizer))
        #             decoded_refs.extend([tokenizer.decode_batch(ref) for ref in references])

        #         # Remove any empty predictions. Record percentage.
        #         cleaned_preds = []
        #         cleaned_refs = []
        #         for pred, ref in zip(decoded_preds, decoded_refs):
        #             if pred.strip():
        #                 cleaned_preds.append(pred)
        #                 cleaned_refs.append(ref)

        #         if len(cleaned_preds) == 0:
        #             print("Unable to compute BLEU: no non-empty predictions.")
        #         else:
        #             bleu_score = bleu.compute(
        #                 predictions=cleaned_preds, references=cleaned_refs
        #             )
        #             percentage_used = 100 * len(cleaned_preds) / len(decoded_preds)
        #             history["bleu_scores"][epoch + 1] = {
        #                 "score": bleu_score["bleu"],
        #                 "percentage_used": percentage_used,
        #             }

        # Checkpoint
        history["epochs_completed"] += 1
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
        }
        epochs_completed = history[
            "epochs_completed"
        ]  # Trouble putting this directly in the f string.
        checkpoint_path = os.path.join(
            paths["captioner_checkpoint"], f"checkpoint{epochs_completed}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
