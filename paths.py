__all__ = ["paths"]

import os

cwd = os.getcwd()
DATA_DIR = os.path.join(cwd, "data/coco")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
ENCODER_CONFIG_PATH = os.path.join(cwd, "encoder_config.yaml")
TOKENIZER_PATH = os.path.join(DATA_DIR, "coco_bpe_tokenizer.json")
ENCODER_CHECKPOINT_PATH = os.path.join(cwd, "checkpoints", "encoder")
CAPTIONER_CHECKPOINT_PATH = os.path.join(cwd, "checkpoints", "captioner")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed")
RESULTS_PATH = os.path.join(cwd, "results.pt")

images = {
    "train": os.path.join(DATA_DIR, "train2017"),
    "unlabeled": os.path.join(DATA_DIR, "unlabeled2017"),
    "val": os.path.join(DATA_DIR, "val2017"),
}
captions = {
    "train": os.path.join(ANNOTATIONS_DIR, "captions_train2017.json"),
    "val": os.path.join(ANNOTATIONS_DIR, "captions_val2017.json"),
}
captions_tokenized = {
    "train": os.path.join(PROCESSED_PATH, "train_tokenized.pt"),
    "val": os.path.join(PROCESSED_PATH, "val_tokenized.pt"),
}
image_to_captions = {
    "train": os.path.join(PROCESSED_PATH, "train_im2capts.pt"),
    "val": os.path.join(PROCESSED_PATH, "val_im2capts.pt"),
}
image_ids = {
    "train": os.path.join(PROCESSED_PATH, "train_image_ids.pt"),
    "val": os.path.join(PROCESSED_PATH, "val_image_ids.pt"),
    "unlabeled": os.path.join(PROCESSED_PATH, "unlabeled_image_ids.pt"),
}

paths = {
    "data": DATA_DIR,
    "annotations": ANNOTATIONS_DIR,
    "encoder_config": ENCODER_CONFIG_PATH,
    "tokenizer": TOKENIZER_PATH,
    "encoder_checkpoint": ENCODER_CHECKPOINT_PATH,
    "captioner_checkpoint": CAPTIONER_CHECKPOINT_PATH,
    "results": RESULTS_PATH,
    "processed": PROCESSED_PATH,
    "images": images,
    "captions": captions,
    "captions_tokenized": captions_tokenized,
    "image_to_captions": image_to_captions,
    "image_ids": image_ids,
}
