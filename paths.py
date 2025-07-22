__all__ = ["paths"]

import os

cwd = os.getcwd()
DATA_DIR = os.path.join(cwd, "data/coco")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
CONFIG_PATH = os.path.join(cwd, "config.yaml")
TOKENIZER_PATH = os.path.join(DATA_DIR, "coco_bpe_tokenizer.json")
ENCODER_CHECKPOINT_PATH = os.path.join(cwd, "checkpoints", "encoder")
CAPTIONER_CHECKPOINT_PATH = os.path.join(cwd, "checkpoints", "captioner")

images = {
    "train": os.path.join(DATA_DIR, "train2017"),
    "val": os.path.join(DATA_DIR, "val2017"),
}
captions = {
    "train": os.path.join(ANNOTATIONS_DIR, "captions_train2017.json"),
    "val": os.path.join(ANNOTATIONS_DIR, "captions_val2017.json"),
}
captions_tokenized = {
    "train": os.path.join(ANNOTATIONS_DIR, "train_tokenized.pt"),
    "val": os.path.join(ANNOTATIONS_DIR, "val_tokenized.pt"),
}

paths = {
    "data": DATA_DIR,
    "annotations": ANNOTATIONS_DIR,
    "config": CONFIG_PATH,
    "tokenizer": TOKENIZER_PATH,
    "encoder_checkpoint": ENCODER_CHECKPOINT_PATH,
    "captioner_checkpoint": CAPTIONER_CHECKPOINT_PATH,
    "images": images,
    "captions": captions,
    "captions_tokenized": captions_tokenized,
}
