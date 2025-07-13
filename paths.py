__all__ = ["paths"]

import os

cwd = os.getcwd()
DATA_DIR = os.path.join(cwd, "data/coco")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
CONFIG_PATH = os.path.join(cwd, "config.yaml")
TOKENIZER_PATH = os.path.join(DATA_DIR, "coco_bpe_tokenizer.json")
TOKENIZER_INFO_PATH = os.path.join(DATA_DIR, "tokenizer_info.pt")
CHECKPOINT_PATH = os.path.join(cwd, "checkpoints")

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
    "tokenizer_info": TOKENIZER_INFO_PATH,
    "checkpoint": CHECKPOINT_PATH,
    "images": images,
    "captions": captions,
    "captions_tokenized": captions_tokenized,
}
