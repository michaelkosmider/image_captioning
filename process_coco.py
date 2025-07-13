import torch
import os
from paths import paths

# Data processing
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

# Downloading COCO
import requests
import zipfile

# Output
from tqdm import tqdm

# =============================================================================
# Section 1: Download COCO dataset.
# =============================================================================


def download_and_extract(url, output_dir, expected_name):

    os.makedirs(output_dir, exist_ok=True)
    extract_path = os.path.join(output_dir, expected_name)

    # If zip already exists, skip download
    if not os.path.exists(extract_path):

        zip_path = os.path.join(output_dir, "tmp")

        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            desc=expected_name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)


# Initiate the download.
# Images
download_and_extract(
    "http://images.cocodataset.org/zips/train2017.zip", paths["data"], "train2017"
)
download_and_extract(
    "http://images.cocodataset.org/zips/val2017.zip", paths["data"], "val2017"
)

# Annotations
download_and_extract(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    paths["data"],
    "annotations",
)

# Remove data that isn't captions (the dataset consists of more than just the images and captions).
keep = {
    "captions_train2017.json",
    "captions_val2017.json",
    "train_tokenized.pt",
    "val_tokenized.pt",
}
for file in os.listdir(paths["annotations"]):
    if file not in keep:
        os.remove(os.path.join(paths["annotations"], file))

# =============================================================================
# Section 2: Create and train the tokenizer.
# =============================================================================

if os.path.exists(paths["tokenizer"]):
    tokenizer = Tokenizer.from_file(paths["tokenizer"])
else:

    # Define the tokenizer.
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Lowercase(), normalizers.NFD(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Get trainer and training data for tokenizer.
    trainer = trainers.BpeTrainer(
        vocab_size=5000, special_tokens=["<SOS>", "<EOS>", "<PAD>", "<UNK>"]
    )
    with open(paths["captions"]["train"], "r") as f:
        train_captions = [
            entry["caption"] for entry in json.load(f)["annotations"]
        ]  # List of every caption in the training set.

    # Train the tokenizer.
    tokenizer.train_from_iterator(train_captions, trainer=trainer)

    # Save the tokenizer.
    tokenizer.save(paths["tokenizer"])

    # Save special token inds.
    tokenizer_info = {
        "<SOS>": tokenizer.token_to_id("<SOS>"),
        "<EOS>": tokenizer.token_to_id("<EOS>"),
        "<PAD>": tokenizer.token_to_id("<PAD>"),
        "<UNK>": tokenizer.token_to_id("<UNK>"),
        "vocab_size": tokenizer.get_vocab_size(),
    }
    torch.save(tokenizer_info, paths["tokenizer_info"])

# =============================================================================
# Section 3: Tokenize all captions.
# =============================================================================

for split in ["train", "val"]:
    if not os.path.exists(paths["captions_tokenized"][split]):

        with open(paths["captions"][split], "r") as f:
            annotations = json.load(f)["annotations"]
            annotations_tokenized = []

            for annotation in annotations:
                caption_ids = (
                    [tokenizer_info["<SOS>"]]
                    + tokenizer.encode(annotation["caption"]).ids
                    + [tokenizer_info["<EOS>"]]
                )
                image_id = annotation["image_id"]
                annotation_tokenized = {"caption": caption_ids, "image_id": image_id}

                annotations_tokenized.append(annotation_tokenized)

            torch.save(annotations_tokenized, paths["captions_tokenized"][split])
