import torch
import os
from paths import paths
from collections import defaultdict

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

    # If unzip already exists, skip download
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
download_and_extract(
    "http://images.cocodataset.org/zips/unlabeled2017.zip",
    paths["data"],
    "unlabeled2017",
)
download_and_extract(
    "http://images.cocodataset.org/zips/test2017.zip", paths["data"], "test2017"
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
    # TODO: fix hardcoded vocab size.
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

# =============================================================================
# Section 3: Tokenize all captions. Save seven data structures: a list of
# captions with image id (train/val), a dict of image ids to all 5 captions
# (train/val), an iterable over image ids (train/val), and a list of all
# unlabeled image ids.
# =============================================================================

os.makedirs(paths["processed"], exist_ok=True)
for split in ["train", "val"]:
    if (
        not os.path.exists(paths["captions_tokenized"][split])
        or not os.path.exists(paths["image_to_captions"][split])
        or not os.path.exists(paths["image_ids"][split])
    ):

        with open(paths["captions"][split], "r") as f:
            annotations = json.load(f)["annotations"]

            annotations_tokenized = []
            image_to_captions = defaultdict(list)

            for annotation in annotations:
                caption_ids = (
                    [tokenizer.token_to_id("<SOS>")]
                    + tokenizer.encode(annotation["caption"]).ids
                    + [tokenizer.token_to_id("<EOS>")]
                )
                image_id = annotation["image_id"]
                annotation_tokenized = {"caption": caption_ids, "image_id": image_id}

                annotations_tokenized.append(annotation_tokenized)
                image_to_captions[image_id].append(caption_ids)

            torch.save(annotations_tokenized, paths["captions_tokenized"][split])
            torch.save(
                list(image_to_captions.items()), paths["image_to_captions"][split]
            )
            torch.save(list(image_to_captions.keys()), paths["image_ids"][split])

# Get a list of unlabeled image_ids as well.
if not os.path.exists(paths["image_ids"]["unlabeled"]):
    unlabeled_filenames = os.listdir(paths["images"]["unlabeled"])
    image_ids = [int(f[:-4]) for f in unlabeled_filenames if f.endswith(".jpg")]
    torch.save(image_ids, paths["image_ids"]["unlabeled"])
