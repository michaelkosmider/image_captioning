# Model Definition (using my very own transformer implementation defined in transformer_components!)
import torch
import torch.nn as nn
from transformer_components_michaelkosmider import (
    TransformerDecoder,
    TransformerEncoder,
)

__all__ = ["ImageEncoder", "ImageDecoder", "ImageAutoEncoder", "CaptionDecoder"]


class PatchEmbedding(nn.Module):

    def __init__(self, hidden_size, image_size, patch_size):
        super().__init__()

        self.image_to_patch_projections = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        num_patches = (image_size // patch_size) ** 2
        self.pos_encoding = nn.Embedding(num_patches, hidden_size)

    def forward(self, X, positions=None):
        X = self.image_to_patch_projections(X)
        X = X.flatten(-2, -1)
        X = X.transpose(1, 2)
        if positions is None:
            X = X + self.pos_encoding(torch.arange(X.shape[1], device=X.device))
        else:
            inds = positions.unsqueeze(-1).expand(-1, -1, X.shape[-1])
            X = torch.gather(X, dim=1, index=inds) + self.pos_encoding(positions)

        return X


class ImageEncoder(nn.Module):

    def __init__(self, image_size, patch_size, image_encoder_config):
        super().__init__()

        self.patch_embedder = PatchEmbedding(
            image_encoder_config["hidden_size"], image_size, patch_size
        )

        self.transformer_encoder = TransformerEncoder(**image_encoder_config)

    def forward(self, X, key_padding_mask=None, positions=None):

        X = self.patch_embedder(X, positions)
        X = self.transformer_encoder(X, key_padding_mask)

        return X


class ImageDecoder(nn.Module):

    def __init__(self, patch_size, num_patches, image_decoder_config):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches

        self.positional_encoding = nn.Embedding(
            num_patches, image_decoder_config["hidden_size"]
        )
        self.masked_patch_token = nn.Parameter(
            torch.zeros((1, 1, image_decoder_config["hidden_size"]))
        )
        nn.init.normal_(self.masked_patch_token)

        self.transformer_decoder = TransformerEncoder(
            **image_decoder_config
        )  # It's actually an encoder, because no cross attention used.

        self.project = nn.Linear(
            image_decoder_config["hidden_size"], 3 * patch_size * patch_size
        )

    def forward(self, encoded_unmasked_patches, positions):
        # Initialize the decoder input to the masked patch token everywhere. Then fill in
        # unmasked positions with their corresponding encoder output.
        encoded_all_patches = torch.expand_copy(
            self.masked_patch_token,
            (encoded_unmasked_patches.shape[0], self.num_patches, -1),
        )
        inds = positions.unsqueeze(-1).expand(
            -1, -1, encoded_unmasked_patches.shape[-1]
        )
        encoded_all_patches.scatter_(1, inds, encoded_unmasked_patches)

        X = encoded_all_patches + self.positional_encoding(
            torch.arange(self.num_patches, device=encoded_unmasked_patches.device)
        )
        X = self.transformer_decoder(X, key_padding_mask=None)
        X = self.project(X)

        return X


class ImageAutoEncoder(nn.Module):

    def __init__(self, ImageEncoder, ImageDecoder):
        super().__init__()

        self.image_encoder = ImageEncoder
        self.image_decoder = ImageDecoder

    def forward(self, image, unmasked_positions):

        encoded_unmasked_patches = self.image_encoder(
            image, positions=unmasked_positions
        )
        reconstructed_image_patches = self.image_decoder(
            encoded_unmasked_patches, positions=unmasked_positions
        )

        return reconstructed_image_patches


class CaptionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, caption_decoder_config):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, caption_decoder_config["hidden_size"])
        self.positional_encoding = nn.Embedding(
            context_size, caption_decoder_config["hidden_size"]
        )
        self.project = nn.Linear(
            caption_decoder_config["hidden_size"],
            vocab_size,
        )
        self.transformer_decoder = TransformerDecoder(**caption_decoder_config)

        # Store for generate function in TransformerEncoderDecoder
        self.key_size = caption_decoder_config["key_size"]
        self.value_size = caption_decoder_config["value_size"]
        self.num_heads = caption_decoder_config["num_heads"]
        self.vocab_size = vocab_size
        self.stack_size = caption_decoder_config["stack_size"]

    def forward(
        self,
        X_tgt,
        X_src,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        src_key_padding_mask=None,
        all_kv_cache=None,
        position=None,
    ):

        if all_kv_cache is not None:
            X_tgt = self.embedding(X_tgt) + self.positional_encoding(
                torch.tensor(position, device=X_tgt.device)
            )
        else:
            X_tgt = self.embedding(X_tgt) + self.positional_encoding(
                torch.arange(X_tgt.shape[1], device=X_tgt.device)
            )

        X_tgt = self.transformer_decoder(
            X_tgt,
            X_src,
            tgt_mask,
            tgt_key_padding_mask,
            src_key_padding_mask,
            all_kv_cache,
        )
        logits = self.project(X_tgt)

        return logits
