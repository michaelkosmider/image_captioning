# Model Definition (using my very own transformer implementation defined in transformer_components!)
import torch
import torch.nn as nn
from transformer_components import (
    TransformerDecoder,
    TransformerEncoder,
)


class PatchEmbedding(nn.Module):

    def __init__(self, hidden_size, image_size, patch_size):
        super().__init__()

        self.image_to_patch_projections = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        num_patches = (image_size // patch_size) ** 2
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

    def forward(self, x):
        x = self.image_to_patch_projections(x)
        x = x.flatten(-2, -1)
        x = x.transpose(1, 2)
        x = x + self.pos_encoding

        return x


class ImageEncoder(nn.Module):

    def __init__(self, image_size, patch_size, transformer_encoder_config):
        super().__init__()

        self.patch_embedder = PatchEmbedding(
            transformer_encoder_config["hidden_size"], image_size, patch_size
        )

        self.transformer_encoder = TransformerEncoder(**transformer_encoder_config)

    def forward(self, X, key_padding_mask):

        X = self.patch_embedder(X)
        X = self.transformer_encoder(X, key_padding_mask)

        return X


class CaptionDecoder(nn.Module):

    def __init__(self, vocab_size, context_size, transformer_decoder_config):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, transformer_decoder_config["hidden_size"]
        )
        self.positional_encoding = nn.Embedding(
            context_size, transformer_decoder_config["hidden_size"]
        )
        self.project = nn.Linear(
            transformer_decoder_config["hidden_size"],
            vocab_size,
        )
        self.transformer_decoder = TransformerDecoder(**transformer_decoder_config)

        # Store for generate function
        self.key_size = transformer_decoder_config["key_size"]
        self.value_size = transformer_decoder_config["value_size"]
        self.num_heads = transformer_decoder_config["num_heads"]
        self.vocab_size = vocab_size
        self.stack_size = transformer_decoder_config["stack_size"]

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
