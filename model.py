"""Character tokenisation and a decoder-only Transformer built from PyTorch layers."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class CharacterTokenizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.indices = {character: index for index, character in enumerate(alphabet, 1)}

    @property
    def vocab_size(self):
        return len(self.alphabet) + 1

    def encode(self, text):
        return [self.indices.get(character, 0) for character in text]

    def decode(self, tokens):
        return "".join(self.alphabet[token - 1] if token else "�" for token in tokens)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context: int = 128
    width: int = 64
    heads: int = 4
    layers: int = 2


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.project = nn.Linear(config.width, config.width)
        mask = torch.ones(config.context, config.context, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], -torch.inf)
        attended = scores.softmax(dim=-1) @ value
        return self.project(attended.transpose(1, 2).reshape(batch, length, width))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = CausalAttention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("Embedding width must divide evenly across attention heads.")
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.readout = nn.Linear(config.width, config.vocab_size, bias=False)

    def forward(self, tokens, targets=None):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.blocks(self.tokens(tokens) + self.positions(positions))
        logits = self.readout(self.norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.inference_mode()
    def generate(self, tokens, count, temperature, generator):
        for _ in range(count):
            logits, _ = self(tokens[:, -self.config.context:])
            probabilities = (logits[:, -1] / temperature).softmax(dim=-1)
            next_token = torch.multinomial(probabilities, 1, generator=generator)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    tokenizer = CharacterTokenizer(checkpoint["alphabet"])
    model = TinyTransformer(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), tokenizer
