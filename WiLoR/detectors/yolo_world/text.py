from __future__ import annotations

import html
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn


from WiLoR.assets import weights_dir

DEFAULT_CLIP_TEXT_MODEL = weights_dir() / "yolo_world" / "clip-vit-base-patch32"


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _word_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


class CLIPBPETokenizer:
    """Small OpenAI-CLIP byte-pair tokenizer for YOLO-World prompts."""

    def __init__(self, model_dir: str | Path = DEFAULT_CLIP_TEXT_MODEL, context_length: int = 77) -> None:
        model_dir = Path(model_dir)
        self.context_length = int(context_length)
        self.encoder: dict[str, int] = json.loads((model_dir / "vocab.json").read_text())
        merges = (model_dir / "merges.txt").read_text().splitlines()
        if merges and merges[0].startswith("#version"):
            merges = merges[1:]
        self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(merges) if merge.strip()}
        self.byte_encoder = _bytes_to_unicode()
        self.cache: dict[str, str] = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.start_token_id = self.encoder["<|startoftext|>"]
        self.end_token_id = self.encoder["<|endoftext|>"]
        self.pad_token_id = 1
        self.token_pattern = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\w]+|[^\s\w]+",
            flags=re.IGNORECASE,
        )

    @staticmethod
    def _clean(text: str) -> str:
        text = html.unescape(html.unescape(text))
        return " ".join(text.strip().lower().split())

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        if not token:
            return ""
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _word_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _word_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode_text(self, text: str) -> list[int]:
        ids = [self.start_token_id]
        for token in re.findall(self.token_pattern, self._clean(text)):
            byte_token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[piece] for piece in self.bpe(byte_token).split(" ") if piece)
        ids.append(self.end_token_id)
        if len(ids) > self.context_length:
            ids = ids[: self.context_length]
            ids[-1] = self.end_token_id
        return ids

    def encode_batch(self, texts: Sequence[str], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [self.encode_text(text) for text in texts]
        input_ids = torch.full((len(rows), self.context_length), self.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(rows), self.context_length), dtype=torch.long, device=device)
        for row_idx, ids in enumerate(rows):
            length = len(ids)
            input_ids[row_idx, :length] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row_idx, :length] = 1
        return input_ids, attention_mask


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int = 49408, hidden_size: int = 512, max_positions: int = 77) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.register_buffer("position_ids", torch.arange(max_positions).expand((1, -1)), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        return self.token_embedding(input_ids) + self.position_embedding(self.position_ids[:, :seq_len])


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int = 512, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + attention_mask
        probs = scores.softmax(dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int = 512, intermediate_size: int = 2048) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = hidden_states * torch.sigmoid(1.702 * hidden_states)
        return self.fc2(hidden_states)


class CLIPEncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = CLIPAttention()
        self.layer_norm1 = nn.LayerNorm(512, eps=1e-5)
        self.mlp = CLIPMLP()
        self.layer_norm2 = nn.LayerNorm(512, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask)
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, num_layers: int = 12) -> None:
        super().__init__()
        self.layers = nn.ModuleList(CLIPEncoderLayer() for _ in range(num_layers))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class CLIPTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = CLIPTextEmbeddings()
        self.encoder = CLIPEncoder()
        self.final_layer_norm = nn.LayerNorm(512, eps=1e-5)

    @staticmethod
    def _attention_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        causal = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min, device=input_ids.device)
        causal = torch.triu(causal, diagonal=1).view(1, 1, seq_len, seq_len)
        padding = (1 - attention_mask[:, None, None, :].to(torch.float32)) * torch.finfo(torch.float32).min
        return causal + padding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        mask = self._attention_mask(input_ids, attention_mask).to(dtype=hidden_states.dtype)
        hidden_states = self.encoder(hidden_states, mask)
        return self.final_layer_norm(hidden_states)


class CLIPTextModelWithProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = CLIPTextModel()
        self.text_projection = nn.Linear(512, 512, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = self.text_model(input_ids, attention_mask)
        pooled = last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), input_ids.argmax(dim=-1)]
        return self.text_projection(pooled)


class YOLOWorldTextEncoder(nn.Module):
    """Torch-only CLIP text encoder loaded from the YOLO-World checkpoint."""

    def __init__(self, model_name: str | Path = DEFAULT_CLIP_TEXT_MODEL, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.tokenizer = CLIPBPETokenizer(model_name)
        self.model = CLIPTextModelWithProjection().eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def load_yolo_world_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        prefix = "backbone.text_model.model."
        local_state = OrderedDict(
            (key.removeprefix(prefix), value)
            for key, value in state_dict.items()
            if key.startswith(prefix)
        )
        incompatible = self.model.load_state_dict(local_state, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    @torch.inference_mode()
    def encode(self, prompts: list[str] | tuple[str, ...]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_ids, attention_mask = self.tokenizer.encode_batch(prompts, device=device)
        features = self.model(input_ids=input_ids, attention_mask=attention_mask)
        features = torch.nn.functional.normalize(features, dim=-1, p=2)
        return features.unsqueeze(0)
