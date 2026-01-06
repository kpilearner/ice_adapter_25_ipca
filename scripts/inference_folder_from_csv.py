#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from diffusers import FluxFillPipeline
from PIL import Image

TRAIN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train"))
if TRAIN_ROOT not in sys.path:
    sys.path.insert(0, TRAIN_ROOT)

from src.flux.transformer import tranformer_forward
from src.flux.block import AdapterCrossAttention


DEFAULT_PROMPT_PREFIX = (
    "A diptych with two side-by-side images of the same scene. "
    "The left image is visible light. "
    "The right image is the corresponding infrared thermal image. "
    "Instruction: "
)

DEFAULT_INSTRUCTION = (
    "Transform the visible image into its infrared thermal counterpart while preserving scene structure and objects."
)


def _load_lora_weights(pipe: FluxFillPipeline, lora_path: str) -> None:
    p = Path(lora_path).expanduser()
    if p.is_file():
        parent = str(p.parent)
        weight_name = p.name
        try:
            pipe.load_lora_weights(parent, weight_name=weight_name)
        except TypeError:
            pipe.load_lora_weights(parent)
        return
    pipe.load_lora_weights(str(p))


def _extend_txt_ids(txt_ids: torch.Tensor, extra_tokens: int) -> torch.Tensor:
    if extra_tokens <= 0:
        return txt_ids
    if txt_ids.ndim == 3:
        base = txt_ids[0]
        add_batch_dim = True
    else:
        base = txt_ids
        add_batch_dim = False

    last = base[-1:].clone()
    extra = last.repeat(extra_tokens, 1)
    extra[:, 2] = extra[:, 2] + torch.arange(
        1, extra_tokens + 1, device=extra.device, dtype=extra.dtype
    )
    out = torch.cat([base, extra], dim=0)
    if add_batch_dim:
        out = out.unsqueeze(0)
    return out


class PooledPromptAdapter(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class TextTokenAdapter(torch.nn.Module):
    def __init__(
        self,
        pooled_dim: int,
        token_dim: int,
        num_tokens: int,
        hidden_dim: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.pooled_dim = int(pooled_dim)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.pooled_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, self.num_tokens * self.pooled_dim),
        )
        self.proj = torch.nn.Linear(self.pooled_dim, self.token_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        bsz = pooled.shape[0]
        tokens = self.mlp(pooled).view(bsz, self.num_tokens, self.pooled_dim)
        tokens = self.proj(tokens)
        return self.scale * tokens


class ThermalTokenAdapter(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        token_dim: int,
        num_tokens: int = 4,
        scale: float = 1.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)
        self.init_std = float(init_std)
        self.tokens = torch.nn.Parameter(torch.randn(self.num_tokens, self.input_dim) * self.init_std)
        self.proj = None
        if self.token_dim != self.input_dim:
            self.proj = torch.nn.Linear(self.input_dim, self.token_dim)

    def _ensure_proj(self, target_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if target_dim == self.token_dim:
            return
        self.proj = torch.nn.Linear(self.input_dim, int(target_dim)).to(device=device, dtype=dtype)
        self.token_dim = int(target_dim)

    def forward(self, batch_size: int, target_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if target_dim is not None and target_dim != self.token_dim:
            self._ensure_proj(target_dim, device=device, dtype=dtype)
        tokens = self.tokens
        if self.proj is not None:
            tokens = self.proj(tokens)
        tokens = tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return self.scale * tokens


class ThermalGate(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        mode: str = "global",
        heads: int = 8,
        hidden_dim: int = 0,
    ):
        super().__init__()
        mode = str(mode).lower()
        if mode not in ("global", "headwise"):
            raise ValueError(f"Unknown thermal gate mode: {mode}")
        self.mode = mode
        self.input_dim = int(input_dim)
        self.heads = int(heads)
        self.hidden_dim = int(hidden_dim)
        out_dim = 1 if self.mode == "global" else self.heads
        if self.hidden_dim > 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hidden_dim, out_dim),
            )
        else:
            self.net = torch.nn.Linear(self.input_dim, out_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(pooled))


class ThermalQueryGenerator(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        output_dim: int,
        num_queries: int = 4,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        scale: float = 1.0,
        init_std: float = 0.02,
        use_type_embed: bool = True,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be > 0")
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.query_dim = int(query_dim)
        self.output_dim = int(output_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.ffn_dim = int(ffn_dim)
        self.scale = float(scale)
        self.init_std = float(init_std)
        self.use_type_embed = bool(use_type_embed)

        self.query_embed = torch.nn.Parameter(torch.randn(self.num_queries, self.query_dim) * self.init_std)
        self.caption_proj = torch.nn.LazyLinear(self.query_dim)
        self.image_proj = torch.nn.LazyLinear(self.query_dim)

        if self.use_type_embed:
            self.caption_type = torch.nn.Parameter(torch.zeros(1, 1, self.query_dim))
            self.image_type = torch.nn.Parameter(torch.zeros(1, 1, self.query_dim))
        else:
            self.caption_type = None
            self.image_type = None

        self.cross_attn_layers = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(self.query_dim, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        self.self_attn_layers = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(self.query_dim, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        self.ffn_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.query_dim, self.ffn_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.ffn_dim, self.query_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm1 = torch.nn.ModuleList([torch.nn.LayerNorm(self.query_dim) for _ in range(self.num_layers)])
        self.norm2 = torch.nn.ModuleList([torch.nn.LayerNorm(self.query_dim) for _ in range(self.num_layers)])
        self.norm3 = torch.nn.ModuleList([torch.nn.LayerNorm(self.query_dim) for _ in range(self.num_layers)])

        self.out_proj = None
        if self.output_dim != self.query_dim:
            self.out_proj = torch.nn.Linear(self.query_dim, self.output_dim)

    def ensure_output_dim(self, target_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if int(target_dim) == self.output_dim:
            return
        self.out_proj = torch.nn.Linear(self.query_dim, int(target_dim)).to(device=device, dtype=dtype)
        self.output_dim = int(target_dim)

    def forward(
        self,
        image_tokens: torch.Tensor | None = None,
        caption_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_tokens is None and caption_tokens is None:
            raise ValueError("Either image_tokens or caption_tokens must be provided.")

        memory = []
        if caption_tokens is not None:
            cap = self.caption_proj(caption_tokens)
            if self.caption_type is not None:
                cap = cap + self.caption_type
            memory.append(cap)
        if image_tokens is not None:
            img = self.image_proj(image_tokens)
            if self.image_type is not None:
                img = img + self.image_type
            memory.append(img)
        memory = torch.cat(memory, dim=1)

        query = self.query_embed.unsqueeze(0).expand(memory.shape[0], -1, -1)
        for i in range(self.num_layers):
            attn_out, _ = self.cross_attn_layers[i](query, memory, memory, need_weights=False)
            query = self.norm1[i](query + attn_out)
            attn_out, _ = self.self_attn_layers[i](query, query, query, need_weights=False)
            query = self.norm2[i](query + attn_out)
            ffn_out = self.ffn_layers[i](query)
            query = self.norm3[i](query + ffn_out)

        if self.out_proj is not None:
            query = self.out_proj(query)
        return self.scale * query


def _flatten_image_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4:
        return tokens.permute(0, 2, 3, 1).reshape(tokens.shape[0], -1, tokens.shape[1])
    if tokens.ndim == 3:
        return tokens
    raise ValueError(f"Unexpected image token shape: {tuple(tokens.shape)}")


def _encode_image_tokens(pipe: FluxFillPipeline, image: Image.Image) -> torch.Tensor:
    image_tensor = pipe.image_processor.preprocess(
        image,
        height=image.height,
        width=image.width,
    )
    image_tensor = image_tensor.to(pipe.device).to(pipe.dtype)
    latents = pipe.vae.encode(image_tensor).latent_dist.sample()
    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    tokens = pipe._pack_latents(latents, *latents.shape)
    return _flatten_image_tokens(tokens)


def _pack_mask_tokens(pipe: FluxFillPipeline, mask_latents: torch.Tensor) -> torch.Tensor:
    if mask_latents.ndim == 4:
        if mask_latents.shape[1] != 1 and mask_latents.shape[-1] == 1:
            mask_latents = mask_latents.permute(0, 3, 1, 2)
        return pipe._pack_latents(mask_latents, *mask_latents.shape)
    if mask_latents.ndim == 3:
        return mask_latents
    raise ValueError(f"Unexpected mask latent shape: {tuple(mask_latents.shape)}")


def _encode_mask_tokens(
    pipe: FluxFillPipeline, image: Image.Image, mask: Image.Image
) -> torch.Tensor:
    image_tensor = pipe.image_processor.preprocess(
        image,
        height=image.height,
        width=image.width,
    )
    mask_tensor = pipe.mask_processor.preprocess(
        mask,
        height=image.height,
        width=image.width,
    )
    image_tensor = image_tensor.to(pipe.device).to(pipe.dtype)
    mask_tensor = mask_tensor.to(pipe.device).to(pipe.dtype)
    masked_image = image_tensor * (1 - mask_tensor)
    num_channels_latents = pipe.vae.config.latent_channels
    height, width = image_tensor.shape[-2:]
    device = pipe._execution_device
    mask_latents, _ = pipe.prepare_mask_latents(
        mask_tensor,
        masked_image,
        image_tensor.shape[0],
        num_channels_latents,
        1,
        height,
        width,
        pipe.dtype,
        device,
        None,
    )
    return _pack_mask_tokens(pipe, mask_latents)


def _normalize_forward_args(args, kwargs):
    names = [
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "timestep",
        "img_ids",
        "txt_ids",
        "guidance",
        "joint_attention_kwargs",
        "return_dict",
    ]
    for i, value in enumerate(args):
        if i < len(names):
            kwargs[names[i]] = value
    return kwargs


def _patch_transformer_forward(transformer, model_config):
    def _forward(self, *args, **kwargs):
        kwargs = _normalize_forward_args(args, kwargs)
        if "cross_attention_kwargs" in kwargs and "joint_attention_kwargs" not in kwargs:
            kwargs["joint_attention_kwargs"] = kwargs.pop("cross_attention_kwargs")
        adapter_tokens = getattr(self, "_adapter_tokens", None)
        adapter_token_mask = getattr(self, "_adapter_token_mask", None)
        return tranformer_forward(
            self,
            condition_latents=None,
            condition_ids=None,
            condition_type_ids=None,
            model_config=model_config,
            c_t=0,
            adapter_tokens=adapter_tokens,
            adapter_token_mask=adapter_token_mask,
            **kwargs,
        )

    transformer.forward = types.MethodType(_forward, transformer)


def _init_adapter_ca(transformer, state_list, ca_cfg):
    scale_init = float(ca_cfg.get("scale", 0.0))
    trainable_scale = bool(ca_cfg.get("trainable_scale", True))
    gate_cfg = ca_cfg.get("gate", {}) or {}
    gate_enabled = bool(gate_cfg.get("enabled", False))
    gate_mode = str(gate_cfg.get("mode", "headwise")).lower()
    gate_hidden_dim = int(gate_cfg.get("hidden_dim", 0))
    gate_init_bias = float(gate_cfg.get("init_bias", 0.0))
    modules = []
    for idx, block in enumerate(transformer.transformer_blocks):
        dim = int(getattr(block.attn.to_q, "in_features", block.attn.to_q.weight.shape[1]))
        heads = int(getattr(block.attn, "heads", 8))
        block.adapter_ca = AdapterCrossAttention(
            dim=dim,
            heads=heads,
            scale_init=scale_init,
            trainable_scale=trainable_scale,
            gate_mode=gate_mode if gate_enabled else None,
            gate_hidden_dim=gate_hidden_dim,
            gate_init_bias=gate_init_bias,
        )
        ref_weight = block.attn.to_q.weight
        block.adapter_ca.to(device=ref_weight.device, dtype=ref_weight.dtype)
        if isinstance(state_list, list) and idx < len(state_list):
            block.adapter_ca.load_state_dict(state_list[idx], strict=False)
        modules.append(block.adapter_ca)
    return modules


def _apply_gate(tokens: torch.Tensor, gate: torch.Tensor, mode: str, heads: int) -> torch.Tensor:
    if tokens is None:
        return tokens
    mode = str(mode).lower()
    if mode == "global":
        return tokens * gate.view(gate.shape[0], 1, 1)
    if mode == "headwise":
        bsz, num_tokens, dim = tokens.shape
        if dim % heads != 0:
            raise ValueError(f"Gate heads ({heads}) must divide token dim ({dim}).")
        head_dim = dim // heads
        tokens = tokens.view(bsz, num_tokens, heads, head_dim)
        tokens = tokens * gate.view(bsz, 1, heads, 1)
        return tokens.view(bsz, num_tokens, dim)
    raise ValueError(f"Unknown gate mode: {mode}")


def _infer_gate_dims(state_dict: dict) -> tuple[int, int]:
    if "net.weight" in state_dict:
        return int(state_dict["net.weight"].shape[1]), int(state_dict["net.weight"].shape[0])
    if "net.0.weight" in state_dict and "net.2.weight" in state_dict:
        return int(state_dict["net.0.weight"].shape[1]), int(state_dict["net.2.weight"].shape[0])
    raise ValueError("Unable to infer gate dimensions from state dict.")


def _ensure_module_on(module: torch.nn.Module, ref: torch.Tensor) -> None:
    if module is None:
        return
    if next(module.parameters()).device != ref.device or next(module.parameters()).dtype != ref.dtype:
        module.to(device=ref.device, dtype=ref.dtype)


def _seed_for(path: Path, seed: int, mode: str) -> int:
    if mode == "fixed":
        return int(seed)
    if mode == "incremental":
        return int(seed)
    data = path.as_posix().encode("utf-8")
    return int(hashlib.sha256(data).hexdigest()[:8], 16)


def _resize_image(image: Image.Image, size: int, mode: str) -> Image.Image:
    if mode == "square":
        if image.size[0] == size and image.size[1] == size:
            return image
        return image.resize((size, size))
    if mode == "width":
        if image.size[0] == size:
            return image
        scale = size / image.size[0]
        new_height = int(image.size[1] * scale)
        new_height = max(8, (new_height // 8) * 8)
        return image.resize((size, new_height))
    raise ValueError(f"Unknown resize_mode: {mode}")


def _make_diptych_and_mask(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    combined_image = Image.new("RGB", (width * 2, height))
    combined_image.paste(image, (0, 0))
    combined_image.paste(image, (width, 0))
    mask_array = np.zeros((height, width * 2), dtype=np.uint8)
    mask_array[:, width:] = 255
    mask = Image.fromarray(mask_array)
    return combined_image, mask


def _load_csv_rows(metadata_path: str) -> list[dict]:
    with open(metadata_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No rows found in metadata CSV: {metadata_path}")
    return rows


def _resolve_path(base_path: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return base_path / p


def _output_path(output_dir: Path, base_path: Path, image_path: Path) -> Path:
    if image_path.is_absolute():
        try:
            rel = image_path.relative_to(base_path)
        except ValueError:
            rel = Path(image_path.name)
    else:
        rel = image_path
    return output_dir / rel


def _load_adapters(adapter_dir: str, debug: bool):
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    adapter_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
    if debug:
        print("[DEBUG][csv] adapter_cfg:", adapter_cfg)
    adapter_affect_prompt = bool(adapter_cfg.get("affect_prompt", True))
    adapter_tokens_to_ca = bool(adapter_cfg.get("tokens_to_ca", False))

    prompt_adapter_path = os.path.join(adapter_dir, "prompt_adapter.pt")
    token_adapter_path = os.path.join(adapter_dir, "text_token_adapter.pt")
    thermal_adapter_path = os.path.join(adapter_dir, "thermal_query.pt")
    thermal_gate_path = os.path.join(adapter_dir, "thermal_gate.pt")

    kind_cfg = adapter_cfg.get("kind", None)
    if kind_cfg is None:
        has_pooled = os.path.exists(prompt_adapter_path)
        has_tokens = os.path.exists(token_adapter_path)
        if has_pooled and has_tokens:
            kind = "both"
        elif has_tokens:
            kind = "tokens"
        elif has_pooled:
            kind = "pooled"
        else:
            kind = "none"
    else:
        kind = str(kind_cfg).lower()

    scale = float(adapter_cfg.get("scale", 1.0))
    if debug:
        print(
            "[DEBUG][csv] adapter files:",
            "prompt_adapter.pt=",
            os.path.exists(prompt_adapter_path),
            "text_token_adapter.pt=",
            os.path.exists(token_adapter_path),
            "kind=",
            kind,
        )

    prompt_adapter = None
    token_adapter = None
    thermal_adapter = None
    thermal_query_generator = None
    thermal_gate = None
    thermal_gate_balance_text = False

    if os.path.exists(prompt_adapter_path) and kind in ("pooled", "both"):
        state = torch.load(prompt_adapter_path, map_location="cpu")
        dim = int(adapter_cfg.get("dim", state["net.0.weight"].shape[1]))
        hidden_dim = int(adapter_cfg.get("hidden_dim", state["net.0.weight"].shape[0]))
        prompt_adapter = PooledPromptAdapter(dim=dim, hidden_dim=hidden_dim, scale=scale)
        prompt_adapter.load_state_dict(state)
        prompt_adapter.eval()
        if debug:
            print("[DEBUG][csv] loaded prompt_adapter:", prompt_adapter_path)

    if os.path.exists(token_adapter_path) and kind in ("tokens", "token", "both"):
        state = torch.load(token_adapter_path, map_location="cpu")
        pooled_dim = int(adapter_cfg.get("pooled_dim", state["mlp.0.weight"].shape[1]))
        token_hidden_dim = int(adapter_cfg.get("token_hidden_dim", state["mlp.0.weight"].shape[0]))
        token_dim = int(adapter_cfg.get("token_dim", state["proj.weight"].shape[0]))
        out_features = int(state["mlp.2.weight"].shape[0])
        num_tokens = int(adapter_cfg.get("num_tokens", out_features // pooled_dim))
        token_adapter = TextTokenAdapter(
            pooled_dim=pooled_dim,
            token_dim=token_dim,
            num_tokens=num_tokens,
            hidden_dim=token_hidden_dim,
            scale=scale,
        )
        token_adapter.load_state_dict(state)
        token_adapter.eval()
        if debug:
            print("[DEBUG][csv] loaded text_token_adapter:", token_adapter_path)

    thermal_cfg = adapter_cfg.get("thermal", {}) or {}
    thermal_enabled = bool(thermal_cfg.get("enabled", False))
    if not thermal_enabled and os.path.exists(thermal_adapter_path):
        thermal_enabled = True
    if thermal_enabled:
        thermal_mode = str(thermal_cfg.get("mode", "static")).lower()
        if thermal_mode in ("query", "image_caption_query", "viscap_query"):
            thermal_query_generator = ThermalQueryGenerator(
                query_dim=int(thermal_cfg.get("query_dim", 256)),
                output_dim=int(thermal_cfg.get("output_dim", thermal_cfg.get("query_dim", 256))),
                num_queries=int(thermal_cfg.get("num_tokens", 4)),
                num_heads=int(thermal_cfg.get("query_heads", 8)),
                num_layers=int(thermal_cfg.get("query_layers", 2)),
                ffn_dim=int(thermal_cfg.get("query_ffn_dim", 1024)),
                scale=float(thermal_cfg.get("scale", 1.0)),
                init_std=float(thermal_cfg.get("init_std", 0.02)),
                use_type_embed=bool(thermal_cfg.get("use_type_embed", True)),
            )
            if os.path.exists(thermal_adapter_path):
                state = torch.load(thermal_adapter_path, map_location="cpu")
                thermal_query_generator.load_state_dict(state)
                thermal_query_generator.eval()
                if debug:
                    print("[DEBUG][csv] loaded thermal_query generator:", thermal_adapter_path)
            elif debug:
                print("[DEBUG][csv] thermal_query generator enabled but file missing:", thermal_adapter_path)
        else:
            thermal_dim = int(thermal_cfg.get("dim", adapter_cfg.get("pooled_dim", 768)))
            thermal_token_dim = int(thermal_cfg.get("token_dim", thermal_dim))
            thermal_num_tokens = int(thermal_cfg.get("num_tokens", 4))
            thermal_scale = float(thermal_cfg.get("scale", 1.0))
            thermal_init_std = float(thermal_cfg.get("init_std", 0.02))
            thermal_adapter = ThermalTokenAdapter(
                input_dim=thermal_dim,
                token_dim=thermal_token_dim,
                num_tokens=thermal_num_tokens,
                scale=thermal_scale,
                init_std=thermal_init_std,
            )
            if os.path.exists(thermal_adapter_path):
                state = torch.load(thermal_adapter_path, map_location="cpu")
                thermal_adapter.load_state_dict(state)
                thermal_adapter.eval()
                if debug:
                    print("[DEBUG][csv] loaded thermal_query:", thermal_adapter_path)
            elif debug:
                print("[DEBUG][csv] thermal_query enabled but file missing:", thermal_adapter_path)

        gate_cfg = thermal_cfg.get("gate", {}) or {}
        gate_enabled = bool(gate_cfg.get("enabled", False))
        gate_path_exists = os.path.exists(thermal_gate_path)
        if not gate_enabled and gate_path_exists:
            gate_enabled = True
        if gate_enabled and not gate_path_exists:
            gate_enabled = False
            if debug:
                print("[DEBUG][csv] thermal_gate enabled but missing file:", thermal_gate_path)
        if gate_enabled:
            state = torch.load(thermal_gate_path, map_location="cpu")
            inferred_in, inferred_out = _infer_gate_dims(state)
            gate_mode = str(gate_cfg.get("mode", "global")).lower()
            if gate_cfg.get("mode") is None and inferred_out > 1:
                gate_mode = "headwise"
            gate_heads = int(gate_cfg.get("heads", inferred_out if inferred_out > 1 else 8))
            gate_hidden_dim = int(gate_cfg.get("hidden_dim", 0))
            gate_dim = int(gate_cfg.get("dim", inferred_in))
            thermal_gate_balance_text = bool(gate_cfg.get("balance_text_tokens", False))
            thermal_gate = ThermalGate(
                input_dim=gate_dim,
                mode=gate_mode,
                heads=gate_heads,
                hidden_dim=gate_hidden_dim,
            )
            thermal_gate.load_state_dict(state)
            thermal_gate.eval()
            if debug:
                print("[DEBUG][csv] loaded thermal_gate:", thermal_gate_path)

    return (
        prompt_adapter,
        token_adapter,
        thermal_adapter,
        thermal_query_generator,
        thermal_gate,
        thermal_gate_balance_text,
        adapter_affect_prompt,
        adapter_tokens_to_ca,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--base-path", type=str, default=".")
    parser.add_argument("--image-key", type=str, default="kontext_images")
    parser.add_argument("--caption-key", type=str, default="prompt")
    parser.add_argument("--default-caption", type=str, default="")
    parser.add_argument("--caption-prefix", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument(
        "--resize-mode",
        type=str,
        choices=["square", "width"],
        default="square",
        help='How to resize inputs before making the diptych. "square" matches 512x512 training; "width" matches the single-image inference script behavior.',
    )
    parser.add_argument("--size", type=int, default=512, help="Resize target size (default: 512).")
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default=DEFAULT_PROMPT_PREFIX,
        help="Prefix prompt template injected before `--instruction`.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction for translation/editing.",
    )
    parser.add_argument(
        "--use-instruction-when-empty",
        action="store_true",
        help="If caption is empty, reuse the instruction text for adapter modulation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-mode",
        type=str,
        choices=["fixed", "incremental", "hash"],
        default="hash",
        help="How to derive per-image seeds.",
    )
    parser.add_argument("--guidance-scale", type=float, default=50.0)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--flux-path", type=str, default="black-forest-labs/flux.1-fill-dev")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--debug-adapter", action="store_true")
    parser.add_argument("--enable-model-cpu-offload", action="store_true")

    args = parser.parse_args()

    base_path = Path(args.base_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_csv_rows(args.metadata_path)
    if args.max_images and args.max_images > 0:
        rows = rows[: args.max_images]

    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
    if args.lora_path:
        _load_lora_weights(pipe, args.lora_path)

    adapter_cfg = {}
    adapter_ca_enabled = False
    adapter_ca_concat = False
    adapter_ca_cfg = {}
    if args.adapter_dir is not None:
        cfg_path = os.path.join(args.adapter_dir, "adapter_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
        thermal_cfg = adapter_cfg.get("thermal", {}) or {}
        adapter_ca_cfg = thermal_cfg.get("adapter_ca", {}) or {}
        thermal_adapter_ca_path = os.path.join(args.adapter_dir, "thermal_adapter_ca.pt")
        if os.path.exists(thermal_adapter_ca_path):
            adapter_ca_cfg.setdefault("enabled", True)
        adapter_ca_enabled = bool(adapter_ca_cfg.get("enabled", False))
        adapter_ca_concat = bool(adapter_ca_cfg.get("concat_to_text", False))
        if adapter_ca_enabled:
            adapter_ca_state = None
            if os.path.exists(thermal_adapter_ca_path):
                adapter_ca_state = torch.load(thermal_adapter_ca_path, map_location="cpu")
            _init_adapter_ca(pipe.transformer, adapter_ca_state, adapter_ca_cfg)
            model_cfg = adapter_cfg.get("model", {}) or {}
            _patch_transformer_forward(pipe.transformer, model_cfg)

    prompt_adapter = None
    token_adapter = None
    thermal_adapter = None
    thermal_query_generator = None
    thermal_gate = None
    thermal_gate_balance_text = False
    adapter_affect_prompt = True
    adapter_tokens_to_ca = False
    current_thermal_image_tokens = None
    current_adapter_token_mask = None
    if args.adapter_dir is not None:
        (
            prompt_adapter,
            token_adapter,
            thermal_adapter,
            thermal_query_generator,
            thermal_gate,
            thermal_gate_balance_text,
            adapter_affect_prompt,
            adapter_tokens_to_ca,
        ) = _load_adapters(args.adapter_dir, debug=args.debug_adapter)
        if (
            prompt_adapter is not None
            or token_adapter is not None
            or thermal_adapter is not None
            or thermal_query_generator is not None
            or thermal_gate is not None
        ):
            orig_encode_prompt = pipe.encode_prompt
            current_caption = ""
            use_instruction_for_adapter = False

            def encode_prompt_wrapped(*encode_args, **encode_kwargs):
                nonlocal current_caption, use_instruction_for_adapter
                prompt_embeds, pooled_prompt_embeds, text_ids = orig_encode_prompt(
                    *encode_args, **encode_kwargs
                )

                caption_kwargs = dict(encode_kwargs)
                caption_kwargs.pop("prompt_embeds", None)
                caption_kwargs.pop("pooled_prompt_embeds", None)
                caption_kwargs.pop("text_ids", None)
                if encode_args:
                    caption_args = list(encode_args)
                    caption_args[0] = current_caption
                    caption_prompt_embeds, adapter_pooled, _ = orig_encode_prompt(
                        *caption_args, **caption_kwargs
                    )
                else:
                    caption_kwargs["prompt"] = current_caption
                    caption_prompt_embeds, adapter_pooled, _ = orig_encode_prompt(**caption_kwargs)

                if prompt_adapter is not None:
                    _ensure_module_on(prompt_adapter, adapter_pooled)
                    adapter_pooled = prompt_adapter(adapter_pooled)
                    if adapter_affect_prompt and use_instruction_for_adapter:
                        pooled_prompt_embeds = adapter_pooled

                gate = None
                if thermal_gate is not None:
                    _ensure_module_on(thermal_gate, adapter_pooled)
                    gate = thermal_gate(adapter_pooled)

                thermal_tokens = None
                if thermal_query_generator is not None:
                    _ensure_module_on(thermal_query_generator, adapter_pooled)
                    if not adapter_ca_enabled or adapter_ca_concat:
                        thermal_query_generator.ensure_output_dim(
                            int(prompt_embeds.shape[-1]),
                            device=adapter_pooled.device,
                            dtype=adapter_pooled.dtype,
                        )
                    image_tokens = current_thermal_image_tokens
                    if image_tokens is not None:
                        image_tokens = image_tokens.to(
                            device=adapter_pooled.device, dtype=adapter_pooled.dtype
                        )
                    thermal_tokens = thermal_query_generator(
                        image_tokens=image_tokens,
                        caption_tokens=caption_prompt_embeds,
                    )
                elif thermal_adapter is not None:
                    _ensure_module_on(thermal_adapter, adapter_pooled)
                    thermal_tokens = thermal_adapter(
                        adapter_pooled.shape[0],
                        target_dim=int(prompt_embeds.shape[-1]),
                        device=adapter_pooled.device,
                        dtype=adapter_pooled.dtype,
                    )
                adapter_ca_tokens = None
                if thermal_tokens is not None:
                    if gate is not None:
                        thermal_tokens = _apply_gate(
                            thermal_tokens,
                            gate.to(dtype=thermal_tokens.dtype),
                            thermal_gate.mode,
                            thermal_gate.heads,
                        )
                    if adapter_ca_enabled:
                        adapter_ca_tokens = thermal_tokens
                    if not adapter_ca_enabled or adapter_ca_concat:
                        prompt_embeds = torch.cat(
                            [prompt_embeds, thermal_tokens.to(dtype=prompt_embeds.dtype)], dim=1
                        )
                        text_ids = _extend_txt_ids(text_ids, thermal_tokens.shape[1])

                if token_adapter is not None:
                    _ensure_module_on(token_adapter, adapter_pooled)
                    tokens = token_adapter(adapter_pooled).to(dtype=prompt_embeds.dtype)
                    if gate is not None and thermal_gate_balance_text:
                        tokens = _apply_gate(
                            tokens,
                            (1.0 - gate).to(dtype=tokens.dtype),
                            thermal_gate.mode,
                            thermal_gate.heads,
                        )
                    if adapter_tokens_to_ca and adapter_ca_enabled:
                        if adapter_ca_tokens is None:
                            adapter_ca_tokens = tokens
                        else:
                            adapter_ca_tokens = torch.cat([adapter_ca_tokens, tokens], dim=1)
                    else:
                        prompt_embeds = torch.cat([prompt_embeds, tokens], dim=1)
                        text_ids = _extend_txt_ids(text_ids, tokens.shape[1])
                if adapter_ca_enabled:
                    if adapter_ca_tokens is not None:
                        pipe.transformer._adapter_tokens = adapter_ca_tokens.to(
                            device=prompt_embeds.device, dtype=prompt_embeds.dtype
                        )
                        if current_adapter_token_mask is not None:
                            pipe.transformer._adapter_token_mask = current_adapter_token_mask.to(
                                device=prompt_embeds.device, dtype=prompt_embeds.dtype
                            )
                        else:
                            pipe.transformer._adapter_token_mask = None
                    else:
                        pipe.transformer._adapter_tokens = None
                        pipe.transformer._adapter_token_mask = None

                return prompt_embeds, pooled_prompt_embeds, text_ids

            pipe.encode_prompt = encode_prompt_wrapped

    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    final_prompt = f"{args.prompt_prefix}{args.instruction}"

    for i, row in enumerate(rows):
        rel_path = row.get(args.image_key, "")
        if not rel_path:
            raise ValueError(f"Missing image_key '{args.image_key}' in row {i}.")

        image_path = _resolve_path(base_path, rel_path)
        out_path = _output_path(output_dir, base_path, Path(rel_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and out_path.exists():
            continue

        caption = row.get(args.caption_key, "") if args.caption_key else ""
        if not caption:
            caption = args.default_caption
        caption = f"{args.caption_prefix}{caption}"

        if args.use_instruction_when_empty and not caption:
            current_caption = final_prompt
            use_instruction_for_adapter = True
        else:
            current_caption = caption
            use_instruction_for_adapter = False

        seed = _seed_for(Path(rel_path), args.seed, args.seed_mode)
        if args.seed_mode == "incremental":
            seed = int(args.seed) + i

        image = Image.open(image_path).convert("RGB")
        image = _resize_image(image, int(args.size), str(args.resize_mode))
        width, height = image.size
        combined_image, mask = _make_diptych_and_mask(image)
        if adapter_ca_enabled:
            current_adapter_token_mask = _encode_mask_tokens(pipe, combined_image, mask)
        if thermal_query_generator is not None:
            current_thermal_image_tokens = _encode_image_tokens(pipe, image)
            if args.debug_adapter:
                print(
                    "[DEBUG][csv] thermal_image_tokens.shape=",
                    tuple(current_thermal_image_tokens.shape),
                )

        result = pipe(
            prompt=final_prompt,
            image=combined_image,
            mask_image=mask,
            height=height,
            width=width * 2,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            generator=torch.Generator("cpu").manual_seed(int(seed)),
        ).images[0]

        result = result.crop((width, 0, width * 2, height))
        result.save(out_path)

        print(f"[{i+1}/{len(rows)}] {rel_path} -> {out_path.as_posix()} (seed={seed})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
