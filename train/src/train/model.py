import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict
import os
import prodigyopt
import json
from pathlib import Path

try:
    from peft import set_peft_model_state_dict
except Exception:  # pragma: no cover - optional PEFT API
    set_peft_model_state_dict = None

try:
    from torchvision import models as tv_models
    from torchvision.models import VGG16_Weights
except Exception:  # pragma: no cover - optional torchvision
    tv_models = None
    VGG16_Weights = None

from ..flux.transformer import tranformer_forward
from ..flux.block import AdapterCrossAttention
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, encode_images_fill, prepare_text_input


class PooledPromptAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class TextTokenAdapter(nn.Module):
    """
    Produce extra context tokens from the instruction (pooled text embedding).

    This is closer to IP-Adapter / prefix-tuning than the current pooled-only MLP:
    we inject adapter tokens into `encoder_hidden_states` so every cross-attention
    layer can attend to them.
    """

    def __init__(
        self,
        pooled_dim: int,
        token_dim: int,
        num_tokens: int = 8,
        hidden_dim: int = 1536,
        scale: float = 1.0,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        self.pooled_dim = int(pooled_dim)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.scale = float(scale)

        self.mlp = nn.Sequential(
            nn.Linear(self.pooled_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_tokens * self.pooled_dim),
        )
        self.proj = nn.Linear(self.pooled_dim, self.token_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        # pooled: [B, pooled_dim]
        bsz = pooled.shape[0]
        tokens = self.mlp(pooled).view(bsz, self.num_tokens, self.pooled_dim)
        tokens = self.proj(tokens)
        return self.scale * tokens


class ThermalTokenAdapter(nn.Module):
    """
    Learnable thermal query tokens that can be prefixed to the text context.
    """

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
        self.tokens = nn.Parameter(torch.randn(self.num_tokens, self.input_dim) * self.init_std)
        self.proj = None
        if self.token_dim != self.input_dim:
            self.proj = nn.Linear(self.input_dim, self.token_dim)

    def _ensure_proj(self, target_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if target_dim == self.token_dim:
            return
        self.proj = nn.Linear(self.input_dim, int(target_dim)).to(device=device, dtype=dtype)
        self.token_dim = int(target_dim)

    def forward(self, batch_size: int, target_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if target_dim is not None and target_dim != self.token_dim:
            self._ensure_proj(target_dim, device=device, dtype=dtype)
        tokens = self.tokens
        if self.proj is not None:
            tokens = self.proj(tokens)
        tokens = tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return self.scale * tokens


class ThermalGate(nn.Module):
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
            self.net = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(input_dim, out_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(pooled))


class ThermalQueryGenerator(nn.Module):
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

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.query_dim) * self.init_std)
        self.caption_proj = nn.LazyLinear(self.query_dim)
        self.image_proj = nn.LazyLinear(self.query_dim)

        if self.use_type_embed:
            self.caption_type = nn.Parameter(torch.zeros(1, 1, self.query_dim))
            self.image_type = nn.Parameter(torch.zeros(1, 1, self.query_dim))
        else:
            self.caption_type = None
            self.image_type = None

        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(self.query_dim, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        self.self_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(self.query_dim, self.num_heads, batch_first=True)
                for _ in range(self.num_layers)
            ]
        )
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.query_dim, self.ffn_dim),
                    nn.SiLU(),
                    nn.Linear(self.ffn_dim, self.query_dim),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(self.query_dim) for _ in range(self.num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(self.query_dim) for _ in range(self.num_layers)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(self.query_dim) for _ in range(self.num_layers)])

        self.out_proj = None
        if self.output_dim != self.query_dim:
            self.out_proj = nn.Linear(self.query_dim, self.output_dim)

    def ensure_output_dim(self, target_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        if int(target_dim) == self.output_dim:
            return
        self.out_proj = nn.Linear(self.query_dim, int(target_dim)).to(device=device, dtype=dtype)
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


_SSIM_WINDOW_CACHE: dict[tuple, torch.Tensor] = {}


class VGGPerceptualLoss(nn.Module):
    def __init__(
        self,
        layer_ids: list[int] | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        if tv_models is None:
            raise RuntimeError("torchvision is required for perceptual loss but is not available.")
        if layer_ids is None:
            layer_ids = [4, 9, 16]
        self.layer_ids = sorted({int(idx) for idx in layer_ids})
        weights = None
        if pretrained and VGG16_Weights is not None:
            weights = VGG16_Weights.DEFAULT
        self.vgg = tv_models.vgg16(weights=weights).features
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        img = img.clamp(0, 1)
        return (img - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = self._prep(pred)
        y = self._prep(target)
        loss = 0.0
        layer_ids = self.layer_ids
        next_idx = 0
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if next_idx < len(layer_ids) and idx == layer_ids[next_idx]:
                loss = loss + F.l1_loss(x, y, reduction="mean")
                next_idx += 1
                if next_idx >= len(layer_ids):
                    break
        return loss


def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    if img.shape[1] == 1:
        return img
    return (
        0.299 * img[:, 0:1]
        + 0.587 * img[:, 1:2]
        + 0.114 * img[:, 2:3]
    )


def _get_ssim_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (window_size, float(sigma), channels, device.type, device.index, str(dtype))
    cached = _SSIM_WINDOW_CACHE.get(key)
    if cached is not None:
        return cached
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]
    window = window_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)
    _SSIM_WINDOW_CACHE[key] = window
    return window


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("SSIM inputs must have the same shape.")
    channels = pred.shape[1]
    window = _get_ssim_window(
        window_size,
        sigma,
        channels,
        device=pred.device,
        dtype=pred.dtype,
    )
    padding = window_size // 2
    mu1 = F.conv2d(pred, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu1_mu2

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


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


def _extend_txt_ids(txt_ids: torch.Tensor, extra_tokens: int) -> torch.Tensor:
    """
    Extend FLUX `txt_ids` (positional ids) by appending `extra_tokens` rows.

    We keep the first 2 columns identical to the last existing token and
    increment the 3rd column to preserve a monotonic position axis.
    """
    if extra_tokens <= 0:
        return txt_ids

    if txt_ids.ndim == 3:
        base = txt_ids[0]
        add_batch_dim = True
    else:
        base = txt_ids
        add_batch_dim = False

    last = base[-1:].clone()  # [1, 3]
    extra = last.repeat(extra_tokens, 1)
    extra[:, 2] = extra[:, 2] + torch.arange(1, extra_tokens + 1, device=extra.device, dtype=extra.dtype)
    out = torch.cat([base, extra], dim=0)

    if add_batch_dim:
        out = out.unsqueeze(0)
    return out


def _flatten_image_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4:
        # [B, C, H, W] -> [B, H*W, C]
        return tokens.permute(0, 2, 3, 1).reshape(tokens.shape[0], -1, tokens.shape[1])
    if tokens.ndim == 3:
        return tokens
    raise ValueError(f"Unexpected image token shape: {tuple(tokens.shape)}")


def _pool_tokens(tokens: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    tokens = _flatten_image_tokens(tokens)
    if tokens.shape[1] == num_tokens:
        return tokens
    tokens_t = tokens.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(tokens_t, num_tokens)
    return pooled.transpose(1, 2)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        loss_config: dict = None,
        adapter_config: dict = None,
        debug_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
        train_mode: str = "default",
        pretrain_config: dict = None,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
            
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config or {}
        self.adapter_config = adapter_config or {}
        self.debug_config = debug_config or {}
        self._debug_printed_step = False
        self.train_mode = str(train_mode or "default").lower()
        self.pretrain_config = pretrain_config or {}
        self.adapter_trainable = bool(self.adapter_config.get("trainable", False))
        self.adapter_affect_prompt = bool(self.adapter_config.get("affect_prompt", True))
        self.adapter_tokens_to_ca = bool(self.adapter_config.get("tokens_to_ca", False))

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise
        
        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')
            
        self.lora_layers = self.init_lora(lora_path, lora_config)
        with torch.no_grad():
            prompt_embeds, _, _ = prepare_text_input(self.flux_fill_pipe, ["warmup"])
            self.text_embed_dim = int(prompt_embeds.shape[-1])
        (
            self.prompt_adapter,
            self.text_token_adapter,
            self.thermal_token_adapter,
            self.thermal_query_generator,
            self.thermal_gate,
            self.thermal_gate_balance_text,
        ) = self._init_adapters()
        self._load_adapter_weights(self.adapter_config.get("load_dir"))
        self.adapter_ca_layers = []
        self.thermal_adapter_ca_enabled = False
        self.thermal_adapter_ca_concat = False
        self._init_adapter_ca()

        self.image_loss_cfg = self.loss_config.get("image_loss", {}) or {}
        perceptual_weight = float(self.image_loss_cfg.get("perceptual_weight", 0.0))
        self.perceptual_loss = None
        if perceptual_weight > 0.0:
            if tv_models is None:
                raise RuntimeError("perceptual_weight > 0 requires torchvision to be installed.")
            layer_ids = self.image_loss_cfg.get("perceptual_layers", [4, 9, 16])
            pretrained = bool(self.image_loss_cfg.get("perceptual_pretrained", True))
            self.perceptual_loss = VGGPerceptualLoss(layer_ids=layer_ids, pretrained=pretrained)

        self.to(device).to(dtype)
        if self.perceptual_loss is not None:
            self.perceptual_loss.to(device=device, dtype=torch.float32)

    def _debug(self, *args):
        if not self.debug_config.get("enabled", False):
            return
        print("[DEBUG][OminiModel]", *args)

    def _load_lora_state_dict(self, lora_path: str) -> dict:
        path = Path(lora_path).expanduser()
        if path.is_dir():
            for name in (
                "pytorch_lora_weights.safetensors",
                "pytorch_lora_weights.bin",
                "adapter_model.safetensors",
                "adapter_model.bin",
            ):
                candidate = path / name
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {path}")
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path))
        return torch.load(str(path), map_location="cpu")

    def _rewrite_lora_adapter_keys(self, state_dict: dict, adapter_name: str) -> dict:
        if adapter_name == "default":
            return state_dict
        if not any(".default." in key for key in state_dict.keys()):
            return state_dict
        return {
            key.replace(".default.", f".{adapter_name}."): value
            for key, value in state_dict.items()
        }

    def _load_lora_into_adapter(self, adapter_name: str, lora_path: str) -> None:
        state_dict = self._load_lora_state_dict(lora_path)
        state_dict = self._rewrite_lora_adapter_keys(state_dict, adapter_name)
        if set_peft_model_state_dict is not None:
            set_peft_model_state_dict(self.transformer, state_dict, adapter_name=adapter_name)
        else:
            self.transformer.load_state_dict(state_dict, strict=False)

    def _set_active_lora_adapters(self, adapter_names: list[str], adapter_scales: list[float]) -> None:
        if not adapter_names:
            return
        if hasattr(self.flux_fill_pipe, "set_adapters"):
            try:
                self.flux_fill_pipe.set_adapters(adapter_names, adapter_scales)
            except TypeError:
                self.flux_fill_pipe.set_adapters(adapter_names)
            return
        if hasattr(self.transformer, "set_adapters"):
            try:
                self.transformer.set_adapters(adapter_names, adapter_scales)
            except TypeError:
                self.transformer.set_adapters(adapter_names)
            return
        if hasattr(self.transformer, "set_adapter"):
            self.transformer.set_adapter(adapter_names)

    def _collect_lora_params(self, adapter_names: list[str]) -> list[torch.nn.Parameter]:
        if not adapter_names:
            return []
        params: list[torch.nn.Parameter] = []
        for name, param in self.transformer.named_parameters():
            if "lora" not in name:
                continue
            if any(f".{adapter_name}." in name for adapter_name in adapter_names):
                params.append(param)
        return params

    def _apply_adapters(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        pooled_prompt_embeds_adapter: torch.Tensor | None = None,
        apply_to_pooled_prompt: bool = True,
        thermal_image_tokens: torch.Tensor | None = None,
        thermal_caption_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        adapter_pooled = pooled_prompt_embeds if pooled_prompt_embeds_adapter is None else pooled_prompt_embeds_adapter
        if self.prompt_adapter is not None:
            adapter_pooled = self.prompt_adapter(adapter_pooled)
            if apply_to_pooled_prompt:
                if pooled_prompt_embeds_adapter is None:
                    pooled_prompt_embeds = adapter_pooled
                else:
                    pooled_prompt_embeds = self.prompt_adapter(pooled_prompt_embeds)

        gate = None
        if self.thermal_gate is not None:
            gate = self.thermal_gate(adapter_pooled)

        thermal_tokens = None
        if self.thermal_query_generator is not None:
            thermal_tokens = self.thermal_query_generator(
                image_tokens=thermal_image_tokens,
                caption_tokens=thermal_caption_tokens,
            )
        elif self.thermal_token_adapter is not None:
            thermal_tokens = self.thermal_token_adapter(
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
                    self.thermal_gate.mode,
                    self.thermal_gate.heads,
                )
            if self.thermal_adapter_ca_enabled:
                adapter_ca_tokens = thermal_tokens
            if not self.thermal_adapter_ca_enabled or self.thermal_adapter_ca_concat:
                prompt_embeds = torch.cat([prompt_embeds, thermal_tokens.to(dtype=prompt_embeds.dtype)], dim=1)
                text_ids = _extend_txt_ids(text_ids, thermal_tokens.shape[1])
            if self.debug_config.get("enabled", False) and (not self._debug_printed_step):
                self._debug(
                    "thermal_tokens injected",
                    "thermal_tokens.shape=",
                    tuple(thermal_tokens.shape),
                )

        if self.text_token_adapter is not None:
            orig_prompt_len = int(prompt_embeds.shape[1])
            orig_txt_ids_len = int(text_ids.shape[1] if text_ids.ndim == 3 else text_ids.shape[0])
            # Lazily fix token_dim to match prompt_embeds if left as placeholder.
            if (
                getattr(self.text_token_adapter, "token_dim", None) == self.text_token_adapter.pooled_dim
                and prompt_embeds.shape[-1] != self.text_token_adapter.pooled_dim
            ):
                self.text_token_adapter.proj = nn.Linear(
                    self.text_token_adapter.pooled_dim, int(prompt_embeds.shape[-1])
                ).to(device=pooled_prompt_embeds.device, dtype=pooled_prompt_embeds.dtype)
                self.text_token_adapter.token_dim = int(prompt_embeds.shape[-1])

            adapter_tokens = self.text_token_adapter(adapter_pooled)  # [B, N, D]
            if gate is not None and self.thermal_gate_balance_text:
                adapter_tokens = _apply_gate(
                    adapter_tokens,
                    (1.0 - gate).to(dtype=adapter_tokens.dtype),
                    self.thermal_gate.mode,
                    self.thermal_gate.heads,
                )
            append_to_prompt = not (self.adapter_tokens_to_ca and self.thermal_adapter_ca_enabled)
            if not append_to_prompt:
                if adapter_ca_tokens is None:
                    adapter_ca_tokens = adapter_tokens
                else:
                    adapter_ca_tokens = torch.cat([adapter_ca_tokens, adapter_tokens], dim=1)
            else:
                prompt_embeds = torch.cat([prompt_embeds, adapter_tokens.to(dtype=prompt_embeds.dtype)], dim=1)
                text_ids = _extend_txt_ids(text_ids, adapter_tokens.shape[1])
            if self.debug_config.get("enabled", False) and (not self._debug_printed_step):
                self._debug(
                    "token_adapter",
                    "adapter_tokens.shape=", tuple(adapter_tokens.shape),
                    "to_ca=", not append_to_prompt,
                    "prompt_embeds_len", orig_prompt_len, "->", int(prompt_embeds.shape[1]),
                    "txt_ids_len",
                    orig_txt_ids_len,
                    "->",
                    int(text_ids.shape[1] if text_ids.ndim == 3 else text_ids.shape[0]),
                )

        return prompt_embeds, pooled_prompt_embeds, text_ids, adapter_ca_tokens

    def _load_adapter_weights(self, adapter_dir: str | None) -> None:
        if not adapter_dir:
            return
        adapter_dir = os.path.expanduser(adapter_dir)
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"adapter.load_dir not found: {adapter_dir}")

        def _load_if_exists(module: nn.Module | None, name: str) -> None:
            if module is None:
                return
            path = os.path.join(adapter_dir, name)
            if not os.path.exists(path):
                return
            state = torch.load(path, map_location="cpu")
            if isinstance(module, TextTokenAdapter):
                proj_weight = state.get("proj.weight")
                if proj_weight is not None:
                    in_dim = int(proj_weight.shape[1])
                    out_dim = int(proj_weight.shape[0])
                    if getattr(module, "pooled_dim", in_dim) != in_dim:
                        raise ValueError(
                            "TextTokenAdapter pooled_dim mismatch: "
                            f"checkpoint={in_dim} module={module.pooled_dim}"
                        )
                    if getattr(module, "token_dim", out_dim) != out_dim:
                        module.proj = nn.Linear(in_dim, out_dim)
                        module.token_dim = out_dim
            module.load_state_dict(state)

        _load_if_exists(self.prompt_adapter, "prompt_adapter.pt")
        _load_if_exists(self.text_token_adapter, "text_token_adapter.pt")
        _load_if_exists(self.thermal_token_adapter, "thermal_query.pt")
        _load_if_exists(self.thermal_query_generator, "thermal_query.pt")
        _load_if_exists(self.thermal_gate, "thermal_gate.pt")

    def _init_adapters(self):
        cfg = self.adapter_config or {}
        if not cfg.get("enabled", False):
            return None, None, None, None, None, False

        kind = str(cfg.get("kind", "pooled")).lower()
        scale = float(cfg.get("scale", 1.0))

        pooled_adapter = None
        token_adapter = None
        thermal_adapter = None
        thermal_query_generator = None
        thermal_gate = None
        thermal_gate_balance_text = False

        if kind in ("pooled", "both"):
            dim = int(cfg.get("dim", 768))
            hidden_dim = int(cfg.get("hidden_dim", dim * 4))
            pooled_adapter = PooledPromptAdapter(dim=dim, hidden_dim=hidden_dim, scale=scale)

        if kind in ("tokens", "token", "both"):
            pooled_dim = int(cfg.get("pooled_dim", 768))
            num_tokens = int(cfg.get("num_tokens", 8))
            token_hidden_dim = int(cfg.get("token_hidden_dim", 1536))
            # token_dim is inferred at runtime from `prompt_embeds` unless explicitly set.
            token_dim = int(cfg.get("token_dim", 0))
            token_adapter = TextTokenAdapter(
                pooled_dim=pooled_dim,
                token_dim=token_dim if token_dim > 0 else pooled_dim,  # placeholder, fixed on first use
                num_tokens=num_tokens,
                hidden_dim=token_hidden_dim,
                scale=scale,
            )

        thermal_cfg = cfg.get("thermal", {}) or {}
        if thermal_cfg.get("enabled", False):
            thermal_mode = str(thermal_cfg.get("mode", "static")).lower()
            if thermal_mode in ("query", "image_caption_query", "viscap_query"):
                query_dim = int(thermal_cfg.get("query_dim", self.text_embed_dim))
                thermal_query_generator = ThermalQueryGenerator(
                    query_dim=query_dim,
                    output_dim=int(thermal_cfg.get("output_dim", self.text_embed_dim)),
                    num_queries=int(thermal_cfg.get("num_tokens", 4)),
                    num_heads=int(thermal_cfg.get("query_heads", 8)),
                    num_layers=int(thermal_cfg.get("query_layers", 2)),
                    ffn_dim=int(thermal_cfg.get("query_ffn_dim", query_dim * 4)),
                    scale=float(thermal_cfg.get("scale", 1.0)),
                    init_std=float(thermal_cfg.get("init_std", 0.02)),
                    use_type_embed=bool(thermal_cfg.get("use_type_embed", True)),
                )
            else:
                thermal_dim = int(thermal_cfg.get("dim", cfg.get("pooled_dim", cfg.get("dim", 768))))
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

            gate_cfg = thermal_cfg.get("gate", {}) or {}
            if gate_cfg.get("enabled", False):
                gate_default_dim = int(
                    gate_cfg.get(
                        "dim",
                        thermal_cfg.get("dim", cfg.get("pooled_dim", cfg.get("dim", 768))),
                    )
                )
                gate_mode = str(gate_cfg.get("mode", "global")).lower()
                gate_heads = int(gate_cfg.get("heads", 8))
                gate_hidden_dim = int(gate_cfg.get("hidden_dim", 0))
                gate_dim = int(gate_cfg.get("dim", gate_default_dim))
                thermal_gate_balance_text = bool(gate_cfg.get("balance_text_tokens", False))
                thermal_gate = ThermalGate(
                    input_dim=gate_dim,
                    mode=gate_mode,
                    heads=gate_heads,
                    hidden_dim=gate_hidden_dim,
                )

        return (
            pooled_adapter,
            token_adapter,
            thermal_adapter,
            thermal_query_generator,
            thermal_gate,
            thermal_gate_balance_text,
        )

    def _init_adapter_ca(self) -> None:
        thermal_cfg = (self.adapter_config or {}).get("thermal", {}) or {}
        ca_cfg = thermal_cfg.get("adapter_ca", {}) or {}
        if not ca_cfg.get("enabled", False):
            return
        scale_init = float(ca_cfg.get("scale", 0.0))
        trainable_scale = bool(ca_cfg.get("trainable_scale", True))
        gate_cfg = ca_cfg.get("gate", {}) or {}
        gate_enabled = bool(gate_cfg.get("enabled", False))
        gate_mode = str(gate_cfg.get("mode", "headwise")).lower()
        gate_hidden_dim = int(gate_cfg.get("hidden_dim", 0))
        gate_init_bias = float(gate_cfg.get("init_bias", 0.0))
        self.thermal_adapter_ca_concat = bool(ca_cfg.get("concat_to_text", False))
        for block in self.transformer.transformer_blocks:
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
            self.adapter_ca_layers.append(block.adapter_ca)
        self.thermal_adapter_ca_enabled = True

    def init_lora(self, lora_path: str, lora_config: dict):
        self.lora_adapter_names: list[str] = []
        self.lora_trainable_adapters: list[str] = []
        self.lora_active_adapters: list[str] = []
        self.lora_adapter_scales: dict[str, float] = {}
        self.lora_save_adapters: list[str] = []

        if not (lora_path or lora_config):
            return []

        if isinstance(lora_config, dict) and lora_config.get("adapters"):
            adapters_cfg = lora_config.get("adapters") or []
            reserved = {"name", "path", "trainable", "scale", "config"}
            for adapter_cfg in adapters_cfg:
                name = str(adapter_cfg.get("name", "default"))
                adapter_lora_cfg = adapter_cfg.get("config") or {
                    k: v for k, v in adapter_cfg.items() if k not in reserved
                }
                if adapter_lora_cfg:
                    self.transformer.add_adapter(
                        LoraConfig(**adapter_lora_cfg), adapter_name=name
                    )
                if adapter_cfg.get("path"):
                    self._load_lora_into_adapter(name, adapter_cfg["path"])
                if bool(adapter_cfg.get("trainable", True)):
                    self.lora_trainable_adapters.append(name)
                self.lora_adapter_names.append(name)
                self.lora_adapter_scales[name] = float(adapter_cfg.get("scale", 1.0))

            self.lora_active_adapters = list(
                lora_config.get("active") or self.lora_adapter_names
            )
            self.lora_save_adapters = list(
                lora_config.get("save_adapters") or self.lora_trainable_adapters
            )
            active_scales = [
                self.lora_adapter_scales.get(name, 1.0)
                for name in self.lora_active_adapters
            ]
            self._set_active_lora_adapters(self.lora_active_adapters, active_scales)
            return self._collect_lora_params(self.lora_trainable_adapters)

        if lora_path:
            self._load_lora_into_adapter("default", lora_path)
            self.lora_adapter_names = ["default"]
            self.lora_trainable_adapters = ["default"]
            self.lora_active_adapters = ["default"]
            self.lora_save_adapters = ["default"]
            self.lora_adapter_scales["default"] = 1.0
            return self._collect_lora_params(self.lora_trainable_adapters)

        self.transformer.add_adapter(LoraConfig(**lora_config))
        self.lora_adapter_names = ["default"]
        self.lora_trainable_adapters = ["default"]
        self.lora_active_adapters = ["default"]
        self.lora_save_adapters = ["default"]
        self.lora_adapter_scales["default"] = 1.0
        lora_layers = filter(lambda p: p.requires_grad, self.transformer.parameters())
        return list(lora_layers)

    def save_lora(self, path: str):
        has_lora = bool(self.lora_adapter_names) or bool(self.lora_layers)
        if has_lora:
            save_adapters = self.lora_save_adapters or ["default"]
            if len(save_adapters) > 1:
                for adapter_name in save_adapters:
                    adapter_dir = os.path.join(path, f"lora_{adapter_name}")
                    os.makedirs(adapter_dir, exist_ok=True)
                    try:
                        state_dict = get_peft_model_state_dict(
                            self.transformer, adapter_name=adapter_name
                        )
                    except TypeError:
                        state_dict = get_peft_model_state_dict(self.transformer)
                    FluxFillPipeline.save_lora_weights(
                        save_directory=adapter_dir,
                        transformer_lora_layers=state_dict,
                        safe_serialization=True,
                    )
            else:
                adapter_name = save_adapters[0] if save_adapters else None
                try:
                    state_dict = get_peft_model_state_dict(
                        self.transformer, adapter_name=adapter_name
                    )
                except TypeError:
                    state_dict = get_peft_model_state_dict(self.transformer)
                FluxFillPipeline.save_lora_weights(
                    save_directory=path,
                    transformer_lora_layers=state_dict,
                    safe_serialization=True,
                )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))
        if (
            self.prompt_adapter is not None
            or self.text_token_adapter is not None
            or self.thermal_token_adapter is not None
            or self.thermal_query_generator is not None
            or self.thermal_gate is not None
        ):
            os.makedirs(path, exist_ok=True)
            if self.prompt_adapter is not None:
                torch.save(self.prompt_adapter.state_dict(), os.path.join(path, "prompt_adapter.pt"))
            if self.text_token_adapter is not None:
                torch.save(self.text_token_adapter.state_dict(), os.path.join(path, "text_token_adapter.pt"))
            if self.thermal_token_adapter is not None:
                torch.save(self.thermal_token_adapter.state_dict(), os.path.join(path, "thermal_query.pt"))
            if self.thermal_query_generator is not None:
                torch.save(self.thermal_query_generator.state_dict(), os.path.join(path, "thermal_query.pt"))
            if self.adapter_ca_layers:
                torch.save(
                    [layer.state_dict() for layer in self.adapter_ca_layers],
                    os.path.join(path, "thermal_adapter_ca.pt"),
                )
            if self.thermal_gate is not None:
                torch.save(self.thermal_gate.state_dict(), os.path.join(path, "thermal_gate.pt"))

            # Persist a resolved adapter config so inference can reconstruct modules with correct dims.
            resolved = dict(self.adapter_config or {})
            resolved.setdefault("enabled", True)
            resolved.setdefault("model", dict(self.model_config or {}))
            resolved["tokens_to_ca"] = bool(self.adapter_tokens_to_ca)
            if self.prompt_adapter is not None:
                # infer dims from module weights (robust to config mismatches)
                resolved.setdefault("kind", "pooled")
                resolved["dim"] = int(self.prompt_adapter.net[0].in_features)
                resolved["hidden_dim"] = int(self.prompt_adapter.net[0].out_features)
                resolved["scale"] = float(getattr(self.prompt_adapter, "scale", resolved.get("scale", 1.0)))

            if self.text_token_adapter is not None:
                resolved["kind"] = str(resolved.get("kind", "tokens")).lower()
                resolved["pooled_dim"] = int(getattr(self.text_token_adapter, "pooled_dim", 768))
                resolved["token_dim"] = int(getattr(self.text_token_adapter, "token_dim", resolved["pooled_dim"]))
                resolved["num_tokens"] = int(getattr(self.text_token_adapter, "num_tokens", 8))
                resolved["token_hidden_dim"] = int(self.text_token_adapter.mlp[0].out_features)
                resolved["scale"] = float(getattr(self.text_token_adapter, "scale", resolved.get("scale", 1.0)))
            if self.thermal_token_adapter is not None:
                thermal_resolved = dict(resolved.get("thermal", {}) or {})
                thermal_resolved["enabled"] = True
                thermal_resolved["dim"] = int(getattr(self.thermal_token_adapter, "input_dim", 768))
                thermal_resolved["token_dim"] = int(getattr(self.thermal_token_adapter, "token_dim", thermal_resolved["dim"]))
                thermal_resolved["num_tokens"] = int(getattr(self.thermal_token_adapter, "num_tokens", 4))
                thermal_resolved["scale"] = float(getattr(self.thermal_token_adapter, "scale", 1.0))
                thermal_resolved["init_std"] = float(getattr(self.thermal_token_adapter, "init_std", 0.02))
                thermal_resolved["mode"] = str(thermal_resolved.get("mode", "static"))
                resolved["thermal"] = thermal_resolved
            if self.thermal_query_generator is not None:
                thermal_resolved = dict(resolved.get("thermal", {}) or {})
                thermal_resolved["enabled"] = True
                thermal_resolved["mode"] = str(thermal_resolved.get("mode", "query"))
                thermal_resolved["num_tokens"] = int(getattr(self.thermal_query_generator, "num_queries", 4))
                thermal_resolved["query_dim"] = int(getattr(self.thermal_query_generator, "query_dim", self.text_embed_dim))
                thermal_resolved["output_dim"] = int(getattr(self.thermal_query_generator, "output_dim", self.text_embed_dim))
                thermal_resolved["query_heads"] = int(getattr(self.thermal_query_generator, "num_heads", 8))
                thermal_resolved["query_layers"] = int(getattr(self.thermal_query_generator, "num_layers", 2))
                thermal_resolved["query_ffn_dim"] = int(getattr(self.thermal_query_generator, "ffn_dim", self.text_embed_dim * 4))
                thermal_resolved["scale"] = float(getattr(self.thermal_query_generator, "scale", 1.0))
                thermal_resolved["init_std"] = float(getattr(self.thermal_query_generator, "init_std", 0.02))
                thermal_resolved["use_type_embed"] = bool(getattr(self.thermal_query_generator, "use_type_embed", True))
                resolved["thermal"] = thermal_resolved
            if self.thermal_adapter_ca_enabled:
                thermal_resolved = dict(resolved.get("thermal", {}) or {})
                ca_resolved = dict(thermal_resolved.get("adapter_ca", {}) or {})
                scale_param = self.adapter_ca_layers[0].scale if self.adapter_ca_layers else None
                trainable_scale = bool(isinstance(scale_param, torch.nn.Parameter)) if scale_param is not None else True
                ca_resolved["enabled"] = True
                ca_resolved["concat_to_text"] = bool(self.thermal_adapter_ca_concat)
                ca_resolved["trainable_scale"] = trainable_scale
                if self.adapter_ca_layers:
                    ca_layer = self.adapter_ca_layers[0]
                    gate_mode = getattr(ca_layer, "gate_mode", "none")
                    if gate_mode and gate_mode != "none":
                        gate_resolved = dict(ca_resolved.get("gate", {}) or {})
                        gate_resolved["enabled"] = True
                        gate_resolved["mode"] = str(gate_mode)
                        gate_resolved["hidden_dim"] = int(getattr(ca_layer, "gate_hidden_dim", 0))
                        gate_resolved["init_bias"] = float(getattr(ca_layer, "gate_init_bias", 0.0))
                        ca_resolved["gate"] = gate_resolved
                thermal_resolved["adapter_ca"] = ca_resolved
                resolved["thermal"] = thermal_resolved
            if self.thermal_gate is not None:
                thermal_resolved = dict(resolved.get("thermal", {}) or {})
                gate_resolved = dict(thermal_resolved.get("gate", {}) or {})
                gate_resolved["enabled"] = True
                gate_resolved["mode"] = str(getattr(self.thermal_gate, "mode", "global"))
                gate_resolved["heads"] = int(getattr(self.thermal_gate, "heads", 1))
                gate_resolved["hidden_dim"] = int(getattr(self.thermal_gate, "hidden_dim", 0))
                gate_resolved["dim"] = int(
                    getattr(
                        self.thermal_gate,
                        "input_dim",
                        thermal_resolved.get("dim", thermal_resolved.get("output_dim", self.text_embed_dim)),
                    )
                )
                gate_resolved["balance_text_tokens"] = bool(self.thermal_gate_balance_text)
                thermal_resolved["gate"] = gate_resolved
                resolved["thermal"] = thermal_resolved

            with open(os.path.join(path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(resolved, f, ensure_ascii=False, indent=2)

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        if self.train_mode == "thermal_query_pretrain":
            if self.thermal_query_generator is None:
                raise ValueError("thermal_query_pretrain requires thermal_query_generator to be enabled.")
            self.trainable_params = list(self.thermal_query_generator.parameters())
            if self.pretrain_config.get("train_gate", False) and self.thermal_gate is not None:
                self.trainable_params += list(self.thermal_gate.parameters())
        else:
            # Set the trainable parameters
            self.trainable_params = list(self.lora_layers)
            if self.adapter_trainable:
                if self.prompt_adapter is not None:
                    self.trainable_params += list(self.prompt_adapter.parameters())
                if self.text_token_adapter is not None:
                    self.trainable_params += list(self.text_token_adapter.parameters())
                if self.thermal_token_adapter is not None:
                    self.trainable_params += list(self.thermal_token_adapter.parameters())
                if self.thermal_query_generator is not None:
                    self.trainable_params += list(self.thermal_query_generator.parameters())
                if self.adapter_ca_layers:
                    for layer in self.adapter_ca_layers:
                        self.trainable_params += list(layer.parameters())
                if self.thermal_gate is not None:
                    self.trainable_params += list(self.thermal_gate.parameters())

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        if self.debug_config.get("enabled", False):
            trainable_numel = sum(int(p.numel()) for p in self.trainable_params)
            lora_numel = 0
            lora_param_count = 0
            for name, p in self.transformer.named_parameters():
                if p.requires_grad and ("lora_" in name or ".lora_" in name):
                    lora_numel += int(p.numel())
                    lora_param_count += 1
            adapter_numel = 0
            if self.prompt_adapter is not None:
                adapter_numel += sum(int(p.numel()) for p in self.prompt_adapter.parameters())
            if self.text_token_adapter is not None:
                adapter_numel += sum(int(p.numel()) for p in self.text_token_adapter.parameters())
            thermal_numel = 0
            if self.thermal_token_adapter is not None:
                thermal_numel += sum(int(p.numel()) for p in self.thermal_token_adapter.parameters())
            if self.thermal_query_generator is not None:
                thermal_numel += sum(int(p.numel()) for p in self.thermal_query_generator.parameters())
            if self.adapter_ca_layers:
                for layer in self.adapter_ca_layers:
                    thermal_numel += sum(int(p.numel()) for p in layer.parameters())
            gate_numel = 0
            if self.thermal_gate is not None:
                gate_numel += sum(int(p.numel()) for p in self.thermal_gate.parameters())
            self._debug(
                "trainable_numel=", trainable_numel,
                "lora_numel=", lora_numel,
                "lora_param_count=", lora_param_count,
                "adapter_numel=", adapter_numel,
                "thermal_numel=", thermal_numel,
                "gate_numel=", gate_numel,
                "adapter_kind=", (self.adapter_config or {}).get("kind", "pooled"),
            )

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def _thermal_query_pretrain_loss(self, batch: dict) -> torch.Tensor:
        if self.thermal_query_generator is None:
            raise ValueError("thermal_query_pretrain requires thermal_query_generator to be enabled.")

        imgs = batch["image"]
        prompts = batch["description"]
        captions = batch.get("caption")

        with torch.no_grad():
            prompt_embeds, _, _ = prepare_text_input(self.flux_fill_pipe, prompts)
            caption_prompt_embeds = None
            if captions is not None:
                caption_prompt_embeds, _, _ = prepare_text_input(self.flux_fill_pipe, captions)

            width = int(imgs.shape[-1] // 2)
            vis = imgs[..., :width]
            ir = imgs[..., width:]

            vis_tokens, _ = encode_images(self.flux_fill_pipe, vis)
            ir_tokens, _ = encode_images(self.flux_fill_pipe, ir)

            thermal_image_tokens = _flatten_image_tokens(vis_tokens).to(
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
            )
            target_tokens = _pool_tokens(ir_tokens, self.thermal_query_generator.num_queries).to(
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
            )

            thermal_caption_tokens = caption_prompt_embeds if caption_prompt_embeds is not None else prompt_embeds

        self.thermal_query_generator.ensure_output_dim(
            int(target_tokens.shape[-1]),
            device=thermal_image_tokens.device,
            dtype=thermal_image_tokens.dtype,
        )
        pred_tokens = self.thermal_query_generator(
            image_tokens=thermal_image_tokens,
            caption_tokens=thermal_caption_tokens,
        )

        if bool(self.pretrain_config.get("normalize", True)):
            pred_tokens = F.normalize(pred_tokens, dim=-1)
            target_tokens = F.normalize(target_tokens, dim=-1)

        loss_type = str(self.pretrain_config.get("loss", "mse")).lower()
        if loss_type == "l1":
            loss = F.l1_loss(pred_tokens, target_tokens)
        elif loss_type == "cosine":
            loss = 1.0 - F.cosine_similarity(pred_tokens, target_tokens, dim=-1).mean()
        else:
            loss = F.mse_loss(pred_tokens, target_tokens)

        weight = float(self.pretrain_config.get("weight", 1.0))
        self.last_t = 0.0
        return loss * weight

    def step(self, batch):
        if self.train_mode == "thermal_query_pretrain":
            return self._thermal_query_pretrain_loss(batch)

        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        captions = batch.get("caption")
        position_delta = batch["position_delta"][0]

        thermal_image_tokens = None
        thermal_caption_tokens = None

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            pooled_prompt_embeds_adapter = pooled_prompt_embeds
            caption_prompt_embeds = None
            if captions is not None:
                caption_prompt_embeds, pooled_prompt_embeds_adapter, _ = prepare_text_input(
                    self.flux_fill_pipe, captions
                )
            x_0, x_cond, img_ids, mask_tokens = encode_images_fill(
                self.flux_fill_pipe,
                imgs,
                mask_imgs,
                prompt_embeds.dtype,
                prompt_embeds.device,
                return_mask_tokens=True,
            )
            if self.thermal_query_generator is not None:
                width = int(imgs.shape[-1] // 2)
                vis = imgs[:, :, :, :width]
                vis_tokens, _ = encode_images(self.flux_fill_pipe, vis)
                thermal_image_tokens = _flatten_image_tokens(vis_tokens).to(
                    device=prompt_embeds.device,
                    dtype=prompt_embeds.dtype,
                )
                if caption_prompt_embeds is not None:
                    thermal_caption_tokens = caption_prompt_embeds
                else:
                    thermal_caption_tokens = prompt_embeds

            # Prepare t (sigma) and x_t
            sigma_schedule = self.loss_config.get("sigma_schedule", None)
            schedule_steps = int(self.loss_config.get("schedule_steps", 1000))
            sampled_timestep_id = None
            if sigma_schedule == "flux_shifted_1000":
                # Match Flux/FlowMatch-style sigma schedule (shifted) used by Kontext training code.
                # See DiffSynth-Studio FlowMatchScheduler.set_timesteps_flux.
                sigma_min = float(self.loss_config.get("sigma_min", 0.003 / 1.002))
                sigma_max = float(self.loss_config.get("sigma_max", 1.0))
                shift = float(self.loss_config.get("shift", 3.0))
                sigmas = torch.linspace(sigma_max, sigma_min, schedule_steps, device=self.device)
                sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
                sampled_timestep_id = torch.randint(0, schedule_steps, (1,), device=self.device)
                t = sigmas[sampled_timestep_id].expand((imgs.shape[0],))
            else:
                t_sampling = self.loss_config.get("t_sampling", "sigmoid_normal")
                if t_sampling == "uniform":
                    t = torch.rand((imgs.shape[0],), device=self.device)
                elif t_sampling == "sigmoid_normal":
                    t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
                else:
                    raise ValueError(f"Unknown t_sampling: {t_sampling}")

            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)
                
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        if self.adapter_trainable:
            apply_to_pooled_prompt = self.adapter_affect_prompt and captions is None
            prompt_embeds, pooled_prompt_embeds, text_ids, adapter_tokens = self._apply_adapters(
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
                pooled_prompt_embeds_adapter=pooled_prompt_embeds_adapter,
                apply_to_pooled_prompt=apply_to_pooled_prompt,
                thermal_image_tokens=thermal_image_tokens,
                thermal_caption_tokens=thermal_caption_tokens,
            )
        else:
            with torch.no_grad():
                apply_to_pooled_prompt = self.adapter_affect_prompt and captions is None
                prompt_embeds, pooled_prompt_embeds, text_ids, adapter_tokens = self._apply_adapters(
                    prompt_embeds,
                    pooled_prompt_embeds,
                    text_ids,
                    pooled_prompt_embeds_adapter=pooled_prompt_embeds_adapter,
                    apply_to_pooled_prompt=apply_to_pooled_prompt,
                    thermal_image_tokens=thermal_image_tokens,
                    thermal_caption_tokens=thermal_caption_tokens,
                )

        # Forward pass
        hidden_states = torch.cat((x_t, x_cond), dim=2)
        if self.thermal_adapter_ca_enabled:
            adapter_token_mask = None
            if mask_tokens is not None:
                adapter_token_mask = mask_tokens
                if adapter_token_mask.ndim == 3:
                    adapter_token_mask = adapter_token_mask.mean(dim=-1)
                adapter_token_mask = adapter_token_mask.to(
                    device=prompt_embeds.device,
                    dtype=prompt_embeds.dtype,
                ).clamp(0, 1)
            transformer_out = tranformer_forward(
                self.transformer,
                condition_latents=None,
                condition_ids=None,
                condition_type_ids=None,
                model_config=self.model_config,
                c_t=0,
                adapter_tokens=adapter_tokens,
                adapter_token_mask=adapter_token_mask,
                hidden_states=hidden_states,
                timestep=t,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )
        else:
            transformer_out = self.transformer(
                hidden_states=hidden_states,
                timestep=t,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )
        pred = transformer_out[0]

        # Compute loss
        target = (x_1 - x_0)

        # Kontext-style timestep weighting (optional).
        t_weighting = self.loss_config.get("t_weighting", None)
        if t_weighting == "flux_gaussian" and sigma_schedule == "flux_shifted_1000":
            # Mirror DiffSynth FlowMatchScheduler.set_training_weight.
            # In Kontext training, one discrete timestep is sampled for the whole batch.
            # Weight is derived from the (shifted) timestep grid.
            assert sampled_timestep_id is not None
            steps = float(schedule_steps)
            timesteps = (sigmas * schedule_steps).detach()
            y = torch.exp(-2.0 * (((timesteps - steps / 2.0) / steps) ** 2))
            y_shifted = y - y.min()
            weights = y_shifted * (steps / (y_shifted.sum() + 1e-8))
            if schedule_steps != 1000:
                weights = weights * (schedule_steps / steps)
                weights = weights + weights[1]
            per_example_weight = weights[sampled_timestep_id].expand((imgs.shape[0],))
        else:
            per_example_weight = None

        use_masked_loss = bool(self.loss_config.get("use_masked_loss", True))
        if use_masked_loss and mask_tokens is not None:
            mask = mask_tokens
            if mask.ndim == 3:
                mask = mask.mean(dim=-1)
            mask = mask.to(dtype=pred.dtype).clamp(0, 1)
            masked_w = float(self.loss_config.get("masked_weight", 1.0))
            unmasked_w = float(self.loss_config.get("unmasked_weight", 0.0))
            weights = mask * masked_w + (1.0 - mask) * unmasked_w

            loss_dtype = str(self.loss_config.get("loss_dtype", "float32")).lower()
            pred_f = pred.float() if loss_dtype == "float32" else pred
            target_f = target.float() if loss_dtype == "float32" else target
            per_token = ((pred_f - target_f) ** 2).mean(dim=-1)
            diff_loss = (per_token * weights).sum() / (weights.sum() + 1e-8)
            if per_example_weight is not None:
                # weights: [B, T], per_example_weight: [B]
                diff_loss = diff_loss * per_example_weight.mean()
            if self.debug_config.get("enabled", False) and (not self._debug_printed_step):
                masked_frac = float((mask > 0.5).to(dtype=torch.float32).mean().item())
                self._debug(
                    "masked_loss",
                    "pred.shape=", tuple(pred.shape),
                    "mask.shape=", tuple(mask.shape),
                    "masked_frac~", masked_frac,
                    "masked_w=", masked_w,
                    "unmasked_w=", unmasked_w,
                    "sigma_schedule=", sigma_schedule,
                    "t_mean=", float(t.mean().item()),
                )
        else:
            loss_dtype = str(self.loss_config.get("loss_dtype", "float32")).lower()
            pred_f = pred.float() if loss_dtype == "float32" else pred
            target_f = target.float() if loss_dtype == "float32" else target
            diff_loss = F.mse_loss(pred_f, target_f, reduction="mean")
            if per_example_weight is not None:
                diff_loss = diff_loss * per_example_weight.mean()

        diff_weight = float(self.loss_config.get("diff_weight", 1.0))
        loss = diff_loss * diff_weight

        image_loss_cfg = self.image_loss_cfg
        l1_weight = float(image_loss_cfg.get("l1_weight", 0.0))
        ssim_weight = float(image_loss_cfg.get("ssim_weight", 0.0))
        perceptual_weight = float(image_loss_cfg.get("perceptual_weight", 0.0))
        use_image_loss = (l1_weight > 0.0) or (ssim_weight > 0.0) or (perceptual_weight > 0.0)

        phys_cfg = self.loss_config.get("phys", {}) or {}
        phys_enabled = bool(phys_cfg.get("enabled", False))

        pred_right = None
        target_right = None
        if use_image_loss or phys_enabled:
            t_ = t.view(-1, 1, 1)
            x0_hat = (x_t - t_ * pred).to(dtype=pred.dtype)
            vae = self.flux_fill_pipe.vae
            height = int(imgs.shape[-2])
            width = int(imgs.shape[-1])
            latents = self.flux_fill_pipe._unpack_latents(
                x0_hat, height, width, self.flux_fill_pipe.vae_scale_factor
            )
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
            pred_img = vae.decode(latents).sample
            pred_img = (pred_img / 2 + 0.5).clamp(0, 1)

            width = imgs.shape[-1] // 2
            pred_right = pred_img[..., width:]
            target_right = imgs[..., width:].to(device=pred_right.device, dtype=pred_right.dtype)

        if use_image_loss:
            pred_img_f = pred_right.float()
            target_img_f = target_right.float()

            if l1_weight > 0.0:
                if bool(image_loss_cfg.get("l1_on_gray", True)):
                    pred_l1 = _rgb_to_gray(pred_img_f)
                    target_l1 = _rgb_to_gray(target_img_f)
                else:
                    pred_l1 = pred_img_f
                    target_l1 = target_img_f
                l1_loss = F.l1_loss(pred_l1, target_l1, reduction="mean")
                loss = loss + l1_weight * l1_loss

            if ssim_weight > 0.0:
                if bool(image_loss_cfg.get("ssim_on_gray", True)):
                    pred_ssim = _rgb_to_gray(pred_img_f)
                    target_ssim = _rgb_to_gray(target_img_f)
                else:
                    pred_ssim = pred_img_f
                    target_ssim = target_img_f
                window_size = int(image_loss_cfg.get("ssim_window", 11))
                sigma = float(image_loss_cfg.get("ssim_sigma", 1.5))
                ssim_value = _ssim(
                    pred_ssim,
                    target_ssim,
                    window_size=window_size,
                    sigma=sigma,
                    data_range=1.0,
                )
                loss = loss + ssim_weight * (1.0 - ssim_value)

            if perceptual_weight > 0.0:
                if self.perceptual_loss is None:
                    raise RuntimeError("perceptual_weight > 0 but perceptual_loss is not initialized.")
                loss = loss + perceptual_weight * self.perceptual_loss(pred_img_f, target_img_f)

        if phys_enabled:
            phys_weight = float(phys_cfg.get("weight", 0.1))
            gray_weight = float(phys_cfg.get("gray_weight", 0.0))
            grad_weight = float(phys_cfg.get("grad_weight", 0.0))
            t_weighting = str(phys_cfg.get("t_weighting", "none")).lower()
            t_power = float(phys_cfg.get("t_power", 1.0))

            pred_y = _rgb_to_gray(pred_right).squeeze(1)
            target_y = _rgb_to_gray(target_right).squeeze(1)
            phys_loss = F.l1_loss(pred_y, target_y, reduction="mean")

            if gray_weight > 0.0:
                gray_loss = (
                    (pred_right[:, 0] - pred_right[:, 1]).abs().mean()
                    + (pred_right[:, 1] - pred_right[:, 2]).abs().mean()
                )
                loss = loss + gray_weight * gray_loss

            if grad_weight > 0.0:
                grad_x_pred = pred_y[:, :, 1:] - pred_y[:, :, :-1]
                grad_y_pred = pred_y[:, 1:, :] - pred_y[:, :-1, :]
                grad_x_tgt = target_y[:, :, 1:] - target_y[:, :, :-1]
                grad_y_tgt = target_y[:, 1:, :] - target_y[:, :-1, :]
                grad_loss = (
                    F.l1_loss(grad_x_pred, grad_x_tgt, reduction="mean")
                    + F.l1_loss(grad_y_pred, grad_y_tgt, reduction="mean")
                )
                loss = loss + grad_weight * grad_loss

            if t_weighting == "one_minus_t":
                t_scale = (1.0 - t).clamp(0, 1).pow(t_power).mean()
                phys_loss = phys_loss * t_scale

            loss = loss + phys_weight * phys_loss
        self.last_t = t.mean().item()
        if self.debug_config.get("enabled", False) and (not self._debug_printed_step):
            self._debug_printed_step = True
        return loss
