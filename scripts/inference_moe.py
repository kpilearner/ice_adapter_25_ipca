# Use the modified diffusers & peft library
import sys
import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
    
from diffusers import FluxFillPipeline

# Below is the original library
import torch
from PIL import Image
import numpy as np
import argparse
import random
import json


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
    extra[:, 2] = extra[:, 2] + torch.arange(1, extra_tokens + 1, device=extra.device, dtype=extra.dtype)
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
    def __init__(self, pooled_dim: int, token_dim: int, num_tokens: int, hidden_dim: int, scale: float = 1.0):
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


def _load_lora_weights(pipe: FluxFillPipeline, lora_path: str) -> None:
    import pathlib

    p = pathlib.Path(lora_path).expanduser()
    if p.is_file():
        parent = str(p.parent)
        weight_name = p.name
        try:
            pipe.load_lora_weights(parent, weight_name=weight_name)
        except TypeError:
            pipe.load_lora_weights(parent)
        return
    pipe.load_lora_weights(str(p))
    
parser = argparse.ArgumentParser() 
parser.add_argument("--image", type=str, help="Name of the image to be edited", required=True)
parser.add_argument(
    "--instruction",
    type=str,
    default="",
    help="Instruction for editing/translation. For paired translation you can leave this empty if your prompt prefix already defines the task.",
)
parser.add_argument(
    "--caption",
    type=str,
    default="",
    help="Optional caption used only for adapter/thermal query modulation.",
)
parser.add_argument(
    "--caption-prefix",
    type=str,
    default="",
    help="Prefix injected before `--caption` when building adapter input.",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the output image")
parser.add_argument("--flux-path", type=str, default='black-forest-labs/flux.1-fill-dev', help="Path to the model")
parser.add_argument("--lora-path", type=str, default='sanaka87/ICEdit-MoE-LoRA', help="Path to the LoRA weights")
parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
parser.add_argument(
    "--prompt-prefix",
    type=str,
    default="A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but ",
    help="Prefix prompt template injected before `--instruction`.",
)
parser.add_argument(
    "--adapter-dir",
    type=str,
    default=None,
    help="Optional directory containing `prompt_adapter.pt`/`text_token_adapter.pt` and `adapter_config.json` saved during training.",
)
parser.add_argument("--debug-adapter", action="store_true", help="Print adapter load/injection debug info.")


args = parser.parse_args()
pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch.bfloat16)
_load_lora_weights(pipe, args.lora_path)

caption = args.caption.strip()
if caption:
    caption = f"{args.caption_prefix}{caption}"
else:
    caption = None

if args.adapter_dir is not None:
    adapter_dir = args.adapter_dir
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    adapter_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
    if args.debug_adapter:
        print("[DEBUG][inference] adapter_cfg:", adapter_cfg)

    adapter_affect_prompt = bool(adapter_cfg.get("affect_prompt", True))
    prompt_adapter_path = os.path.join(adapter_dir, "prompt_adapter.pt")
    token_adapter_path = os.path.join(adapter_dir, "text_token_adapter.pt")
    thermal_adapter_path = os.path.join(adapter_dir, "thermal_query.pt")
    thermal_gate_path = os.path.join(adapter_dir, "thermal_gate.pt")

    # If `adapter_config.json` is missing (older checkpoints), infer what to load.
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
    if args.debug_adapter:
        print(
            "[DEBUG][inference] adapter files:",
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
    thermal_gate = None
    thermal_gate_balance_text = False

    if os.path.exists(prompt_adapter_path) and kind in ("pooled", "both"):
        state = torch.load(prompt_adapter_path, map_location="cpu")
        dim = int(adapter_cfg.get("dim", state["net.0.weight"].shape[1]))
        hidden_dim = int(adapter_cfg.get("hidden_dim", state["net.0.weight"].shape[0]))
        prompt_adapter = PooledPromptAdapter(dim=dim, hidden_dim=hidden_dim, scale=scale)
        prompt_adapter.load_state_dict(state)
        prompt_adapter.eval()
        if args.debug_adapter:
            print("[DEBUG][inference] loaded prompt_adapter:", prompt_adapter_path)

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
        if args.debug_adapter:
            print("[DEBUG][inference] loaded text_token_adapter:", token_adapter_path)
            print(
                "[DEBUG][inference] token_adapter dims: pooled_dim=",
                pooled_dim,
                "token_dim=",
                token_dim,
                "num_tokens=",
                num_tokens,
                "hidden_dim=",
                token_hidden_dim,
            )

    thermal_cfg = adapter_cfg.get("thermal", {}) or {}
    thermal_enabled = bool(thermal_cfg.get("enabled", False))
    if not thermal_enabled and os.path.exists(thermal_adapter_path):
        thermal_enabled = True
    if thermal_enabled:
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
            if args.debug_adapter:
                print("[DEBUG][inference] loaded thermal_query:", thermal_adapter_path)
        elif args.debug_adapter:
            print("[DEBUG][inference] thermal_query enabled but file missing:", thermal_adapter_path)

        gate_cfg = thermal_cfg.get("gate", {}) or {}
        gate_enabled = bool(gate_cfg.get("enabled", False))
        gate_path_exists = os.path.exists(thermal_gate_path)
        if not gate_enabled and gate_path_exists:
            gate_enabled = True
        if gate_enabled and not gate_path_exists:
            gate_enabled = False
            if args.debug_adapter:
                print("[DEBUG][inference] thermal_gate enabled but missing file:", thermal_gate_path)
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
            if args.debug_adapter:
                print("[DEBUG][inference] loaded thermal_gate:", thermal_gate_path)

    if prompt_adapter is not None or token_adapter is not None or thermal_adapter is not None or thermal_gate is not None:
        orig_encode_prompt = pipe.encode_prompt

        def encode_prompt_wrapped(*encode_args, **encode_kwargs):
            prompt_embeds, pooled_prompt_embeds, text_ids = orig_encode_prompt(*encode_args, **encode_kwargs)
            adapter_pooled = pooled_prompt_embeds
            if caption is not None:
                caption_kwargs = dict(encode_kwargs)
                caption_kwargs.pop("prompt_embeds", None)
                caption_kwargs.pop("pooled_prompt_embeds", None)
                caption_kwargs.pop("text_ids", None)
                if encode_args:
                    caption_args = list(encode_args)
                    caption_args[0] = caption
                    _, adapter_pooled, _ = orig_encode_prompt(*caption_args, **caption_kwargs)
                else:
                    caption_kwargs["prompt"] = caption
                    _, adapter_pooled, _ = orig_encode_prompt(**caption_kwargs)

            if prompt_adapter is not None:
                _ensure_module_on(prompt_adapter, adapter_pooled)
                adapter_pooled = prompt_adapter(adapter_pooled)
                if adapter_affect_prompt and caption is None:
                    pooled_prompt_embeds = adapter_pooled

            gate = None
            if thermal_gate is not None:
                _ensure_module_on(thermal_gate, adapter_pooled)
                gate = thermal_gate(adapter_pooled)

            if thermal_adapter is not None:
                _ensure_module_on(thermal_adapter, adapter_pooled)
                thermal_tokens = thermal_adapter(
                    adapter_pooled.shape[0],
                    target_dim=int(prompt_embeds.shape[-1]),
                    device=adapter_pooled.device,
                    dtype=adapter_pooled.dtype,
                )
                if gate is not None:
                    thermal_tokens = _apply_gate(
                        thermal_tokens,
                        gate.to(dtype=thermal_tokens.dtype),
                        thermal_gate.mode,
                        thermal_gate.heads,
                    )
                prompt_embeds = torch.cat([prompt_embeds, thermal_tokens.to(dtype=prompt_embeds.dtype)], dim=1)
                text_ids = _extend_txt_ids(text_ids, thermal_tokens.shape[1])
                if args.debug_adapter:
                    print(
                        "[DEBUG][inference] thermal_tokens injected",
                        "tokens.shape=",
                        tuple(thermal_tokens.shape),
                    )

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
                prompt_embeds = torch.cat([prompt_embeds, tokens], dim=1)
                text_ids = _extend_txt_ids(text_ids, tokens.shape[1])
                if args.debug_adapter:
                    print(
                        "[DEBUG][inference] token_adapter injected",
                        "tokens.shape=", tuple(tokens.shape),
                        "prompt_embeds.shape=", tuple(prompt_embeds.shape),
                        "text_ids.shape=", tuple(text_ids.shape),
                    )

            return prompt_embeds, pooled_prompt_embeds, text_ids

        pipe.encode_prompt = encode_prompt_wrapped

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload() 
else:
    pipe = pipe.to("cuda")

image = Image.open(args.image)
image = image.convert("RGB")

if image.size[0] != 512:
    print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
    new_width = 512
    scale = new_width / image.size[0]
    new_height = int(image.size[1] * scale)
    new_height = (new_height // 8) * 8  
    image = image.resize((new_width, new_height))
    print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")

instruction = args.instruction

print(f"Instruction: {instruction}")
instruction = f"{args.prompt_prefix}{instruction}"

width, height = image.size
combined_image = Image.new("RGB", (width * 2, height))
combined_image.paste(image, (0, 0))
combined_image.paste(image, (width, 0))
mask_array = np.zeros((height, width * 2), dtype=np.uint8)
mask_array[:, width:] = 255 
mask = Image.fromarray(mask_array)

result_image = pipe(
    prompt=instruction,
    image=combined_image,
    mask_image=mask,
    height=height,
    width=width * 2,
    guidance_scale=50,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(args.seed) if args.seed is not None else None,
).images[0]

result_image = result_image.crop((width,0,width*2,height))

os.makedirs(args.output_dir, exist_ok=True)

image_name = args.image.split("/")[-1]
result_image.save(os.path.join(args.output_dir, f"{image_name}"))
print(f"\033[92mResult saved as {os.path.abspath(os.path.join(args.output_dir, image_name))}\033[0m")
