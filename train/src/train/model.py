import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict
import os
import prodigyopt

from ..flux.transformer import tranformer_forward
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
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
            
        self.optimizer_config = optimizer_config
        self.loss_config = loss_config or {}
        self.adapter_config = adapter_config or {}

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
        self.prompt_adapter = self._init_prompt_adapter()

        self.to(device).to(dtype)

    def _init_prompt_adapter(self):
        cfg = self.adapter_config or {}
        if not cfg.get("enabled", False):
            return None
        dim = int(cfg.get("dim", 768))
        hidden_dim = int(cfg.get("hidden_dim", dim * 4))
        scale = float(cfg.get("scale", 1.0))
        return PooledPromptAdapter(dim=dim, hidden_dim=hidden_dim, scale=scale)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = list(self.lora_layers)
        if self.prompt_adapter is not None:
            self.trainable_params += list(self.prompt_adapter.parameters())

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

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

    def step(self, batch):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            if self.prompt_adapter is not None:
                pooled_prompt_embeds = self.prompt_adapter(pooled_prompt_embeds)
            
            x_0, x_cond, img_ids, mask_tokens = encode_images_fill(
                self.flux_fill_pipe,
                imgs,
                mask_imgs,
                prompt_embeds.dtype,
                prompt_embeds.device,
                return_mask_tokens=True,
            )

            # Prepare t and x_t
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

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
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

        use_masked_loss = bool(self.loss_config.get("use_masked_loss", True))
        if use_masked_loss and mask_tokens is not None:
            mask = mask_tokens
            if mask.ndim == 3:
                mask = mask.mean(dim=-1)
            mask = mask.to(dtype=pred.dtype).clamp(0, 1)
            masked_w = float(self.loss_config.get("masked_weight", 1.0))
            unmasked_w = float(self.loss_config.get("unmasked_weight", 0.0))
            weights = mask * masked_w + (1.0 - mask) * unmasked_w

            per_token = ((pred - target) ** 2).mean(dim=-1)
            loss = (per_token * weights).sum() / (weights.sum() + 1e-8)
        else:
            loss = F.mse_loss(pred, target, reduction="mean")
        self.last_t = t.mean().item()
        return loss
