from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import random
import time
import numpy as np
from datasets import load_dataset

from .data import (
    EditDataset,
    OminiDataset,
    EditDataset_with_Omini,
    PairedCSVDataset,
)
from .model import OminiModel
from .callbacks import TrainingCallback


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)
        
    if 'use_offset_noise' not in config.keys():
        config['use_offset_noise'] = False

    # Initialize dataset and dataloader
    debug_cfg = training_config.get("debug", {}) or {}
    debug_enabled = bool(debug_cfg.get("enabled", False))
    
    if training_config["dataset"]["type"] == "edit":
        dataset = load_dataset('osunlp/MagicBrush')
        dataset = EditDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "omini":
        dataset = load_dataset(training_config["dataset"]["path"])
        dataset = OminiDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )

    elif training_config["dataset"]["type"] == "edit_with_omini":
        omni = load_dataset("parquet", data_files=os.path.abspath(training_config["dataset"]["path"]), split="train")
        magic = load_dataset('osunlp/MagicBrush')
        dataset = EditDataset_with_Omini(
            magic,
            omni,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )

    elif training_config["dataset"]["type"] == "paired_csv":
        ds_cfg = training_config["dataset"]
        dataset = PairedCSVDataset(
            base_path=os.path.abspath(ds_cfg.get("base_path", ".")),
            metadata_path=os.path.abspath(ds_cfg["metadata_path"]),
            condition_size=ds_cfg.get("condition_size", 512),
            target_size=ds_cfg.get("target_size", 512),
            drop_text_prob=ds_cfg.get("drop_text_prob", 0.0),
            return_pil_image=ds_cfg.get("return_pil_image", False),
            debug=bool(ds_cfg.get("debug", False)) or debug_enabled,
            source_key=ds_cfg.get("source_key", "kontext_images"),
            target_key=ds_cfg.get("target_key", "image"),
            prompt_key=ds_cfg.get("prompt_key", "prompt"),
            default_prompt=ds_cfg.get("default_prompt", ""),
            instruction_text=ds_cfg.get("instruction_text"),
            caption_key=ds_cfg.get("caption_key"),
            caption_prefix=ds_cfg.get("caption_prefix", ""),
            default_caption=ds_cfg.get("default_caption", ""),
            prompt_prefix=ds_cfg.get(
                "prompt_prefix",
                "A diptych with two side-by-side images of the same scene. "
                "On the right, the scene is exactly the same as on the left but ",
            ),
        )
          

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_fill_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        loss_config=training_config.get("loss", None),
        adapter_config=training_config.get("adapter", None),
        debug_config=debug_cfg,
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        use_offset_noise=config["use_offset_noise"],
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
