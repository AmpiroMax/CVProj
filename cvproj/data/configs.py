from typing import List, Optional

from pydantic import BaseModel


class GenerationConfig(BaseModel):
    input_image_path: Optional[str] = None
    output_image_path: Optional[str] = None
    gen_type: str = "edge"
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200
    device: str = "mps"
    seed: int = 42
    lora_gamma: float = 0.4


class TrainConfig(BaseModel):
    # ===================== Logging =====================
    tracker_project_name: str = "CVProj Train Pix2Pix"
    logger_type: str = "wandb"
    output_dir: str = "log"
    track_fid_metrci_val: bool = False
    image_log_freq: int = 100
    model_log_freq: int = 100
    eval_freq: int = 100

    # ===================== Model =====================
    lora_rank_unet: int = 8
    lora_rank_vae: int = 4
    resolution: int = 256

    # ===================== Train =====================
    learning_rate: float = 5e-6
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 0.99
    train_batch_size: int = 4
    train_dataloader_num_workers: int = 0
    eval_batch_size: int = 1
    epoch_num: int = 100
    grad_clip: float = 1.0

    # ===================== Losses weights =====================
    l_rec: float = 1
    l_lpips: float = 1
    l_clipsim: float = 4

    # ===================== Dataset =====================
    dataset_type: str = "pokemon"
    dataset_folder: str = None

    # ===================== Device =====================
    device: str = "mps"
    seed: int = 42
