from dataclasses import dataclass, field
from enum import Enum


class GPU(Enum):
    RTX3060 = "rtx3060"  # 12GB VRAM, fp16 only
    T4 = "t4"            # 16GB VRAM, fp16 only (Colab free)
    A100 = "a100"        # 80GB VRAM, bf16 (Colab Pro / cloud)
    H100 = "h100"        # 80GB VRAM, bf16

# batch_size, grad_accum, bf16, fp16, dataloader_num_workers, save_steps, save_total_limit
_GPU_PROFILES = {
    GPU.RTX3060: dict(batch_size=4,  grad_accum=16, bf16=False, fp16=True,  workers=4, save_steps=100, save_total_limit=5),
    GPU.T4:      dict(batch_size=8,  grad_accum=8,  bf16=False, fp16=True,  workers=2, save_steps=100, save_total_limit=5),
    GPU.A100:    dict(batch_size=32, grad_accum=2,  bf16=True,  fp16=False, workers=8, save_steps=200, save_total_limit=30),
    GPU.H100:    dict(batch_size=48, grad_accum=2,  bf16=True,  fp16=False, workers=8, save_steps=200, save_total_limit=30),
}


@dataclass
class TrainConfig:
    # --- GPU Profile ---
    # Change this one line to switch between environments
    gpu: GPU = GPU.A100

    # --- Paths ---
    model_dir: str = "./pretrained_models"
    csv_path: str = "./MyTTSDataset/metadata.csv"
    metadata_path: str = "./WolneLekturyDataset/metadata.json"
    wav_dir: str = "./WolneLekturyDataset/wavs"
    preprocessed_dir: str = "./preprocess"
    output_dir: str = "./chatterbox_output"

    is_inference: bool = True
    inference_prompt_path: str = "./speaker_reference/1984.wav"
    inference_test_text: str = "Dzień dobry, witam w polskim systemie syntezy mowy."

    ljspeech: bool = False
    json_format: bool = True
    preprocess: bool = False
    is_turbo: bool = True

    # --- Vocabulary ---
    new_vocab_size: int = 50276 if is_turbo else 2454

    # --- Hyperparameters (GPU-independent) ---
    learning_rate: float = 5e-6
    num_epochs: int = 30
    eval_split: float = 0.05

    # --- Constraints ---
    start_text_token: int = 255
    stop_text_token: int = 0
    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = 3.0

    # --- GPU-dependent (auto-filled from profile) ---
    batch_size: int = field(init=False)
    grad_accum: int = field(init=False)
    bf16: bool = field(init=False)
    fp16: bool = field(init=False)
    dataloader_num_workers: int = field(init=False)
    save_steps: int = field(init=False)
    save_total_limit: int = field(init=False)

    def __post_init__(self):
        p = _GPU_PROFILES[self.gpu]
        self.batch_size = p["batch_size"]
        self.grad_accum = p["grad_accum"]
        self.bf16 = p["bf16"]
        self.fp16 = p["fp16"]
        self.dataloader_num_workers = p["workers"]
        self.save_steps = p["save_steps"]
        self.save_total_limit = p["save_total_limit"]
