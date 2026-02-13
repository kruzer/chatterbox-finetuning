from dataclasses import dataclass

@dataclass
class TrainConfig:
    # --- Paths ---
    # Directory where setup.py downloaded the files
    model_dir: str = "./pretrained_models"

    # Path to your metadata CSV (Format: ID|RawText|NormText)
    csv_path: str = "./MyTTSDataset/metadata.csv"
    metadata_path: str = "./WolneLekturyDataset/metadata.json"

    # Directory containing WAV files
    wav_dir: str = "./WolneLekturyDataset/wavs"

    preprocessed_dir = "./preprocess"  # Local storage - fast I/O!
    
    # Output directory for the finetuned model
    output_dir: str = "./chatterbox_output"

    is_inference = True  # Enable inference callback to generate samples during training
    inference_prompt_path: str = "./speaker_reference/1984.wav"
    inference_test_text: str = "Dzień dobry, witam w polskim systemie syntezy mowy."  # Polish test text


    ljspeech = False # Set True if the dataset format is ljspeech, and False if it's file-based.
    json_format = True # Set True if the dataset format is json, and False if it's file-based or ljspeech.
    preprocess = False # If you've already done preprocessing once, set it to false.
    is_turbo: bool = True # Set True if you're training Turbo, False if you're training Normal.

    # --- Vocabulary ---
    # 50276 = original T3 Turbo vocab (no added tokens, pure GPT-2)
    # Set to 52260 if using multilingual vocab merge from setup.py
    new_vocab_size: int = 50276 if is_turbo else 2454

    # --- Hyperparameters ---
    batch_size: int = 32        # łagodniejsze gradienty
    grad_accum: int = 2         # effective batch = 64
    learning_rate: float = 5e-6 # kompromis: 1e-5 dawało mode collapse, 3e-6 za wolno
    num_epochs: int = 30        # peak powinien być epoka 15-25

    save_steps: int = 200       # gęstsze snapshoty
    save_total_limit: int = 30  # zachowaj wszystkie checkpointy
    eval_split: float = 0.05   # 5% danych na walidację
    dataloader_num_workers: int = 8  # A100 80GB

    # --- Constraints ---
    start_text_token = 255
    stop_text_token = 0
    max_text_len: int = 256
    max_speech_len: int = 850   # Truncates very long audio
    prompt_duration: float = 3.0 # Duration for the reference prompt (seconds)
