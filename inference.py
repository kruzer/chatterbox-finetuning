import os
import torch
import numpy as np
import soundfile as sf
import random
import re
from safetensors.torch import load_file


from src.utils import setup_logger, trim_silence_with_vad
from src.config import TrainConfig
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.models.t3.t3 import T3


logger = setup_logger("Chatterbox-Inference")


cfg = TrainConfig()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_TURBO = cfg.is_turbo
BASE_MODEL_DIR = cfg.model_dir
OUTPUT_DIR = cfg.output_dir


if IS_TURBO:
    
    FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, "t3_turbo_finetuned.safetensors")
    PARAMS = {
        "temperature": 0.8,
        "exaggeration": 0.5,
        "repetition_penalty": 1.2,
    }
else:
    
    FINETUNED_WEIGHTS = os.path.join(OUTPUT_DIR, "t3_finetuned.safetensors")
    PARAMS = {
        "temperature": 0.8,
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "repetition_penalty": 1.2,
    }


TEXT_TO_SAY = "Bu, artık hem normal hem de turbo modelleri otomatik olarak destekleyen yeni çıkarım komut dosyasının bir testidir."
AUDIO_PROMPT = "./reference.wav"
OUTPUT_FILE = "./output.wav"



def load_finetuned_engine(device):
    """
    Loads the correct Chatterbox engine (Normal or Turbo) and replaces the T3 module
    with the fine-tuned version.
    """
    
    logger.info(f"Loading in {'TURBO' if IS_TURBO else 'NORMAL'} mode.")
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    EngineClass = ChatterboxTurboTTS if IS_TURBO else ChatterboxTTS

    tts_engine = EngineClass.from_local(BASE_MODEL_DIR, device="cpu")
    
    # Configure New T3 Model
    logger.info(f"Initializing new T3 with vocab size: {cfg.new_vocab_size}")
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size
  
    new_t3 = T3(hp=t3_config)

    if IS_TURBO:
        logger.info("Turbo Mode: Removing 'wte' layer from new T3 model to match fine-tuned state.")
        if hasattr(new_t3.tfmr, "wte"):
            del new_t3.tfmr.wte
    
    if os.path.exists(FINETUNED_WEIGHTS):
        logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
        state_dict = load_file(FINETUNED_WEIGHTS, device="cpu")
        new_t3.load_state_dict(state_dict, strict=True)
        logger.info("Fine-tuned weights loaded successfully.")
    else:
        logger.error(f"FATAL: Fine-tuned file not found at {FINETUNED_WEIGHTS}.")
        logger.error("Please ensure you have a trained model before running inference.")
        raise FileNotFoundError(FINETUNED_WEIGHTS)

    tts_engine.t3 = new_t3
    
    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()

    tts_engine.device = device
    
    return tts_engine


def generate_sentence_audio(engine, text, prompt_path, **kwargs):
    """Generates audio for a single sentence and trims silence."""
    try:
        wav_tensor = engine.generate(text=text, audio_prompt_path=prompt_path, **kwargs)
        wav_np = wav_tensor.squeeze().cpu().numpy()
        trimmed_wav = trim_silence_with_vad(wav_np, engine.sr)
        return engine.sr, trimmed_wav
    except Exception as e:
        logger.error(f"Error generating sentence '{text[:30]}...': {e}")
        return 24000, np.zeros(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Inference running on: {device}")
    
    engine = load_finetuned_engine(device)
    
    sentences = re.split(r'(?<=[.?!])\s+', TEXT_TO_SAY.strip())
    sentences = [s for s in sentences if s.strip()]
    
    logger.info(f"Found {len(sentences)} sentences to synthesize.")
    
    all_chunks = []
    sample_rate = 24000
    
    set_seed(42)
    
    for i, sent in enumerate(sentences):
        logger.info(f"Synthesizing ({i+1}/{len(sentences)}): {sent}")
        sr, audio_chunk = generate_sentence_audio(engine, sent, AUDIO_PROMPT, **PARAMS)
        
        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            pause_samples = int(sr * 0.2)
            all_chunks.append(np.zeros(pause_samples, dtype=np.float32))

    if all_chunks:
        final_audio = np.concatenate(all_chunks)
        sf.write(OUTPUT_FILE, final_audio, sample_rate)
        logger.info(f"Result saved to: {OUTPUT_FILE}")
    else:
        logger.error("No audio was generated.")


if __name__ == "__main__":
    main()