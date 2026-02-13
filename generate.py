"""CLI do generowania audio z dowolnego checkpointu chatterbox-finetuning."""

import argparse
import gc
import glob
import os
import re

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from src.chatterbox_.models.t3.t3 import T3
from src.chatterbox_.tts_turbo import ChatterboxTurboTTS
from src.chatterbox_.tts import ChatterboxTTS
from src.config import TrainConfig
from src.utils import setup_logger, trim_silence_with_vad

logger = setup_logger("Generate")
cfg = TrainConfig()


def load_checkpoint_engine(checkpoint_num: int, device: str):
    """Load TTS engine with weights from checkpoint-{N}/model.safetensors."""
    checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-{checkpoint_num}")
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights found in {checkpoint_dir}")

    is_turbo = cfg.is_turbo
    logger.info(f"Loading {'TURBO' if is_turbo else 'NORMAL'} from checkpoint-{checkpoint_num}")

    EngineClass = ChatterboxTurboTTS if is_turbo else ChatterboxTTS
    tts_engine = EngineClass.from_local(cfg.model_dir, device="cpu")

    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = cfg.new_vocab_size
    new_t3 = T3(hp=t3_config)

    if is_turbo and hasattr(new_t3.tfmr, "wte"):
        del new_t3.tfmr.wte

    # Load and clean state dict (same logic as inference_callback.py)
    if weights_path.endswith(".safetensors"):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    clean_state_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace("module.", "")
        if k_clean.startswith("t3."):
            clean_state_dict[k_clean.replace("t3.", "")] = v
        elif not any(x in k_clean for x in ["s3gen", "ve.", "tokenizer"]):
            clean_state_dict[k_clean] = v

    missing_keys, _ = new_t3.load_state_dict(clean_state_dict, strict=False)
    non_wte_missing = [k for k in missing_keys if "wte" not in k]
    if non_wte_missing:
        logger.warning(f"Missing keys ({len(non_wte_missing)}): {non_wte_missing[:3]}...")
    else:
        logger.info("Weights loaded successfully.")

    tts_engine.t3 = new_t3
    tts_engine.t3.to(device).eval()
    tts_engine.s3gen.to(device).eval()
    tts_engine.ve.to(device).eval()
    tts_engine.device = device
    return tts_engine


def next_sequence_number(output_dir: str, prefix: str) -> int:
    """Find next available sequence number for ckptN_NNN.wav files."""
    existing = glob.glob(os.path.join(output_dir, f"{prefix}_*.wav"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        m = re.search(rf"{re.escape(prefix)}_(\d+)\.wav$", f)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def main():
    parser = argparse.ArgumentParser(description="Generate audio from a finetuned checkpoint")
    parser.add_argument("--checkpoint", "-c", type=int, required=True, help="Checkpoint number (e.g. 5000)")
    parser.add_argument("--speaker", "-s", type=str, default=cfg.inference_prompt_path, help="Speaker reference WAV")
    parser.add_argument("--text", "-t", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    engine = load_checkpoint_engine(args.checkpoint, device)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = {
        "temperature": args.temperature,
        "exaggeration": args.exaggeration,
        "repetition_penalty": args.repetition_penalty,
    }
    if not cfg.is_turbo:
        params["cfg_weight"] = 0.5

    # Split text into sentences
    sentences = re.split(r'(?<=[.?!])\s+', args.text.strip())
    sentences = [s for s in sentences if s.strip()]
    logger.info(f"Synthesizing {len(sentences)} sentence(s)")

    all_chunks = []
    sample_rate = 24000

    with torch.no_grad():
        for i, sent in enumerate(sentences):
            logger.info(f"  ({i+1}/{len(sentences)}): {sent}")
            wav_tensor = engine.generate(text=sent, audio_prompt_path=args.speaker, **params)
            wav_np = wav_tensor.squeeze().cpu().numpy()
            trimmed = trim_silence_with_vad(wav_np, engine.sr)
            if len(trimmed) > 0:
                all_chunks.append(trimmed)
                sample_rate = engine.sr
                all_chunks.append(np.zeros(int(sample_rate * 0.2), dtype=np.float32))

    if not all_chunks:
        logger.error("No audio generated.")
        return

    final_audio = np.concatenate(all_chunks)

    # Output with autonumbering
    output_dir = os.path.join(cfg.output_dir, "inference_samples")
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"ckpt{args.checkpoint}"
    seq = next_sequence_number(output_dir, prefix)
    output_path = os.path.join(output_dir, f"{prefix}_{seq:03d}.wav")

    sf.write(output_path, final_audio, sample_rate)
    logger.info(f"Saved: {output_path}")

    # Cleanup
    del engine
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
