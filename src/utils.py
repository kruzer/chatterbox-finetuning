import logging
import os
import sys
import torch
import torchaudio
import numpy as np



def setup_logger(name: str, level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


_VAD_MODEL = None
_GET_SPEECH_TIMESTAMPS = None

def load_vad_model():
    """Lazy loads the Silero VAD model."""
    
    global _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    if _VAD_MODEL is not None:
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    try:
        
        #print("Loading Silero VAD model...")
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        _GET_SPEECH_TIMESTAMPS = utils[0]
        _VAD_MODEL = model
        
        #print("Silero VAD loaded.")
        
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    except Exception as e:
        print(f"Error loading VAD: {e}")
        return None, None


def trim_silence_with_vad(audio_waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Trims silence/noise from the end of the audio using Silero VAD.
    """
    
    vad_model, get_timestamps = load_vad_model()
    if vad_model is None:
        return audio_waveform

    VAD_SR = 16000
    # Convert numpy to tensor
    audio_tensor = torch.from_numpy(audio_waveform).float()

    # Resample for VAD if necessary
    if sample_rate != VAD_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SR)
        vad_input = resampler(audio_tensor)
        
    else:
        vad_input = audio_tensor

    try:
        # Get speech timestamps
        speech_timestamps = get_timestamps(vad_input, vad_model, sampling_rate=VAD_SR)
        
        if not speech_timestamps:
            return audio_waveform

        # Get the end of the last speech chunk
        last_speech_end_vad = speech_timestamps[-1]['end']

        # Scale back to original sample rate
        scale_factor = sample_rate / VAD_SR
        cut_point = int(last_speech_end_vad * scale_factor)

        trimmed_wav = audio_waveform[:cut_point]
        
        return trimmed_wav


    except Exception as e:
        print(f"VAD trimming failed: {e}")
        return audio_waveform
    
    
    
def check_pretrained_models(model_dir="pretrained_models", mode="chatterbox"):
    """Checks for the existence of the necessary model files. """

    if mode == "chatterbox_turbo":
        required_files = [
            "ve.safetensors",
            "t3_turbo_v1.safetensors",
            "s3gen_meanflow.safetensors",
            "conds.pt",
            "vocab.json",
            "added_tokens.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "merges.txt",
            "grapheme_mtl_merged_expanded_v1.json"
        ]

    else:

        required_files = [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "conds.pt",
            "tokenizer.json"
        ]


    missing_files = []


    if not os.path.exists(model_dir):
        print(f"\nERROR: '{model_dir}' folder doesn't exist!")
        missing_files = required_files
        
    else:

        for filename in required_files:
            file_path = os.path.join(model_dir, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)


    if missing_files:
        print("\n" + "!" * 60)
        print("ATTENTION: The following model files could not be found:")
        for f in missing_files:
            print(f"   - {f}")
        
        print("\nPlease run the following command to download the models:")
        print(f" python setup.py")
        print("!" * 60 + "\n")
        return False
    
    print(f"All necessary models are available under '{model_dir}'.")
    return True