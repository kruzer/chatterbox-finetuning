"""FastAPI server do testowania checkpointów chatterbox-finetuning.

Uruchomienie:
    uv run server.py          # port 7860
    uv run server.py --port 8080
"""

import gc
import glob
import os
import re
import time
import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.chatterbox_.models.s3gen import S3GEN_SR
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.chatterbox_.models.t3.modules.cond_enc import T3Cond
from src.chatterbox_.tts_turbo import Conditionals
from src.config import TrainConfig
from src.utils import setup_logger, trim_silence_with_vad
from generate import load_checkpoint_engine

logger = setup_logger("Server")
cfg = TrainConfig()
app = FastAPI()

AUDIO_OUT_DIR = os.path.join(cfg.output_dir, "inference_samples")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# Aktualnie załadowany engine (max 1 w pamięci)
_engine: object = None
_engine_checkpoint: int | None = None

# Cache embeddingów speakera: {abs_path: {ve_embed, tokens, s3gen_ref}}
# VE i S3Gen są zawsze z pretrained — embeddingi są ważne między checkpointami
_speaker_cache: dict[str, dict] = {}

# Historia generacji (in-memory, newest first)
_history: list[dict] = []


# ---------------------------------------------------------------------------
# Engine management
# ---------------------------------------------------------------------------

def get_engine(checkpoint_num: int):
    """Zwróć silnik dla danego checkpointu. Ładuje jeśli inny / brak."""
    global _engine, _engine_checkpoint

    if _engine is not None and _engine_checkpoint == checkpoint_num:
        return _engine

    # Zwolnij stary silnik
    if _engine is not None:
        logger.info(f"Unloading checkpoint-{_engine_checkpoint}")
        del _engine
        _engine = None
        gc.collect()
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading checkpoint-{checkpoint_num} on {device}")
    _engine = load_checkpoint_engine(checkpoint_num, device)
    _engine_checkpoint = checkpoint_num
    return _engine


# ---------------------------------------------------------------------------
# Speaker embedding cache
# ---------------------------------------------------------------------------

def precompute_speaker(engine, speaker_path: str) -> dict:
    """Oblicz i zapamiętaj embeddingi dla pliku referencyjnego.

    Zwraca dict z kluczami: ve_embed, t3_cond_prompt_tokens, s3gen_ref_dict.
    Drogie operacje (VE forward, S3Gen embed_ref, tokenizacja audio) wykonywane
    tylko raz na plik — wynik jest ważny dla wszystkich checkpointów (VE/S3Gen
    to zawsze te same wagi pretrained).
    """
    abs_path = os.path.abspath(speaker_path)
    if abs_path in _speaker_cache:
        logger.info(f"Speaker cache hit: {Path(abs_path).name}")
        return _speaker_cache[abs_path]

    logger.info(f"Precomputing speaker embeddings: {Path(abs_path).name}")
    t0 = time.perf_counter()

    # 1. Wczytaj i znormalizuj audio (ta sama logika co prepare_conditionals)
    s3gen_ref_wav, _ = librosa.load(abs_path, sr=S3GEN_SR)
    s3gen_ref_wav = s3gen_ref_wav.astype(np.float32)

    # Normalizacja głośności (opcjonalna, ale spójna z generate())
    try:
        import pyloudnorm as ln
        meter = ln.Meter(S3GEN_SR)
        loudness = meter.integrated_loudness(s3gen_ref_wav)
        target_lufs = -24.0
        import math
        gain_db = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)
        if math.isfinite(gain_linear) and gain_linear > 0.0:
            s3gen_ref_wav = s3gen_ref_wav * gain_linear
    except Exception as e:
        logger.warning(f"norm_loudness skipped: {e}")

    ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
    ref_16k_wav = ref_16k_wav.astype(np.float32)

    # 2. S3Gen: embed ref (używa DEC_COND_LEN — pierwsze ~10s przy 22050Hz)
    dec_cond_len = getattr(engine, "DEC_COND_LEN", int(10 * S3GEN_SR))
    s3gen_ref_slice = s3gen_ref_wav[:dec_cond_len]
    s3gen_ref_dict = engine.s3gen.embed_ref(s3gen_ref_slice, S3GEN_SR, device=engine.device)

    # 3. Tokenizacja audio dla speech_cond_prompt (T3)
    t3_cond_prompt_tokens = None
    if plen := engine.t3.hp.speech_cond_prompt_len:
        enc_cond_len = getattr(engine, "ENC_COND_LEN", int(10 * S3_SR))
        s3_tokzr = engine.s3gen.tokenizer
        tokens, _ = s3_tokzr.forward([ref_16k_wav[:enc_cond_len]], max_len=plen)
        t3_cond_prompt_tokens = torch.atleast_2d(tokens).to(engine.device)

    # 4. VE: speaker embedding (najdroższy krok)
    ve_embed_np = engine.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
    ve_embed = torch.from_numpy(ve_embed_np).float().mean(axis=0, keepdim=True).to(engine.device)

    elapsed = time.perf_counter() - t0
    logger.info(f"Speaker precomputed in {elapsed:.2f}s")

    result = {
        "ve_embed": ve_embed,
        "t3_cond_prompt_tokens": t3_cond_prompt_tokens,
        "s3gen_ref_dict": s3gen_ref_dict,
    }
    _speaker_cache[abs_path] = result
    return result


def build_conditionals(engine, speaker_data: dict, exaggeration: float) -> Conditionals:
    """Zbuduj Conditionals z cache'owanych komponentów + aktualnym exaggeration."""
    t3_cond = T3Cond(
        speaker_emb=speaker_data["ve_embed"],
        cond_prompt_speech_tokens=speaker_data["t3_cond_prompt_tokens"],
        emotion_adv=exaggeration * torch.ones(1, 1, 1).to(engine.device),
    ).to(device=engine.device)
    return Conditionals(t3_cond, speaker_data["s3gen_ref_dict"])


# ---------------------------------------------------------------------------
# Generacja audio
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    checkpoint: int
    text: str
    speaker: str
    temperature: float = 0.3
    repetition_penalty: float = 1.2
    seed: int = 42


def _do_generate(req: GenerateRequest) -> dict:
    """Synchroniczna generacja — wywoływana z endpointu POST /api/generate."""
    t_total = time.perf_counter()

    engine = get_engine(req.checkpoint)

    if not os.path.exists(req.speaker):
        raise HTTPException(status_code=400, detail=f"Speaker file not found: {req.speaker}")

    # Seed
    torch.manual_seed(req.seed)
    np.random.seed(req.seed)

    # Precompute speaker embeddings (z cache'u jeśli dostępne)
    speaker_data = precompute_speaker(engine, req.speaker)

    # Ustaw conds (exaggeration=0.5 to domyślna wartość, w Turbo i tak ignorowana)
    engine.conds = build_conditionals(engine, speaker_data, 0.5)

    # Podziel tekst na zdania
    sentences = re.split(r"(?<=[.?!])\s+", req.text.strip())
    sentences = [s for s in sentences if s.strip()]
    logger.info(f"Synthesizing {len(sentences)} sentence(s), ckpt={req.checkpoint}")

    all_chunks = []
    sample_rate = 24000

    with torch.no_grad():
        for i, sent in enumerate(sentences):
            logger.info(f"  ({i+1}/{len(sentences)}): {sent}")
            # audio_prompt_path=None → użyj engine.conds ustawionych wyżej
            wav_tensor = engine.generate(
                text=sent,
                audio_prompt_path=None,
                temperature=req.temperature,
                repetition_penalty=req.repetition_penalty,
            )
            wav_np = wav_tensor.squeeze().cpu().numpy()
            trimmed = trim_silence_with_vad(wav_np, engine.sr)
            if len(trimmed) > 0:
                sample_rate = engine.sr
                all_chunks.append(trimmed)
                all_chunks.append(np.zeros(int(sample_rate * 0.2), dtype=np.float32))

    if not all_chunks:
        raise HTTPException(status_code=500, detail="No audio generated.")

    final_audio = np.concatenate(all_chunks)
    duration = len(final_audio) / sample_rate

    # Zapisz plik
    filename = f"ckpt{req.checkpoint}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(AUDIO_OUT_DIR, filename)
    sf.write(out_path, final_audio, sample_rate)
    file_size = os.path.getsize(out_path)

    elapsed = time.perf_counter() - t_total

    return {
        "filename": filename,
        "audio_url": f"/audio/{filename}",
        "duration_s": round(duration, 2),
        "gen_time_s": round(elapsed, 2),
        "file_size_kb": round(file_size / 1024, 1),
        "checkpoint": req.checkpoint,
        "temperature": req.temperature,
        "repetition_penalty": req.repetition_penalty,
        "seed": req.seed,
        "speaker": os.path.basename(req.speaker),
        "text": req.text,
    }


# ---------------------------------------------------------------------------
# Endpointy
# ---------------------------------------------------------------------------

@app.get("/api/checkpoints")
def list_checkpoints():
    """Lista dostępnych checkpointów (mają model.safetensors lub pytorch_model.bin)."""
    pattern = os.path.join(cfg.output_dir, "checkpoint-*")
    dirs = sorted(glob.glob(pattern))
    result = []
    for d in dirs:
        has_weights = (
            os.path.exists(os.path.join(d, "model.safetensors"))
            or os.path.exists(os.path.join(d, "pytorch_model.bin"))
        )
        if has_weights:
            m = re.search(r"checkpoint-(\d+)$", d)
            if m:
                result.append(int(m.group(1)))
    return sorted(result)


@app.get("/api/speakers")
def list_speakers():
    """Lista plików WAV w katalogu speaker_reference/ z długością audio."""
    ref_dir = "speaker_reference"
    if not os.path.isdir(ref_dir):
        return []
    wavs = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    result = []
    for w in wavs:
        try:
            info = sf.info(w)
            duration = round(info.duration, 1)
        except Exception:
            duration = None
        result.append({"path": os.path.abspath(w), "name": Path(w).name, "duration_s": duration})
    return result


@app.post("/api/generate")
def generate(req: GenerateRequest):
    result = _do_generate(req)
    _history.insert(0, result)
    if len(_history) > 50:
        _history.pop()
    return result


@app.get("/audio/{filename}")
def serve_audio(filename: str):
    # Zabezpieczenie przed path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(AUDIO_OUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/history")
def get_history():
    return _history


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


# ---------------------------------------------------------------------------
# Frontend (inline HTML)
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chatterbox TTS — Testowanie checkpointów</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 20px; }
  h1 { font-size: 1.3rem; margin-bottom: 20px; color: #fff; }
  .layout { display: flex; gap: 20px; max-width: 1200px; }
  .panel { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 18px; }
  .form-panel { flex: 0 0 360px; }
  .history-panel { flex: 1; overflow-y: auto; max-height: 90vh; }
  label { display: block; font-size: 0.8rem; color: #999; margin-bottom: 4px; margin-top: 14px; }
  label:first-child { margin-top: 0; }
  select, textarea, input[type="number"] {
    width: 100%; background: #252525; border: 1px solid #444; border-radius: 5px;
    color: #e0e0e0; padding: 7px 10px; font-size: 0.9rem;
  }
  textarea { resize: vertical; min-height: 80px; }
  .slider-row { display: flex; align-items: center; gap: 10px; }
  input[type="range"] { flex: 1; accent-color: #4a9eff; }
  .slider-val { min-width: 36px; text-align: right; font-size: 0.85rem; color: #aaa; }
  button#generate-btn {
    margin-top: 18px; width: 100%; padding: 10px; background: #4a9eff;
    color: #fff; border: none; border-radius: 6px; font-size: 1rem;
    cursor: pointer; font-weight: 600;
  }
  button#generate-btn:hover { background: #3a8eef; }
  button#generate-btn:disabled { background: #333; color: #666; cursor: not-allowed; }
  .status { margin-top: 10px; font-size: 0.8rem; color: #888; min-height: 18px; }
  .status.error { color: #ff6b6b; }
  h2 { font-size: 1rem; color: #ccc; margin-bottom: 12px; }
  .history-item {
    border: 1px solid #333; border-radius: 6px; padding: 14px; margin-bottom: 12px;
    background: #141414;
  }
  .history-item:first-child { border-color: #4a9eff55; background: #14203a; }
  .history-item audio { width: 100%; margin: 8px 0 10px 0; }
  .history-text { font-size: 0.85rem; color: #ccc; margin-bottom: 8px; font-style: italic; }
  .meta { display: flex; flex-wrap: wrap; gap: 6px; }
  .meta span {
    background: #252525; border: 1px solid #333; border-radius: 4px;
    padding: 2px 8px; font-size: 0.75rem; color: #aaa;
  }
  .meta span b { color: #e0e0e0; }
  .empty { color: #555; font-size: 0.9rem; text-align: center; padding: 40px 0; }
</style>
</head>
<body>
<h1>Chatterbox TTS — Testowanie checkpointów</h1>
<div class="layout">

  <div class="panel form-panel">
    <label>Checkpoint</label>
    <select id="checkpoint"><option>Ładowanie...</option></select>

    <label>Speaker reference</label>
    <select id="speaker"><option>Ładowanie...</option></select>

    <label>Tekst</label>
    <textarea id="text">Dzień dobry, witam w polskim systemie syntezy mowy.</textarea>

    <label>Temperatura: <span id="temp-val">0.30</span></label>
    <div class="slider-row">
      <input type="range" id="temperature" min="0.0" max="1.5" step="0.05" value="0.30">
    </div>

    <label>Repetition penalty: <span id="rep-val">1.20</span></label>
    <div class="slider-row">
      <input type="range" id="repetition_penalty" min="1.0" max="2.0" step="0.05" value="1.20">
    </div>

    <label>Seed</label>
    <input type="number" id="seed" value="42">

    <button id="generate-btn" onclick="doGenerate()">Generate</button>
    <div class="status" id="status"></div>
  </div>

  <div class="panel history-panel">
    <h2>Historia generacji</h2>
    <div id="history-list"><div class="empty">Brak generacji.</div></div>
  </div>

</div>

<script>
// --- Inicjalizacja dropdownów ---
async function loadCheckpoints() {
  const res = await fetch('/api/checkpoints');
  const data = await res.json();
  const sel = document.getElementById('checkpoint');
  if (!data.length) { sel.innerHTML = '<option>Brak checkpointów</option>'; return; }
  sel.innerHTML = data.slice().reverse().map(n =>
    `<option value="${n}">checkpoint-${n}</option>`
  ).join('');
}

async function loadSpeakers() {
  const res = await fetch('/api/speakers');
  const data = await res.json();
  const sel = document.getElementById('speaker');
  if (!data.length) { sel.innerHTML = '<option value="">Brak plików w speaker_reference/</option>'; return; }
  sel.innerHTML = data.map(item => {
    const dur = item.duration_s !== null ? ` (${item.duration_s}s)` : '';
    return `<option value="${item.path}">${item.name}${dur}</option>`;
  }).join('');
}

// Aktualizuj wartości sliderów
['temperature', 'repetition_penalty'].forEach(id => {
  const input = document.getElementById(id);
  const map = { temperature: 'temp-val', repetition_penalty: 'rep-val' };
  input.addEventListener('input', () => {
    document.getElementById(map[id]).textContent = parseFloat(input.value).toFixed(2);
  });
});

// --- Generacja ---
async function doGenerate() {
  const btn = document.getElementById('generate-btn');
  const statusEl = document.getElementById('status');
  const checkpoint = parseInt(document.getElementById('checkpoint').value);
  const text = document.getElementById('text').value.trim();
  const speaker = document.getElementById('speaker').value;

  if (!text) { setStatus('Wpisz tekst.', true); return; }
  if (!speaker) { setStatus('Wybierz speaker reference.', true); return; }

  btn.disabled = true;
  setStatus('Generowanie...', false);

  const body = {
    checkpoint,
    text,
    speaker,
    temperature: parseFloat(document.getElementById('temperature').value),
    repetition_penalty: parseFloat(document.getElementById('repetition_penalty').value),
    seed: parseInt(document.getElementById('seed').value),
  };

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      setStatus('Błąd: ' + (err.detail || res.status), true);
      return;
    }
    const data = await res.json();
    setStatus(`OK — ${data.duration_s}s audio, ${data.gen_time_s}s generacja`, false);
    prependHistory(data);
  } catch(e) {
    setStatus('Błąd połączenia: ' + e.message, true);
  } finally {
    btn.disabled = false;
  }
}

function setStatus(msg, isError) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status' + (isError ? ' error' : '');
}

function prependHistory(item) {
  const list = document.getElementById('history-list');
  const empty = list.querySelector('.empty');
  if (empty) empty.remove();

  const div = document.createElement('div');
  div.className = 'history-item';
  div.innerHTML = `
    <div class="history-text">"${escHtml(item.text)}"</div>
    <audio controls src="${item.audio_url}"></audio>
    <div class="meta">
      <span>ckpt <b>${item.checkpoint}</b></span>
      <span>spk <b>${escHtml(item.speaker)}</b></span>
      <span>temp <b>${item.temperature}</b></span>
      <span>rep <b>${item.repetition_penalty}</b></span>
      <span>seed <b>${item.seed}</b></span>
      <span>dur <b>${item.duration_s}s</b></span>
      <span>gen <b>${item.gen_time_s}s</b></span>
      <span><b>${item.file_size_kb}KB</b></span>
    </div>
  `;
  list.prepend(div);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// --- Ładowanie historii po odświeżeniu strony ---
async function loadHistory() {
  const res = await fetch('/api/history');
  const data = await res.json();
  data.forEach(item => prependHistory(item));
}

loadCheckpoints();
loadSpeakers();
loadHistory();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    logger.info(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
