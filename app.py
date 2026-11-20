import base64
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import noisereduce as nr
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS
from rapidfuzz import fuzz

# Optional local Whisper model (loaded once at startup)
try:
    from faster_whisper import WhisperModel
    WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
except Exception:
    WHISPER_MODEL = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="Punjabi Farmer Voice Helpline (H6)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def load_json(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


MANDI_DATA = load_json("mandi_prices.json")
SCHEMES_DATA = load_json("schemes.json")

WEATHER_DATA = {
    "Amritsar": {
        "condition_pa": "ਅੱਜ ਅੰਮ੍ਰਿਤਸਰ ਵਿੱਚ ਮੌਸਮ ਸਾਫ਼ ਅਤੇ ਹਲਕੀ ਠੰਢ ਵਾਲਾ ਹੈ",
        "temp_c": 24,
        "rain_chance_pct": 10,
    }
}


def denoise_audio_if_wav(input_bytes: bytes, suffix: str = ".wav") -> str:
    """Saves audio to temp file; cleans background noise if readable as WAV."""
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_in.write(input_bytes)
    tmp_in.close()

    try:
        data, sr = sf.read(tmp_in.name)
        if data.ndim > 1:
            data = data.mean(axis=1)
        reduced = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.75)
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp_out.name, reduced, sr)
        os.unlink(tmp_in.name)
        return tmp_out.name
    except Exception:
        return tmp_in.name


def transcribe_audio(audio_path: str) -> str:
    if WHISPER_MODEL is None:
        return ""
    segments, _ = WHISPER_MODEL.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def find_best_match(query: str, records: list[dict], score_cutoff: int = 68) -> Optional[dict]:
    q_low = query.lower()
    best_rec = None
    best_score = 0

    for rec in records:
        for alias in rec.get("aliases", []):
            a_low = alias.lower()
            if a_low in q_low:
                return rec
            score = fuzz.partial_ratio(a_low, q_low)
            if score > best_score:
                best_score = score
                best_rec = rec

    return best_rec if best_score >= score_cutoff else None


def build_grounded_punjabi_response(transcript: str) -> tuple[str, dict]:
    """Strictly grounded in local datasets—never invents prices or scheme facts."""
    q = transcript.strip()
    if not q:
        return (
            "ਮਾਫ਼ ਕਰਨਾ, ਮੈਨੂੰ ਤੁਹਾਡੀ ਆਵਾਜ਼ ਸਾਫ਼ ਨਹੀਂ ਸੁਣੀ। ਕਿਰਪਾ ਕਰਕੇ ਫਸਲ ਜਾਂ ਯੋਜਨਾ ਦਾ ਨਾਮ ਦੁਬਾਰਾ ਬੋਲੋ।",
            {"matched_type": "none"},
        )

    # 1. Weather
    weather_words = ["weather", "mausam", "ਮੌਸਮ", "मौसम", "barish", "ਮੀਂਹ", "temperature"]
    if any(w in q.lower() for w in weather_words):
        w = WEATHER_DATA["Amritsar"]
        reply = (
            f"{w['condition_pa']}। ਤਾਪਮਾਨ ਲਗਭਗ {w['temp_c']} ਡਿਗਰੀ ਸੈਲਸੀਅਸ ਹੈ "
            f"ਅਤੇ ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ {w['rain_chance_pct']} ਪ੍ਰਤੀਸ਼ਤ ਹੈ।"
        )
        return reply, {"matched_type": "weather", "record": w}

    # 2. Mandi prices
    crop_rec = find_best_match(q, MANDI_DATA, score_cutoff=68)
    if crop_rec:
        reply = (
            f"{crop_rec['mandi']} ਮੰਡੀ ਵਿੱਚ {crop_rec['aliases'][0]} ਦਾ ਭਾਅ "
            f"{crop_rec['price_per_quintal_inr']} ਰੁਪਏ ਪ੍ਰਤੀ ਕੁਇੰਟਲ ਦਰਜ ਹੈ "
            f"(ਮਿਤੀ {crop_rec['date']})।"
        )
        return reply, {"matched_type": "mandi", "record": crop_rec}

    # 3. Schemes
    scheme_rec = find_best_match(q, SCHEMES_DATA, score_cutoff=68)
    if scheme_rec:
        reply = f"{scheme_rec['title_pa']}: {scheme_rec['summary_pa']}"
        return reply, {"matched_type": "scheme", "record": scheme_rec}

    # 4. Safe fallback (no hallucinated numbers)
    return (
        "ਮਾਫ਼ ਕਰਨਾ, ਇਸ ਸਵਾਲ ਲਈ ਮੇਰੇ ਡਾਟਾ ਵਿੱਚ ਪੱਕੀ ਜਾਣਕਾਰੀ ਮੌਜੂਦ ਨਹੀਂ ਹੈ। "
        "ਤੁਸੀਂ ਕਣਕ, ਝੋਨਾ, ਆਲੂ ਦੇ ਭਾਅ, ਮੌਸਮ ਜਾਂ ਕਿਸਾਨ ਯੋਜਨਾ ਬਾਰੇ ਪੁੱਛ ਸਕਦੇ ਹੋ।",
        {"matched_type": "fallback"},
    )


def synthesize_punjabi_tts_base64(text_pa: str) -> str:
    tts = gTTS(text=text_pa, lang="pa", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/helpline")
async def helpline(
    audio: Optional[UploadFile] = File(default=None),
    text_query: Optional[str] = Form(default=None),
):
    t0 = time.perf_counter()
    clean_path = None

    try:
        if audio is not None:
            raw_bytes = await audio.read()
            suffix = Path(audio.filename or "input.wav").suffix or ".wav"
            clean_path = denoise_audio_if_wav(raw_bytes, suffix=suffix)
            t_stt_start = time.perf_counter()
            transcript = transcribe_audio(clean_path)
            stt_ms = round((time.perf_counter() - t_stt_start) * 1000)
        else:
            transcript = (text_query or "").strip()
            stt_ms = 0

        t_logic_start = time.perf_counter()
        reply_pa, meta = build_grounded_punjabi_response(transcript)
        logic_ms = round((time.perf_counter() - t_logic_start) * 1000)

        t_tts_start = time.perf_counter()
        audio_b64 = synthesize_punjabi_tts_base64(reply_pa)
        tts_ms = round((time.perf_counter() - t_tts_start) * 1000)

        total_sec = round(time.perf_counter() - t0, 2)
        return {
            "transcript": transcript,
            "reply_punjabi": reply_pa,
            "audio_base64": audio_b64,
            "grounding": meta,
            "timing_ms": {"stt": stt_ms, "grounding": logic_ms, "tts": tts_ms},
            "total_latency_sec": total_sec,
            "within_6s_target": total_sec < 6.0,
        }
    finally:
        if clean_path and os.path.exists(clean_path):
            try:
                os.unlink(clean_path)
            except OSError:
                pass