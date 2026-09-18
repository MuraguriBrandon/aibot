"""Local biometric voice authentication for the JARVIS assistant.

This module is intentionally designed to run entirely on the local machine. It stores a
reference speaker embedding on disk and verifies a live recording against it using cosine
similarity. SpeechBrain's ECAPA-TDNN is preferred when available, but the module also
includes a deterministic fallback embedding so the project remains functional even when a
pretrained model has not yet been downloaded.

The fallback embedding is not a production-grade biometric engine, but it is a useful local
prototype for a private assistant pipeline and keeps the code testable in constrained
environments.
"""

from __future__ import annotations

import io
import json
import logging
import os
import wave
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional for CPU-only environments
    torch = None

try:
    from scipy.io import wavfile
except ImportError:  # pragma: no cover
    wavfile = None

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:  # pragma: no cover
    SpeakerRecognition = None

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover
    sr = None

logger = logging.getLogger("jarvis.voice_auth")

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
PROFILE_PATH = Path(os.getenv("VOICE_PROFILE_PATH", "user_voice_profile.npy"))


class VoiceAuthenticator:
    """Locally verifies that the current speaker matches the enrolled owner profile."""

    def __init__(
        self,
        profile_path: Union[str, os.PathLike[str], None] = None,
        threshold: float = SIMILARITY_THRESHOLD,
        device_index: int = -1,
    ) -> None:
        self.profile_path = Path(profile_path) if profile_path else PROFILE_PATH
        self.threshold = threshold
        self.device_index = device_index
        self._speaker_model: Any = None
        self._model_cache_dir = Path(__file__).resolve().parent / "speechbrain_models"
        self._model_cache_dir.mkdir(exist_ok=True, parents=True)

    def _audio_to_float32(self, audio: Any, sample_rate: int = 16000) -> np.ndarray:
        """Normalize incoming audio to a 1D float32 NumPy array."""
        if isinstance(audio, np.ndarray):
            arr = audio.astype(np.float32, copy=False)
        elif isinstance(audio, (bytes, bytearray)):
            arr = self._bytes_to_float32(audio)
        elif hasattr(audio, "get_wav_data"):
            arr = self._bytes_to_float32(audio.get_wav_data())
        elif isinstance(audio, (list, tuple)):
            arr = np.asarray(audio, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)!r}")

        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.size == 0:
            raise ValueError("Audio input is empty.")
        arr = np.asarray(arr, dtype=np.float32)
        arr = arr / max(float(np.max(np.abs(arr))), 1e-6)
        if sample_rate != 16000:
            arr = self._resample_audio(arr, sample_rate, 16000)
        return arr.astype(np.float32)

    def _bytes_to_float32(self, wav_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes into a float32 NumPy array."""
        with io.BytesIO(wav_bytes) as bio:
            with wave.open(bio, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                frames = wav_file.readframes(wav_file.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16)
                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels)
                    samples = samples.mean(axis=1)
                audio = samples.astype(np.float32) / 32768.0
                return self._resample_audio(audio, sample_rate, 16000)

    def _resample_audio(self, audio: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
        """Resample audio to a common sample rate for consistent comparisons."""
        if in_rate == out_rate:
            return np.asarray(audio, dtype=np.float32)
        if wavfile is None:
            return np.asarray(audio, dtype=np.float32)
        if audio.size < 2:
            return np.asarray(audio, dtype=np.float32)
        # scipy-based resampling is the practical fallback when needed.
        sample_count = int(len(audio) * out_rate / in_rate)
        if sample_count < 1:
            return np.asarray(audio, dtype=np.float32)
        import scipy.signal as signal
        new_audio = signal.resample(audio, sample_count)
        return np.asarray(new_audio, dtype=np.float32)

    def _fallback_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Generate a compact spectral embedding when the pretrained model is unavailable.

        This is not a production-grade biometric model, but it gives the system a deterministic,
        CPU-only local baseline for early prototypes and testing.
        """
        if audio.size == 0:
            raise ValueError("Audio input is empty.")

        frame_size = 512
        frame_hop = 256
        features: List[float] = []
        for start in range(0, len(audio) - frame_size + 1, frame_hop):
            frame = audio[start:start + frame_size]
            if len(frame) < frame_size:
                continue
            window = np.hanning(frame_size)
            signal = frame * window
            spectrum = np.abs(np.fft.rfft(signal))
            bins = spectrum[:64]
            mean_mag = float(np.mean(bins))
            std_mag = float(np.std(bins))
            peak_mag = float(np.max(bins))
            features.extend([mean_mag, std_mag, peak_mag])
            features.extend(bins.tolist())

        if not features:
            return np.zeros(64, dtype=np.float32)
        embedding = np.asarray(features[:256], dtype=np.float32)
        embedding = embedding / max(np.linalg.norm(embedding), 1e-6)
        return embedding.astype(np.float32)

    def _load_speaker_model(self):
        """Lazy-load the SpeechBrain ECAPA-TDNN model if installed."""
        if SpeakerRecognition is None:
            logger.warning("speechbrain is not installed; using local fallback embedding extraction.")
            return None

        if self._speaker_model is None:
            try:
                self._speaker_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(self._model_cache_dir),
                    run_opts={"device": "cpu"},
                )
            except Exception as exc:  # pragma: no cover - network/model-download dependent
                logger.warning("Unable to initialize SpeechBrain ECAPA-TDNN model: %s", exc)
                self._speaker_model = None
        return self._speaker_model

    def extract_embedding(self, audio_data: Any, sample_rate: int = 16000) -> np.ndarray:
        """Return a dense embedding vector for the provided audio input."""
        audio = self._audio_to_float32(audio_data, sample_rate=sample_rate)

        model = self._load_speaker_model()
        if model is not None:
            try:
                if torch is not None:
                    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                    if hasattr(model, "encode_batch"):
                        embedding = model.encode_batch(tensor)
                    elif hasattr(model, "encode"):
                        embedding = model.encode(tensor)
                    else:
                        raise AttributeError("Speaker model must expose encode_batch or encode")
                    embedding_array = np.asarray(embedding).reshape(-1).astype(np.float32)
                    if embedding_array.size > 0:
                        norm = np.linalg.norm(embedding_array)
                        if norm > 0:
                            embedding_array = embedding_array / norm
                        return embedding_array.astype(np.float32)
            except Exception as exc:  # pragma: no cover - runtime model errors are handled below
                logger.warning("SpeechBrain embedding extraction failed; using fallback embedding: %s", exc)

        return self._fallback_embedding(audio, sample_rate=sample_rate)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise ValueError(f"Embedding size mismatch: {a.size} != {b.size}")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, similarity))

    def save_profile(self, embedding: np.ndarray, profile_path: Optional[Union[str, os.PathLike[str]]] = None) -> Path:
        """Persist a normalized embedding to disk for later verification."""
        target = Path(profile_path) if profile_path else self.profile_path
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, np.asarray(embedding, dtype=np.float32))
        logger.info("Voice profile saved to %s", target)
        return target

    def load_profile(self, profile_path: Optional[Union[str, os.PathLike[str]]] = None) -> np.ndarray:
        """Load the local voice reference embedding."""
        target = Path(profile_path) if profile_path else self.profile_path
        if not target.exists():
            raise FileNotFoundError(f"Voice profile not found: {target}")
        data = np.load(target)
        return np.asarray(data, dtype=np.float32)

    def enroll_user(self, record_count: int = 4, phrase_prompt: str = "Jarvis, register my voice") -> np.ndarray:
        """Record a few short phrases and average them into one enrollment embedding."""
        if sr is None:
            raise RuntimeError("speech_recognition is required for voice enrollment. Install the project requirements.")

        recognizer = sr.Recognizer()
        microphone = sr.Microphone(device_index=self.device_index) if self.device_index >= 0 else sr.Microphone()
        embeddings: List[np.ndarray] = []

        for index in range(1, record_count + 1):
            logger.info("Recording sample %d/%d. Please say: %s", index, record_count, phrase_prompt)
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            embedding = self.extract_embedding(audio)
            embeddings.append(embedding)
            logger.info("Captured embedding sample %d/%d", index, record_count)

        if not embeddings:
            raise RuntimeError("No embeddings were captured during enrollment.")

        stacked = np.vstack(embeddings)
        reference = np.mean(stacked, axis=0)
        reference = reference / max(np.linalg.norm(reference), 1e-6)
        self.save_profile(reference)
        return reference

    def verify_speaker(self, audio_data: Any) -> bool:
        """Verify that the incoming voice clip matches the enrolled owner.

        Returns:
            True when the live embedding meets or exceeds the configured threshold.
        """
        if not self.profile_path.exists():
            logger.warning("[Security] No voice profile found. Please enroll the owner first.")
            return False

        try:
            reference = self.load_profile(self.profile_path)
            live_embedding = self.extract_embedding(audio_data)
            similarity = self._cosine_similarity(reference, live_embedding)
            logger.info("[Security] Speaker similarity score: %.4f (threshold %.2f)", similarity, self.threshold)
            return bool(similarity >= self.threshold)
        except Exception as exc:
            logger.exception("[Security] Speaker verification failed: %s", exc)
            return False


def enroll_voice_profile(profile_path: Optional[Union[str, os.PathLike[str]]] = None, device_index: int = -1) -> np.ndarray:
    """Convenience wrapper that records and saves the owner profile."""
    authenticator = VoiceAuthenticator(profile_path=profile_path, device_index=device_index)
    return authenticator.enroll_user()


def verify_voice_profile(audio_data: Any, profile_path: Optional[Union[str, os.PathLike[str]]] = None) -> bool:
    """Convenience wrapper for verifying a single audio sample against the saved profile."""
    authenticator = VoiceAuthenticator(profile_path=profile_path)
    return authenticator.verify_speaker(audio_data)
