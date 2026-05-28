from __future__ import annotations

import math
import os
import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import safe_open
from scipy.signal import resample_poly

from .utils.audio_quality import LogMelSpectrogram, create_audio_quality_model

class AudioQualityRouter:
    DEFAULT_CHECKPOINT = "ckpt/Mega-ASR/audio_quality_router/best_acc_model.safetensors"

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        *,
        device: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self.checkpoint_path = str(
            Path(checkpoint_path or self.DEFAULT_CHECKPOINT).expanduser()
        )
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.threshold = threshold
        self.sample_rate = sample_rate

        self.model, self.mel_extractor = self._load_model()

    def _load_model(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            checkpoint_config = json.loads(metadata.get("config", "{}"))
            config = checkpoint_config.get("model", {})
            state_dict = safe_load_file(str(checkpoint_path), device=self.device)
        else:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            config = checkpoint.get("config", {}).get("model", {})
            state_dict = checkpoint["model_state_dict"]

        model = create_audio_quality_model(config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        mel_extractor = LogMelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=config.get("n_mels", 80),
        ).to(self.device)
        mel_extractor.eval()

        return model, mel_extractor

    def _load_audio(self, audio_path: str | os.PathLike[str]) -> torch.Tensor:
        audio_np, sr = sf.read(str(audio_path), always_2d=True)
        audio_np = audio_np.mean(axis=1)

        if sr != self.sample_rate:
            gcd = math.gcd(sr, self.sample_rate)
            audio_np = resample_poly(
                audio_np,
                self.sample_rate // gcd,
                sr // gcd,
            )

        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)

        return waveform.to(self.device)

    @torch.no_grad()
    def infer(self, audio_path: str | os.PathLike[str]) -> dict[str, Any]:
        waveform = self._load_audio(audio_path)
        mel = self.mel_extractor(waveform)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)

        logits = self.model(mel, mask=None)
        probs = torch.softmax(logits, dim=-1)
        degraded_prob = float(probs[0, 1].item())
        is_degraded = degraded_prob >= self.threshold

        return {
            "is_degraded": is_degraded,
            "degraded_prob": degraded_prob,
            "label": int(is_degraded),
        }

    def predict(self, audio_path: str | os.PathLike[str]) -> tuple[bool, float]:
        result = self.infer(audio_path)
        return result["is_degraded"], result["degraded_prob"]


def get_router(*args: Any, **kwargs: Any) -> AudioQualityRouter:
    return AudioQualityRouter(*args, **kwargs)
