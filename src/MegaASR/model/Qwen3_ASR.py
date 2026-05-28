from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class Qwen3ASR:
    NAME = "Qwen3-ASR-1.7B"
    HF_REPO_ID = "Qwen/Qwen3-ASR-1.7B"
    DEFAULT_MODEL_DIR = "ckpt/Mega-ASR/Qwen3-ASR-1.7B"

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        repo_id: str | None = None,
        device_map: str | None = None,
        dtype: Any | None = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 2048,
        download_kwargs: dict[str, Any] | None = None,
        **model_kwargs: Any,
    ) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        repo_id = repo_id or self.HF_REPO_ID
        self.model_path = str(Path(model_path or self.DEFAULT_MODEL_DIR).expanduser())
        if not self._has_local_model(self.model_path):
            self.model_path = self.download_model(
                self.model_path,
                repo_id=repo_id,
                **(download_kwargs or {}),
            )

        if device_map is None:
            if torch.cuda.is_available():
                device_map = "cuda:0"
            elif torch.backends.mps.is_available():
                device_map = "mps"
            else:
                device_map = "cpu"
        if dtype is None:
            if device_map == "cpu":
                dtype = torch.float32
            elif device_map == "mps":
                dtype = torch.float16
            else:
                dtype = torch.bfloat16

        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )

    @staticmethod
    def _has_local_model(model_path: str | os.PathLike[str]) -> bool:
        path = Path(model_path).expanduser()
        return path.is_dir() and (path / "config.json").is_file()

    @staticmethod
    def download_model(
        model_path: str | os.PathLike[str],
        *,
        repo_id: str,
        **snapshot_kwargs: Any,
    ) -> str:
        from huggingface_hub import snapshot_download

        local_dir = Path(model_path).expanduser()
        local_dir.mkdir(parents=True, exist_ok=True)

        return snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            **snapshot_kwargs,
        )

    def infer(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        **transcribe_kwargs: Any,
    ) -> str | list[str] | Any:
        if isinstance(audio, os.PathLike):
            audio = str(audio)
        elif isinstance(audio, (list, tuple)):
            audio = [str(item) if isinstance(item, os.PathLike) else item for item in audio]

        results = self.model.transcribe(
            audio=audio,
            language=language,
            **transcribe_kwargs,
        )

        if return_objects:
            return results

        if isinstance(results, list):
            return [str(getattr(result, "text", result)).strip() for result in results]

        return str(getattr(results, "text", results)).strip()


def get_mega_asr(*args: Any, **kwargs: Any) -> Qwen3ASR:
    return Qwen3ASR(*args, **kwargs)
