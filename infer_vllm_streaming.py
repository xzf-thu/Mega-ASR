import sys
sys.path.append("src")

import argparse
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from infer_vllm import build_vllm_kwargs, materialized_lora_dir, resolve_path, str2bool


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO = ROOT_DIR / "assets/example/F01_22GC010K_STR.wav"
DEFAULT_CKPT_DIR = ROOT_DIR / "ckpt/Mega-ASR"


def parse_args():
    parser = argparse.ArgumentParser(description="Mega-ASR vLLM streaming inference")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="audio file path")
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="Mega-ASR ckpt root")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--step_ms", type=int, default=1000, help="streaming input step in milliseconds")
    parser.add_argument("--chunk_size_sec", type=float, default=2.0, help="Qwen3-ASR streaming chunk size")
    parser.add_argument("--unfixed_chunk_num", type=int, default=2)
    parser.add_argument("--unfixed_token_num", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=32, help="small value is recommended for streaming")
    parser.add_argument(
        "--vllm_materialize_lora_force",
        type=str2bool,
        default=False,
        help="rebuild the materialized LoRA checkpoint even if the cache is fresh",
    )
    parser.add_argument(
        "--vllm_materialize_lora_device_map",
        default=None,
        help="device_map used only while materializing LoRA, e.g. cpu or cuda:0",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    return parser.parse_args()


def read_audio_16k(path: str | os.PathLike[str]) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr == 16000:
        return wav.astype(np.float32, copy=False)

    duration = wav.shape[0] / float(sr)
    target_len = int(round(duration * 16000))
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from MegaASR.model.megaASR import MegaASR

    audio = resolve_path(args.audio)
    ckpt_dir = resolve_path(args.ckpt_dir)
    vllm_kwargs = build_vllm_kwargs(args)

    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        routing_enabled=False,
        backend="vllm",
        vllm_apply_lora_on_load=True,
        vllm_materialized_lora_dir=materialized_lora_dir(ckpt_dir),
        vllm_materialize_lora_force=args.vllm_materialize_lora_force,
        vllm_materialize_lora_device_map=args.vllm_materialize_lora_device_map,
        max_new_tokens=args.max_new_tokens,
        **vllm_kwargs,
    )

    wav16k = read_audio_16k(audio)
    audio_duration = wav16k.shape[0] / 16000.0
    step = int(round(args.step_ms / 1000.0 * 16000))
    if step <= 0:
        raise ValueError("--step_ms must be positive.")

    state = model.init_streaming_state(
        unfixed_chunk_num=args.unfixed_chunk_num,
        unfixed_token_num=args.unfixed_token_num,
        chunk_size_sec=args.chunk_size_sec,
    )

    pos = 0
    call_id = 0
    total_infer_time = 0.0
    while pos < wav16k.shape[0]:
        seg = wav16k[pos:pos + step]
        pos += seg.shape[0]
        call_id += 1
        chunk_duration = seg.shape[0] / 16000.0
        t0 = time.perf_counter()
        model.streaming_transcribe(seg, state)
        chunk_infer = time.perf_counter() - t0
        total_infer_time += chunk_infer
        chunk_rtf = chunk_infer / chunk_duration if chunk_duration > 0 else 0.0
        print(f"[call {call_id:03d}] language={state.language!r} text={state.text!r}"
              f"  chunk_RTF={chunk_rtf:.3f}")

    t0 = time.perf_counter()
    model.finish_streaming_transcribe(state)
    total_infer_time += time.perf_counter() - t0

    rtf = total_infer_time / audio_duration if audio_duration > 0 else 0.0
    rtfx = 1.0 / rtf if rtf > 0 else 0.0
    print(f"[final] language={state.language!r} text={state.text!r}")
    print(f"[rtf]   audio={audio_duration:.2f}s  infer={total_infer_time:.2f}s"
          f"  RTF={rtf:.4f}  RTFx={rtfx:.2f}x")


if __name__ == "__main__":
    main()
