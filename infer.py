import sys
sys.path.append("src")

import argparse
import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO = ROOT_DIR / "assets/example/F01_22GC010K_STR.wav"
DEFAULT_CKPT_DIR = ROOT_DIR / "ckpt/Mega-ASR"
DEFAULT_ROUTING = True
DEFAULT_THRESHOLD = 0.5


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("1", "true", "yes", "y")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(description="Mega-ASR inference")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="audio file path")
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR, help="Mega-ASR ckpt root")
    parser.add_argument("--routing", type=str2bool, default=DEFAULT_ROUTING, help="enable router")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="router threshold")
    parser.add_argument("--device_map", default=None, help="transformers device_map, e.g. cuda:0, mps, cpu")
    parser.add_argument("--mps", action="store_true", help="use Apple MPS backend (shorthand for --device_map mps)")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--keep_delta_on_gpu", type=str2bool, default=True, help="keep LoRA deltas on GPU")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device_map = args.device_map
    if args.mps and device_map is None:
        device_map = "mps"

    # MPS 模式下 LoRA delta 默认放 CPU，节省 GPU 内存
    # 用户若显式传了 --keep_delta_on_gpu true 则尊重用户设置
    keep_delta_on_gpu = args.keep_delta_on_gpu
    if device_map == "mps" and keep_delta_on_gpu:
        import sys
        user_set_keep_delta = any(a.startswith("--keep_delta_on_gpu") for a in sys.argv[1:])
        if not user_set_keep_delta:
            keep_delta_on_gpu = False

    from MegaASR.model.megaASR import MegaASR
    import soundfile as sf

    audio = resolve_path(args.audio)
    ckpt_dir = resolve_path(args.ckpt_dir)

    audio_info = sf.info(str(audio))
    audio_duration = audio_info.duration

    t0 = time.perf_counter()
    model = MegaASR(
        model_path=ckpt_dir / "Qwen3-ASR-1.7B",
        lora_dir=ckpt_dir / "mega-asr-merged",
        router_checkpoint=ckpt_dir / "audio_quality_router/best_acc_model.safetensors",
        routing_enabled=args.routing,
        quality_threshold=args.threshold,
        device_map=device_map,
        keep_delta_on_gpu=keep_delta_on_gpu,
        backend="transformers",
    )
    t1 = time.perf_counter()
    result = model.infer(audio, return_route=True)
    t2 = time.perf_counter()

    load_time = t1 - t0
    infer_time = t2 - t1
    rtf = infer_time / audio_duration
    rtfx = 1.0 / rtf

    print(result)
    print(f"[timing] load={load_time:.2f}s  infer={infer_time:.2f}s  total={t2-t0:.2f}s")
    print(f"[rtf]    audio={audio_duration:.2f}s  RTF={rtf:.4f}  RTFx={rtfx:.2f}x")

if __name__ == "__main__":
    main()
