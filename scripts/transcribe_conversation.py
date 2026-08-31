# coding=utf-8
"""
Qwen3-ASR conversation transcription script (Transformers backend).

Single-file mode:
  python scripts/transcribe_conversation.py -i <audio_file> [OPTIONS]

Batch mode (model loaded once):
  python scripts/transcribe_conversation.py --input-dir <dir> [OPTIONS]
  python scripts/transcribe_conversation.py -i file1.wav file2.wav [OPTIONS]

Processes each channel independently:
  1. VAD splits each channel into utterance segments.
  2. Each segment is transcribed separately (fast, no forced aligner required).
  3. All segments are merged and sorted by start time into a conversation JSON.

Options:
  --model-path/-mp        ASR model path (default: ./ckpt/Mega-ASR/Qwen3-ASR-1.7B)
  --lora-dir/-ld          LoRA adapter directory (optional)
  --keep-delta-on-gpu     Keep LoRA deltas on GPU (default: True)
  --input/-i              Audio file path(s); multiple paths enable batch mode
  --input-dir/-id         Directory with audio files to batch-process
  --extensions            File extensions to scan in --input-dir (default: wav,mp3,flac,m4a)
  --skip-existing         Skip files whose output JSON already exists (default: True)
  --no-skip-existing      Re-process files even if output JSON exists
  --output/-o             JSON output path (single file) or output directory (batch mode)
  --language/-l           Force language, e.g. "Chinese", "English"
  --device/-d             Inference device (default: cuda:0)
  --dtype                 Model dtype: bfloat16 / float16 / float32 (default: bfloat16)
  --silence-gap/-sg       Min silence duration (s) between words (default: 0.5)
  --silence-thresh/-st    RMS energy threshold for silence (default: 0.01)
  --min-speech/-ms        Min speech segment duration (s) (default: 0.2)
  --channels/-c           Number of channels to process (default: 2)
  --vad                   VAD backend: simple-vad / silero-vad / ten-vad / fsmn-vad
  --vad_model_path        Path to VAD model
  --max-new-tokens        Max new tokens per segment (default: 256)

Output format (per file):
  {
    "source": "...",
    "filename": "...",
    "channels": 2,
    "audio_dur_s": 120.0,
    "transcribe_s": 20.0,
    "rtf": 0.167,
    "rtfx": 6.0,
    "vad_s": 0.5,
    "vad_rtf": 0.004,
    "vad_rtfx": 240.0,
    "model_name": "Qwen3-ASR-1.7B",
    "lora_dir": null,
    "vad_model": "silero-vad",
    "aligner_model": "no_aligner",
    "conversations": [
      {"speaker": "channel_0", "text": "...", "start": 0.0, "end": 1.2},
      ...
    ]
  }
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch

from qwen_asr import Qwen3ASRModel
from vad_utils import apply_vad, init_vad

# LoRADeltaSwitch lives in src/MegaASR
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR conversation transcription tool (single file or batch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", "-mp", default="./ckpt/Mega-ASR/Qwen3-ASR-1.7B", help="ASR model path")
    parser.add_argument("--lora-dir", "-ld", default=None, dest="lora_dir",
                        help="LoRA adapter directory (e.g. ckpt/Mega-ASR/mega-asr-merged)")
    parser.add_argument("--keep-delta-on-gpu", type=lambda x: x.lower() in ("1", "true", "yes", "y"),
                        default=True, dest="keep_delta_on_gpu",
                        help="Keep LoRA deltas on GPU (use False on MPS to save memory)")
    parser.add_argument("--input", "-i", nargs="*", default=None,
                        help="Audio file path(s); pass multiple paths to enable batch mode")
    parser.add_argument("--input-dir", "-id", default=None, dest="input_dir",
                        help="Directory with audio files to batch-process (model loaded once)")
    parser.add_argument("--extensions", default="wav,mp3,flac,m4a",
                        help="File extensions to scan in --input-dir (comma-separated)")
    parser.add_argument("--skip-existing", action="store_true", default=True, dest="skip_existing",
                        help="Skip files whose output JSON already exists")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-process files even if output JSON exists")
    parser.add_argument("--output", "-o", default=None,
                        help="JSON output path (single file) or output directory (batch mode); "
                             "default: results/<basename>.conversation.<model>.<vad>.no-aligner.json")
    parser.add_argument("--language", "-l", default=None, help='Force language, e.g. "Chinese", "English"')
    parser.add_argument("--device", "-d", default="cuda:0", help='Inference device, e.g. "mps", "cuda:0", "cpu"')
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE_MAP.keys()), help="Model dtype")
    parser.add_argument("--silence-gap", "-sg", type=float, default=0.5, dest="silence_gap",
                        help="Min silence duration (s) between words to split utterances")
    parser.add_argument("--silence-thresh", "-st", type=float, default=0.01, dest="silence_thresh",
                        help="RMS energy threshold below which a frame is considered silent")
    parser.add_argument("--min-speech", "-ms", type=float, default=0.2, dest="min_speech",
                        help="Min speech segment duration (s); shorter segments are discarded")
    parser.add_argument("--channels", "-c", type=int, default=-1,
                        help="Channels to process: 0=ch0 only, 1=ch1 only, -1 or >=num_channels=all (default: -1)")
    parser.add_argument("--vad", default="simple",
                        choices=["simple", "simple-vad", "silero", "silero-vad", "ten", "ten-vad", "fsmn", "fsmn-vad"],
                        help="VAD backend")
    parser.add_argument("--vad_model_path", default=None, help="Path to VAD model")
    parser.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens",
                        help="Max new tokens for generation per segment")


    args = parser.parse_args()

    if "fsmn" in args.vad:
        args.vad = "fsmn-vad"
    elif "ten" in args.vad:
        args.vad = "ten-vad"
    elif "silero" in args.vad:
        args.vad = "silero-vad"
    elif "simple" in args.vad:
        args.vad = "simple-vad"

    if not args.input and not args.input_dir:
        parser.error("at least one of --input / --input-dir is required")

    return args


def collect_input_files(args) -> list:
    """Collect all input file paths from --input and/or --input-dir."""
    files = []

    if args.input:
        for p in args.input:
            if os.path.isfile(p):
                files.append(p)
            else:
                logger.warning("[warning] not a file, skipping: %s", p)

    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            raise ValueError(f"--input-dir is not a directory: {args.input_dir!r}")
        exts = {f".{e.strip().lstrip('.')}" for e in args.extensions.split(",")}
        for fname in sorted(os.listdir(args.input_dir)):
            fpath = os.path.join(args.input_dir, fname)
            if os.path.isfile(fpath) and Path(fpath).suffix.lower() in exts:
                files.append(fpath)

    return files


def resolve_channels(channels_arg: int, num_channels: int) -> list:
    """-1 or value >= num_channels → all channels; otherwise [channels_arg]."""
    if channels_arg < 0 or channels_arg >= num_channels:
        return list(range(num_channels))
    return [channels_arg]


def get_output_path(audio_path: str, model_name: str, vad_name: str, output_arg, batch_mode: bool) -> str:
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    default_name = f"{basename}.conversation.{model_name}.{vad_name}.no-aligner.json"
    if output_arg is None:
        return os.path.join("results", default_name)
    elif batch_mode:
        return os.path.join(output_arg, default_name)
    else:
        return output_arg


def transcribe_one(
    asr,
    vad_instance,
    audio_path: str,
    args,
    model_name: str,
    output_path: str,
) -> None:
    """Transcribe a single audio file and save JSON results."""
    audio_data, sample_rate = sf.read(audio_path, always_2d=True)
    total_dur_s = audio_data.shape[0] / sample_rate
    num_channels = audio_data.shape[1]
    channels_to_process = resolve_channels(args.channels, num_channels)
    logger.info("[input]  %s  (%d ch, %.1fs)  processing ch%s",
                audio_path, num_channels, total_dur_s,
                "+".join(str(c) for c in channels_to_process))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_utterances = []
    total_vad_s = 0.0
    t_transcribe = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        for ch in channels_to_process:
            channel_audio = audio_data[:, ch]

            tmp_ch_wav = os.path.join(tmpdir, f"ch{ch}_full.wav")
            sf.write(tmp_ch_wav, channel_audio, sample_rate)

            t_vad_ch = time.perf_counter()
            segments = apply_vad(
                tmp_ch_wav,
                vad_type=args.vad,
                vad_instance=vad_instance,
                silence_gap_s=args.silence_gap,
                silence_thresh=args.silence_thresh,
                min_speech_s=args.min_speech,
            )
            total_vad_s += time.perf_counter() - t_vad_ch
            logger.info("[channel %d] %d segment(s) detected by %s VAD", ch, len(segments), args.vad)

            if not segments:
                logger.info("[channel %d] no speech detected, skipping", ch)
                continue

            for seg_idx, seg in enumerate(segments):
                start_sample = int(seg["start"] * sample_rate)
                end_sample   = int(seg["end"]   * sample_rate)
                seg_audio    = channel_audio[start_sample:end_sample]

                tmp_wav = os.path.join(tmpdir, f"ch{ch}_seg{seg_idx}.wav")
                sf.write(tmp_wav, seg_audio, sample_rate)

                t1 = time.perf_counter()
                results = asr.transcribe(
                    audio=tmp_wav,
                    language=args.language,
                    return_time_stamps=False,
                )
                elapsed = time.perf_counter() - t1

                text = results[0].text.strip()
                logger.info("[channel %d] seg %03d [%.2fs-%.2fs] (%.2fs) -> %r",
                            ch, seg_idx, seg["start"], seg["end"], elapsed, text)

                if text:
                    all_utterances.append({
                        "speaker": f"channel_{ch}",
                        "text":  text,
                        "start": seg["start"],
                        "end":   seg["end"],
                    })

    all_utterances.sort(key=lambda u: (u["start"], u["speaker"]))

    transcribe_s = time.perf_counter() - t_transcribe
    rtf = transcribe_s / total_dur_s if total_dur_s > 0 else None
    vad_rtf = total_vad_s / total_dur_s if total_dur_s > 0 else None
    logger.info("[timing] transcribe total: %.3fs", transcribe_s)
    logger.info("[timing] vad total:        %.3fs", total_vad_s)
    if rtf is not None:
        logger.info("[RTF]    RTF=%.4f  RTFx=%.2f", rtf, 1 / rtf)
    if vad_rtf is not None:
        logger.info("[VAD RTF] RTF=%.4f  RTFx=%.2f", vad_rtf, 1 / vad_rtf)

    output = {
        "source":        audio_path,
        "filename":      os.path.basename(audio_path),
        "channels":      channels_to_process,
        "audio_dur_s":   round(total_dur_s, 3),
        "transcribe_s":  round(transcribe_s, 3),
        "rtf":           round(rtf, 4) if rtf is not None else None,
        "rtfx":          round(1 / rtf, 2) if rtf else None,
        "vad_s":         round(total_vad_s, 3),
        "vad_rtf":       round(vad_rtf, 4) if vad_rtf is not None else None,
        "vad_rtfx":      round(1 / vad_rtf, 2) if vad_rtf else None,
        "model_name":    model_name,
        "lora_dir":      args.lora_dir,
        "vad_model":     args.vad,
        "aligner_model": "no_aligner",
        "conversations": all_utterances,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("[result] %d utterance(s) in conversation", len(all_utterances))
    for u in all_utterances[:8]:
        logger.info("  [%.2f-%.2f] %s: %r", u["start"], u["end"], u["speaker"], u["text"])
    if len(all_utterances) > 8:
        logger.info("  ... (%d more)", len(all_utterances) - 8)
    logger.info("[output] saved: %s", output_path)


def main() -> None:
    args = parse_args()

    input_files = collect_input_files(args)
    if not input_files:
        logger.error("No input files found. Use --input or --input-dir.")
        sys.exit(1)

    batch_mode = len(input_files) > 1 or args.input_dir is not None
    model_name = os.path.basename(os.path.normpath(args.model_path))

    dtype = _DTYPE_MAP[args.dtype]
    logger.info("[config] model=%s", args.model_path)
    logger.info("[config] device=%s  dtype=%s  vad=%s", args.device, args.dtype, args.vad)
    if args.vad == "simple-vad":
        logger.info("[config] silence_gap=%.2fs  silence_thresh=%.4f", args.silence_gap, args.silence_thresh)
    logger.info("[config] min_speech=%.2fs  channels=%s", args.min_speech, args.channels)
    if batch_mode:
        logger.info("[batch] %d file(s) to process", len(input_files))

    t0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map=args.device,
        max_inference_batch_size=32,
        max_new_tokens=args.max_new_tokens,
    )
    logger.info("[timing] model loaded: %.3fs", time.perf_counter() - t0)

    if args.lora_dir:
        from MegaASR.model.utils.lora_switch import LoRADeltaSwitch
        logger.info("[lora] loading adapter from %s", args.lora_dir)
        t_lora = time.perf_counter()
        keep_on_gpu = args.keep_delta_on_gpu
        if args.device == "mps" and keep_on_gpu:
            keep_on_gpu = False
            logger.info("[lora] MPS detected: keeping LoRA deltas on CPU to save GPU memory")
        lora_switch = LoRADeltaSwitch(keep_delta_on_gpu=keep_on_gpu)
        lora_switch.add_adapter(
            parent_module=asr.model,
            adapter_dir=args.lora_dir,
            name="mega_asr_adapter",
        )
        lora_switch.set_active(True)
        logger.info("[lora] adapter loaded and applied: %.3fs", time.perf_counter() - t_lora)

    logger.info("[vad] initializing %s VAD...", args.vad)
    t_vad = time.perf_counter()
    vad_instance = init_vad(args.vad, args.vad_model_path)
    logger.info("[vad] initialized in %.3fs", time.perf_counter() - t_vad)

    stats = {"done": 0, "skip": 0, "fail": 0}

    for file_idx, audio_path in enumerate(input_files, 1):
        output_path = get_output_path(audio_path, model_name, args.vad, args.output, batch_mode)

        if batch_mode:
            logger.info("[batch %d/%d] %s", file_idx, len(input_files), audio_path)

        if args.skip_existing and os.path.isfile(output_path):
            logger.info("[skip] output exists: %s", output_path)
            stats["skip"] += 1
            continue

        try:
            transcribe_one(asr, vad_instance, audio_path, args, model_name, output_path)
            stats["done"] += 1
        except Exception as e:
            logger.error("[FAILED] %s: %s", audio_path, e, exc_info=True)
            stats["fail"] += 1

    if batch_mode:
        logger.info(
            "[summary] done=%d  skip=%d  fail=%d  total=%d",
            stats["done"], stats["skip"], stats["fail"],
            stats["done"] + stats["skip"] + stats["fail"],
        )


if __name__ == "__main__":
    main()
