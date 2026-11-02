# coding=utf-8
"""Overlay the per-sample WER CDFs of several eval runs on one figure.

Each input is a per-sample JSONL as written by eval_checkpoint.py --out_json
(one object per line, with a "wer" field; NaN/empty-ref rows are dropped):

    {"index": 42, "audio_path": "...", "ref": "...", "hyp": "...", "wer": 0.12}

An empirical CDF answers "what fraction of utterances have WER <= x". A curve
that sits UP-AND-TO-THE-LEFT of another is strictly better: more of its mass is
at low WER. Overlaying the fusion variants on shared axes makes that
domination (or the lack of it, curves crossing) directly readable.

Two ways to name the series:

  # explicit, controls order + label text
  python tests/plot_wer_cdf_overlay.py \
      --series "Baseline=/home/ASR/daniel/eval/baseline/persample.jsonl" \
      --series "No fusion=/home/ASR/daniel/eval/no_fusion/persample.jsonl" \
      --series "Early fusion=/home/ASR/daniel/eval/early_fusion/persample.jsonl" \
      --series "Late fusion=/home/ASR/daniel/eval/late_fusion/persample.jsonl" \
      --series "FDDT fusion=/home/ASR/daniel/eval/fddt_fusion/persample.jsonl" \
      --out_png /home/ASR/daniel/eval/wer_cdf_overlay.png --threshold 0.3

  # or auto-discover <eval_root>/<name>/persample.jsonl, subdir name -> label
  python tests/plot_wer_cdf_overlay.py \
      --eval_root /home/ASR/daniel/eval \
      --out_png /home/ASR/daniel/eval/wer_cdf_overlay.png --threshold 0.3

WER is unbounded above -- a hypothesis longer than its reference (repetition
loop, hallucinated continuation) scores well past 1.0 -- so the right edge is
a real choice, not a formality. --xmax controls it:

    --xmax auto     fit every sample (default)
    --xmax p99.9    crop to a percentile, so one runaway sample cannot squash
                    the 0-1 region every other curve lives in
    --xmax 3        an explicit number

auto/pNN never crop below 1.0, and the legend reports how many samples spill
past whatever edge is chosen.
"""
import argparse
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np


# Canonical color+style per known variant, so a given fusion type keeps the
# SAME color across every chart regardless of the order it was passed in
# (color follows the entity, not its rank). Colors are the validated
# categorical palette (colorblind-safe in fixed slot order); each also gets a
# distinct line style so 5 overlapping curves stay separable under CVD / when
# printed grayscale, not by hue alone. Unknown labels fall through to _FALLBACK.
_STYLES = {
    "baseline":     dict(color="#898781", linestyle=(0, (1, 1)), zorder=2),  # muted, dotted: the reference
    "no fusion":    dict(color="#2a78d6", linestyle="-",  zorder=3),
    "early fusion": dict(color="#eb6834", linestyle="--", zorder=4),
    "late fusion":  dict(color="#1baf7a", linestyle="-.", zorder=4),
    "fddt fusion":  dict(color="#e87ba4", linestyle=(0, (3, 1, 1, 1)), zorder=4),
}
_FALLBACK = [
    dict(color="#eda100", linestyle="-"),
    dict(color="#4a3aa7", linestyle="--"),
    dict(color="#e34948", linestyle="-."),
    dict(color="#008300", linestyle=(0, (3, 1, 1, 1))),
]


def _norm_label(label):
    """Normalize a label to match _STYLES keys: lowercase, collapse runs of
    non-alphanumerics to single spaces, strip. So "No_fusion", "no-fusion",
    "No Fusion" all map to the same canonical style."""
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _load_wers(path):
    """Return the finite WER values from a per-sample JSONL, plus each row's
    ref word count (for corpus_wer below). Rows with a missing/None/NaN wer
    (e.g. empty-reference utterances jiwer can't score) are dropped -- they
    carry no CDF information."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            wer = rec.get("wer")
            if wer is None:
                continue
            wer = float(wer)
            if math.isnan(wer):
                continue
            ref_word_count = len(rec.get("ref", "").split())
            rows.append((wer, ref_word_count))
    rows.sort(key=lambda r: r[0])
    wers = np.array([r[0] for r in rows], dtype=float)
    ref_word_counts = np.array([r[1] for r in rows], dtype=float)
    return wers, ref_word_counts


def _corpus_wer(wers, ref_word_counts):
    """Edit-count-weighted WER: sum(wer_i * ref_word_count_i) /
    sum(ref_word_count_i), reconstructing jiwer's corpus-level WER
    (jiwer.wer(all_refs, all_hyps)) from per-sample wer + ref length --
    NOT a naive mean of per-sample wer, which overweights short utterances
    (a 1-word reference with one error contributes WER=1.0 with equal
    weight to a 40-word utterance) and disagrees with the corpus WER
    eval_checkpoint.py itself reports."""
    total_words = ref_word_counts.sum()
    if total_words <= 0:
        return float("nan")
    return float((wers * ref_word_counts).sum() / total_words)


def _cdf_xy(wers, xmax):
    """Step-function CDF coordinates. Start at (0, 0) so every curve shares the
    origin, and extend the last level flat to xmax so curves that top out early
    (all mass below xmax) still read as reaching their final height across the
    full axis.

    Points beyond xmax are dropped rather than plotted-then-clipped: keeping
    them makes the polyline run out to the last sample and then jump back left
    to xmax, painting a spurious horizontal segment across the top of the
    visible area."""
    n = len(wers)
    y = np.arange(1, n + 1) / n
    keep = wers <= xmax          # wers is sorted, so this is a prefix
    x = np.concatenate([[0.0], wers[keep]])
    y = np.concatenate([[0.0], y[keep]])
    if x[-1] < xmax:
        x = np.concatenate([x, [xmax]])
        y = np.concatenate([y, [y[-1]]])
    return x, y


def _parse_xmax(spec):
    """Parse --xmax into (mode, value): ("fixed", float) for an explicit edge,
    ("quantile", q) for "auto"/"max" (q=100) or "pNN.N"."""
    text = str(spec).strip().lower()
    if text in ("auto", "max"):
        return ("quantile", 100.0)
    if text.startswith("p"):
        try:
            q = float(text[1:])
        except ValueError:
            raise SystemExit(f"--xmax: bad percentile {spec!r} (want e.g. p99.9)")
        if not 0.0 < q <= 100.0:
            raise SystemExit(f"--xmax: percentile must be in (0, 100], got {q}")
        return ("quantile", q)
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"--xmax: want a number, 'auto', or 'pNN', got {spec!r}")
    if value <= 0.0:
        raise SystemExit(f"--xmax: must be > 0, got {value}")
    return ("fixed", value)


def _round_up_nice(value):
    """Round up to a quarter-decade tick (0.25/2.5/25/...) so an auto-fitted
    axis lands on a readable number instead of 2.7481."""
    if value <= 0.0:
        return 1.0
    step = 10.0 ** math.floor(math.log10(value)) / 4.0
    return math.ceil(value / step) * step


def _collect_series(args):
    """Resolve --series / --eval_root into an ordered [(label, path)] list."""
    series = []
    for spec in args.series:
        if "=" not in spec:
            raise SystemExit(f"--series must be LABEL=PATH, got: {spec!r}")
        label, path = spec.split("=", 1)
        series.append((label.strip(), path.strip()))
    if args.eval_root:
        found = []
        for path in glob.glob(os.path.join(args.eval_root, "*", "persample.jsonl")):
            label = os.path.basename(os.path.dirname(path)).replace("_", " ")
            found.append((label, path))
        # Order known variants by their canonical progression (baseline -> no
        # fusion -> early -> late -> fddt) rather than glob's alphabetical order;
        # unknown labels sort after, alphabetically. Explicit --series keep their
        # given order and come first.
        rank = {name: i for i, name in enumerate(_STYLES)}
        found.sort(key=lambda lp: (rank.get(_norm_label(lp[0]), len(rank)), lp[0].lower()))
        series.extend(found)
    if not series:
        raise SystemExit("no series: pass --series LABEL=PATH ... or --eval_root DIR")
    return series


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", default=[], metavar="LABEL=PATH",
                    help="a per-sample JSONL and its legend label; repeatable. "
                         "Order given is the legend/color order.")
    ap.add_argument("--eval_root", default=None,
                    help="auto-add every <eval_root>/*/persample.jsonl, using the "
                         "subdirectory name (underscores -> spaces) as the label.")
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="draw a vertical reference line at this WER (e.g. 0.3, "
                         "the flagging threshold) and report %% of samples <= it.")
    ap.add_argument("--xmax", default="auto", metavar="AUTO|pNN|FLOAT",
                    help="right edge of the WER axis (default: auto). WER can exceed "
                         "1.0, so a fixed edge hides the tail: 'auto' fits every "
                         "sample, 'pNN.N' (e.g. p99.9) crops to a percentile of the "
                         "pooled samples, or give an explicit number. auto/pNN never "
                         "crop below 1.0; the legend notes how many samples spill "
                         "past the chosen edge.")
    ap.add_argument("--title", default="WER CDF by fusion type")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    series = _collect_series(args)
    xmax_mode, xmax_value = _parse_xmax(args.xmax)

    # Pass 1: load everything. An auto/percentile edge is a function of the
    # pooled samples, so no curve can be placed until all of them are in hand.
    loaded = []
    fallback_i = 0
    for label, path in series:
        wers, ref_word_counts = _load_wers(path)
        if len(wers) == 0:
            print(f"[warn] {label}: no scorable samples in {path} -- skipping")
            continue

        style = _STYLES.get(_norm_label(label))
        if style is None:
            style = _FALLBACK[fallback_i % len(_FALLBACK)]
            fallback_i += 1

        loaded.append((label, wers, ref_word_counts, style))

    if not loaded:
        raise SystemExit("no scorable samples in any series")

    if xmax_mode == "fixed":
        xmax = xmax_value
    else:
        pooled = np.concatenate([w for _, w, _, _ in loaded])
        raw = (float(pooled.max()) if xmax_value >= 100.0
               else float(np.percentile(pooled, xmax_value)))
        # Never crop below 1.0 (the reference-length boundary) or below the
        # threshold line, whichever is further right.
        xmax = _round_up_nice(max(raw, args.threshold or 0.0, 1.0))
    print(f"[plot_wer_cdf_overlay] x-axis 0 -> {xmax:g} (--xmax {args.xmax})")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Pass 2: draw.
    for label, wers, ref_word_counts, style in loaded:
        x, y = _cdf_xy(wers, xmax)

        n = len(wers)
        median = float(np.median(wers))
        corpus = _corpus_wer(wers, ref_word_counts)
        over = int(np.sum(wers > xmax))
        parts = [f"med {median:.0%}", f"corpus {corpus:.0%}"]
        if args.threshold is not None:
            frac = float(np.mean(wers <= args.threshold))
            parts.append(f"≤{args.threshold:.0%}: {frac:.0%}")
        parts.append(f"n={n}")
        if over:
            parts.append(f"{over}>{xmax:g}")
        legend_label = f"{label}  ({', '.join(parts)})"

        ax.plot(x, y, linewidth=2.0, label=legend_label,
                solid_capstyle="round", **style)

    if args.threshold is not None:
        ax.axvline(args.threshold, color="#c3c2b7", linewidth=1.0,
                   linestyle=":", zorder=1)
        ax.text(args.threshold, 0.01, f" WER={args.threshold:.0%}",
                color="#898781", fontsize=8, va="bottom", ha="left")

    # WER=1.0 is the point where a hypothesis carries as many errors as the
    # reference has words; past it the output is net-insertion (hallucination,
    # repetition). Worth marking once the axis actually reaches out there.
    if xmax > 1.05 and args.threshold != 1.0:
        ax.axvline(1.0, color="#c3c2b7", linewidth=1.0, linestyle="--", zorder=1)
        ax.text(1.0, 0.01, " WER=100%", color="#898781", fontsize=8,
                va="bottom", ha="left")

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Word error rate (WER)")
    ax.set_ylabel("Fraction of utterances with WER ≤ x")
    ax.set_title(args.title)
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    # up-and-to-the-left = better; legend at lower-right stays clear of the curves
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, title="better = up/left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    fig.savefig(args.out_png, dpi=args.dpi)
    print(f"[plot_wer_cdf_overlay] wrote {args.out_png}  ({len(series)} series)")


if __name__ == "__main__":
    main()
