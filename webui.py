"""
Mega-ASR Console — Streamlit 前端（接入真实后端）

依赖（硬性）:
    pip install streamlit numpy soundfile streamlit-mic-recorder
    可选: psutil（系统监控）、pynvml（NVIDIA GPU 监控）
    后端: 项目内的 MegaASR.model.megaASR.MegaASR

启动:
    streamlit run webui.py

可选环境变量（覆盖默认配置）:
    MEGA_ASR_CKPT_DIR     默认 ./ckpt/Mega-ASR
    MEGA_ASR_ROUTING      "1"/"true" 开启路由（默认开启）
    MEGA_ASR_THRESHOLD    路由阈值（默认 0.5）
    MEGA_ASR_DEVICE_MAP   传给 transformers 的 device_map，例如 "auto" / "cuda:0"
"""

import base64
import io
import os
import platform
import sys
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from streamlit_mic_recorder import mic_recorder

# ──────────────────────────────────────────────
# 路径：把 src/ 加进 sys.path，与 infer.py 行为一致
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ──────────────────────────────────────────────
# 后端配置（可被环境变量覆盖）
# ──────────────────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")


CKPT_DIR = Path(os.environ.get("MEGA_ASR_CKPT_DIR", str(ROOT_DIR / "ckpt/Mega-ASR")))
ROUTING_ENABLED = _env_bool("MEGA_ASR_ROUTING", True)
ROUTING_THRESHOLD = float(os.environ.get("MEGA_ASR_THRESHOLD", "0.5"))
DEVICE_MAP = os.environ.get("MEGA_ASR_DEVICE_MAP") or None  # None 表示不传

# ──────────────────────────────────────────────
# 可选依赖
# ──────────────────────────────────────────────
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

OS_NAME = platform.system()

# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mega-ASR Console",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# i18n
# ──────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "brand_sub": "Speech Recognition · Realtime",
        "sec_system": "System Monitor",
        "sec_console": "Console",
        "sec_log": "Transcripts",
        "m_cpu": "CPU Usage",
        "m_mem": "Memory",
        "m_gpu": "GPU Usage",
        "m_vram": "VRAM",
        "m_disk": "Disk",
        "m_net": "Network I/O",
        "status_rec": "Recording",
        "status_proc": "Processing",
        "status_pending": "Awaiting recognition",
        "status_idle": "Idle",
        "status_loading": "Loading model…",
        "label_model": "Model",
        "label_lang": "Interface Language",
        "toggle_compare": "Enable Qwen3-ASR comparison (coming soon)",
        "btn_start": "Start Recording",
        "btn_stop": "Stop",
        "btn_upload": "Upload File",
        "btn_recognize": "Start Recognition",
        "btn_recognizing": "Recognizing…",
        "btn_clear": "Clear records",
        "count_suffix": "items",
        "empty": "No records · click Start Recording or Upload File",
        "pending_note": "Pending: audio ready, click \"Start Recognition\" to transcribe",
        "pending_recorded": "Recorded · {dur}s",
        "pending_uploaded": "Uploaded · {name}",
        "spec_truncated": "spectrum shows first 10s only",
        "download": "Download",
        "lang_label": "Recognition language",
        "infer_error": "Recognition failed: {err}",
        "model_loading": "Loading Mega-ASR weights, this may take a while…",
        "router_tag": "router",
    },
    "zh": {
        "brand_sub": "语音识别 · 实时",
        "sec_system": "系统监控",
        "sec_console": "控制台",
        "sec_log": "转写记录",
        "m_cpu": "CPU 使用率",
        "m_mem": "内存",
        "m_gpu": "GPU 使用率",
        "m_vram": "显存",
        "m_disk": "磁盘",
        "m_net": "网络 I/O",
        "status_rec": "录音中",
        "status_proc": "识别中",
        "status_pending": "待识别",
        "status_idle": "待机",
        "status_loading": "加载模型中…",
        "label_model": "模型",
        "label_lang": "界面语言",
        "toggle_compare": "启用 Qwen3-ASR 对比（即将上线）",
        "btn_start": "开始录音",
        "btn_stop": "停止",
        "btn_upload": "上传文件",
        "btn_recognize": "开始识别",
        "btn_recognizing": "识别中…",
        "btn_clear": "清空记录",
        "count_suffix": "条",
        "empty": "暂无记录 · 点击\"开始录音\"或\"上传文件\"",
        "pending_note": "已就绪：音频已加载，点击\"开始识别\"以生成转写",
        "pending_recorded": "录音 · {dur}s",
        "pending_uploaded": "已上传 · {name}",
        "spec_truncated": "频谱仅显示前 10s",
        "download": "下载",
        "lang_label": "识别语言",
        "infer_error": "识别失败：{err}",
        "model_loading": "正在加载 Mega-ASR 权重，请稍候…",
        "router_tag": "路由",
    },
    "ja": {
        "brand_sub": "音声認識 · リアルタイム",
        "sec_system": "システム監視",
        "sec_console": "コンソール",
        "sec_log": "文字起こし",
        "m_cpu": "CPU 使用率",
        "m_mem": "メモリ",
        "m_gpu": "GPU 使用率",
        "m_vram": "VRAM",
        "m_disk": "ディスク",
        "m_net": "ネットワーク I/O",
        "status_rec": "録音中",
        "status_proc": "認識中",
        "status_pending": "認識待ち",
        "status_idle": "待機",
        "status_loading": "モデル読み込み中…",
        "label_model": "モデル",
        "label_lang": "表示言語",
        "toggle_compare": "Qwen3-ASR 比較を有効化（近日公開）",
        "btn_start": "録音開始",
        "btn_stop": "停止",
        "btn_upload": "ファイルをアップロード",
        "btn_recognize": "認識開始",
        "btn_recognizing": "認識中…",
        "btn_clear": "記録をクリア",
        "count_suffix": "件",
        "empty": "記録なし ·「録音開始」または「ファイルをアップロード」",
        "pending_note": "準備完了：音声がロード済み、「認識開始」で文字起こし",
        "pending_recorded": "録音 · {dur}s",
        "pending_uploaded": "アップロード済み · {name}",
        "spec_truncated": "スペクトルは最初の10秒のみ",
        "download": "ダウンロード",
        "lang_label": "認識言語",
        "infer_error": "認識に失敗しました：{err}",
        "model_loading": "Mega-ASR の重みを読み込み中…",
        "router_tag": "ルーター",
    },
}

UI_LANG_LABELS = {
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
}

if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"


def t(key: str) -> str:
    return TRANSLATIONS[st.session_state.ui_lang].get(key, key)


# ──────────────────────────────────────────────
# 配色
# ──────────────────────────────────────────────
GOLD = "#d4af37"
GOLD_SOFT = "#b8941f"
INK = "#1a1a1a"
LINE = "#ececec"
MUTE = "#8a8a8a"

PURPLE = "#7c4dff"
PURPLE_DARK = "#5e35d6"
RED = "#e53935"
RED_DARK = "#c62828"
BLUE = "#1e88e5"
BLUE_DARK = "#1565c0"
GRAY_DISABLED = "#d0d0d0"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', -apple-system, sans-serif;
    }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}

    .brand {{ display:flex; align-items:center; gap:0.9rem; margin-bottom:1.5rem; }}
    .brand img {{ width:38px; height:38px; border-radius:6px; }}
    .brand-title {{ font-size:1.35rem; font-weight:600; color:{INK}; letter-spacing:0.01em; line-height:1.1; }}
    .brand-sub {{ font-size:0.7rem; color:{MUTE}; letter-spacing:0.18em; text-transform:uppercase; margin-top:2px; }}

    .section-label {{
        font-size:0.68rem; font-weight:600; letter-spacing:0.16em;
        text-transform:uppercase; color:#9a9a9a; margin:0 0 0.8rem 0;
    }}

    .panel {{ background:#ffffff; border:1px solid {LINE}; border-radius:8px; padding:1.25rem; }}

    .metric {{ padding:0.9rem 0; border-bottom:1px solid #f3f3f3; }}
    .metric:last-child {{ border-bottom:none; }}
    .metric-label {{ font-size:0.68rem; color:{MUTE}; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.35rem; }}
    .metric-row {{ display:flex; align-items:baseline; justify-content:space-between; gap:0.5rem; }}
    .metric-val {{ font-size:1.5rem; font-weight:600; color:{INK}; font-variant-numeric:tabular-nums; line-height:1; }}
    .metric-unit {{ font-size:0.75rem; color:{MUTE}; font-weight:500; }}
    .metric-bar {{ position:relative; height:3px; background:#f0f0f0; border-radius:2px; margin-top:0.55rem; overflow:hidden; }}
    .metric-bar > span {{ position:absolute; left:0; top:0; bottom:0; background:{INK}; border-radius:2px; transition:width 0.4s ease; }}
    .metric-bar.gold > span {{ background:{GOLD}; }}
    .metric-na {{ color:#c0c0c0; font-size:1.2rem; font-weight:500; }}

    .status-pill {{
        display:inline-flex; align-items:center; gap:0.5rem;
        font-size:0.75rem; color:#555;
        padding:0.3rem 0.7rem; background:#f7f7f7;
        border-radius:999px; border:1px solid {LINE};
        margin-bottom:1rem;
    }}
    .dot {{ width:6px; height:6px; border-radius:50%; background:#c0c0c0; }}
    .dot.live {{ background:{RED}; animation:pulse 1.4s infinite; }}
    .dot.pending {{ background:{BLUE}; }}
    .dot.proc {{ background:{GOLD}; animation:pulse 1.4s infinite; }}
    .dot.loading {{ background:{PURPLE}; animation:pulse 1.4s infinite; }}
    @keyframes pulse {{
        0%   {{ box-shadow:0 0 0 0   rgba(229,57,53,.5); }}
        70%  {{ box-shadow:0 0 0 8px rgba(229,57,53,0); }}
        100% {{ box-shadow:0 0 0 0   rgba(229,57,53,0); }}
    }}

    .stButton > button {{
        width:100%;
        border-radius:6px;
        border:1px solid {INK};
        background:{INK};
        color:#fafafa;
        font-weight:500; letter-spacing:0.04em;
        padding:0.55rem 1rem;
        transition:all 0.15s ease;
    }}
    .stButton > button:hover {{ background:#333; border-color:#333; color:#fff; transform:translateY(-1px); }}

    div[data-purpose="mic-wrap"] {{ min-height:42px; }}
    div[data-purpose="mic-wrap"] [data-testid="stIFrame"],
    div[data-purpose="mic-wrap"] [data-testid="stCustomComponentV1"] {{
        margin:0 !important; padding:0 !important;
    }}
    div[data-purpose="mic-wrap"] iframe {{
        width:100% !important; min-height:42px !important; border:none !important;
    }}
    @keyframes recpulse {{
        0%, 100% {{ box-shadow:0 0 0 0   rgba(229,57,53,0.35); }}
        50%      {{ box-shadow:0 0 0 6px rgba(229,57,53,0.10); }}
    }}

    div[data-purpose="btn-upload"] [data-testid="stFileUploader"] section {{
        padding:0; border:none; background:transparent;
    }}
    div[data-purpose="btn-upload"] [data-testid="stFileUploaderDropzone"] {{
        background:{BLUE} !important;
        border:1px solid {BLUE} !important;
        border-radius:6px;
        min-height:auto !important;
        padding:0 !important;
        transition:all 0.15s ease;
        box-shadow:0 1px 0 rgba(30,136,229,0.25);
    }}
    div[data-purpose="btn-upload"] [data-testid="stFileUploaderDropzone"]:hover {{
        background:{BLUE_DARK} !important; border-color:{BLUE_DARK} !important;
        transform:translateY(-1px);
    }}
    div[data-purpose="btn-upload"] [data-testid="stFileUploaderDropzoneInstructions"] {{
        display:none !important;
    }}
    div[data-purpose="btn-upload"] [data-testid="stFileUploaderDropzone"] button {{
        background:transparent !important;
        border:none !important;
        color:#ffffff !important;
        font-weight:500 !important;
        font-size:1rem !important;
        letter-spacing:0.04em;
        width:100% !important;
        min-height:42px !important;
        padding:0.55rem 1rem !important;
        box-shadow:none !important;
        display:inline-flex !important;
        align-items:center !important;
        justify-content:center !important;
        gap:0.55rem !important;
        line-height:1 !important;
    }}
    div[data-purpose="btn-upload"] [data-testid="stFileUploaderDropzone"] button::before {{
        content:"";
        display:inline-block;
        width:14px; height:14px;
        flex-shrink:0;
        background-color:#ffffff;
        -webkit-mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M11 3h2v8h8v2h-8v8h-2v-8H3v-2h8z"/></svg>') no-repeat center / contain;
                mask: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M11 3h2v8h8v2h-8v8h-2v-8H3v-2h8z"/></svg>') no-repeat center / contain;
    }}
    div[data-purpose="btn-upload"] [data-testid="stFileUploaderFile"] {{ font-size:0.72rem; }}

    div[data-purpose="btn-recognize-disabled"] .stButton > button {{
        background:#f2f2f2 !important;
        border:1px solid {GRAY_DISABLED} !important;
        color:#a0a0a0 !important;
        cursor:not-allowed;
    }}
    div[data-purpose="btn-recognize-disabled"] .stButton > button:hover {{
        background:#f2f2f2 !important; transform:none;
        border-color:{GRAY_DISABLED} !important; color:#a0a0a0 !important;
    }}
    div[data-purpose="btn-recognize-active"] .stButton > button {{
        background:{INK}; border-color:{INK}; color:#ffffff;
    }}
    div[data-purpose="btn-recognize-active"] .stButton > button:hover {{
        background:#000; border-color:#000;
    }}

    div[data-purpose="btn-clear"] .stButton > button {{
        background:#ffffff; color:{INK}; border:1px solid #d8d8d8;
    }}
    div[data-purpose="btn-clear"] .stButton > button:hover {{
        background:#f5f5f5; border-color:{INK};
    }}

    .stSelectbox label, .stToggle label, .stRadio label {{
        font-size:0.78rem !important; color:#555 !important; font-weight:500 !important;
    }}

    /* 禁用态的 toggle（对比按钮）整体变灰 */
    div[data-purpose="toggle-disabled"] [data-testid="stToggle"] {{
        opacity: 0.45;
        pointer-events: none;
    }}
    div[data-purpose="toggle-disabled"] [data-testid="stToggle"] label {{
        color: #a0a0a0 !important;
    }}

    .pending-card {{
        margin-top:0.9rem; padding:0.8rem 0.9rem;
        background:#eef5fd; border:1px solid #cfe3f7;
        border-radius:6px;
        font-size:0.78rem; color:#1d4f7c; line-height:1.5;
    }}
    .pending-title {{
        font-weight:600; font-size:0.72rem;
        letter-spacing:0.1em; text-transform:uppercase;
        color:{BLUE_DARK}; margin-bottom:0.3rem;
    }}
    .error-card {{
        margin-top:0.9rem; padding:0.8rem 0.9rem;
        background:#fdecea; border:1px solid #f5c2c0;
        border-radius:6px;
        font-size:0.78rem; color:#7c1d1d; line-height:1.5;
    }}

    .entry {{ padding:1.1rem 0 1.25rem 0; border-bottom:1px solid #f0f0f0; }}
    .entry:last-child {{ border-bottom:none; }}
    .entry-meta {{
        font-size:0.72rem; color:#a8a8a8;
        letter-spacing:0.04em; font-variant-numeric:tabular-nums;
        margin-bottom:0.6rem; display:flex; gap:0.6rem; align-items:center;
        flex-wrap:wrap;
    }}
    .entry-meta .sep {{ color:#d0d0d0; }}
    .entry-meta .badge {{
        font-size:0.65rem;
        padding:0.1rem 0.45rem;
        border:1px solid {LINE};
        border-radius:4px;
        color:{MUTE};
        letter-spacing:0.08em;
        text-transform:uppercase;
    }}
    .entry-meta .badge.gold {{
        border-color:#e9d391; color:{GOLD_SOFT}; background:#fffaf0;
    }}
    .entry-text {{ font-size:0.98rem; color:#1f1f1f; line-height:1.65; }}

    .spec-wrap {{ background:#000; border-radius:6px; padding:0; margin:0.6rem 0 0.7rem 0; overflow:hidden; }}
    audio {{ width:100%; height:34px; margin-top:0.2rem; }}

    .dl-link {{
        display:inline-block; font-size:0.72rem; color:{MUTE};
        text-decoration:none; padding:0.25rem 0.65rem;
        border:1px solid {LINE}; border-radius:4px; margin-left:0.5rem;
        transition:all 0.15s ease;
    }}
    .dl-link:hover {{ color:{INK}; border-color:{INK}; }}

    .empty {{ text-align:center; padding:4rem 1rem; color:#bdbdbd; font-size:0.9rem; }}
    .empty-mark {{ font-size:2rem; color:#e0e0e0; margin-bottom:0.8rem; font-weight:200; }}

    hr {{ border:none; border-top:1px solid {LINE}; margin:1rem 0; }}
    </style>
    <script>
    const setUploadLabel = () => {{
        document.querySelectorAll('div[data-purpose="btn-upload"] [data-testid="stFileUploaderDropzone"] button')
            .forEach(b => {{
                const label = document.querySelector('div[data-purpose="btn-upload"]').getAttribute('data-label');
                if (label) {{
                    b.setAttribute('data-label', label);
                    if (!b.textContent.trim() || b.textContent !== label) b.textContent = label;
                }}
            }});
    }};

    const MIC_STYLE_ID = 'mega-asr-mic-style';
    const MIC_CSS = `
      :root, body {{ margin:0 !important; padding:0 !important; background:transparent !important; }}
      .myButton {{
        all:unset;
        box-sizing:border-box;
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        gap:0.55rem;
        width:100% !important;
        min-height:42px !important;
        padding:0.55rem 1rem !important;
        background:#7c4dff !important;
        border:1px solid #7c4dff !important;
        border-radius:6px !important;
        color:#ffffff !important;
        font-family:'Inter','PingFang SC','Microsoft YaHei',-apple-system,sans-serif !important;
        font-size:1rem !important;
        font-weight:500 !important;
        letter-spacing:0.04em !important;
        cursor:pointer !important;
        transition:all 0.15s ease !important;
        box-shadow:0 1px 0 rgba(124,77,255,0.25) !important;
      }}
      .myButton:hover {{
        background:#5e35d6 !important;
        border-color:#5e35d6 !important;
        transform:translateY(-1px);
      }}
      .myButton.is-recording {{
        background:#e53935 !important;
        border-color:#e53935 !important;
        box-shadow:0 0 0 4px rgba(229,57,53,0.18) !important;
        animation: megaRec 1.6s ease-in-out infinite;
      }}
      .myButton.is-recording:hover {{
        background:#c62828 !important;
        border-color:#c62828 !important;
        transform:none;
      }}
      @keyframes megaRec {{
        0%, 100% {{ box-shadow:0 0 0 0 rgba(229,57,53,0.35) !important; }}
        50%      {{ box-shadow:0 0 0 6px rgba(229,57,53,0.10) !important; }}
      }}
      .myButton::before {{
        content:"";
        display:inline-block;
        width:14px; height:14px;
        background:#ffffff;
        flex-shrink:0;
        -webkit-mask: var(--mega-icon) no-repeat center / contain;
                mask: var(--mega-icon) no-repeat center / contain;
        --mega-icon: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/></svg>');
      }}
      .myButton.is-recording::before {{
        --mega-icon: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>');
      }}
    `;

    const wireMicIframe = (iframe) => {{
        try {{
            const doc = iframe.contentDocument;
            if (!doc || !doc.body) return false;
            if (!doc.getElementById(MIC_STYLE_ID)) {{
                const s = doc.createElement('style');
                s.id = MIC_STYLE_ID;
                s.textContent = MIC_CSS;
                doc.head.appendChild(s);
            }}
            const wrap = iframe.closest('div[data-purpose="mic-wrap"]');
            const stopLabel = wrap ? wrap.getAttribute('data-stop-label') : '';
            const refresh = () => {{
                const btn = doc.querySelector('.myButton');
                if (!btn) return;
                const txt = (btn.textContent || '').trim();
                if (stopLabel && txt === stopLabel) {{
                    btn.classList.add('is-recording');
                }} else {{
                    btn.classList.remove('is-recording');
                }}
            }};
            refresh();
            if (!iframe._megaObs) {{
                iframe._megaObs = new MutationObserver(refresh);
                iframe._megaObs.observe(doc.body, {{ childList:true, subtree:true, characterData:true }});
            }}
            return true;
        }} catch (e) {{
            console.warn('[mega-asr] cannot access mic iframe:', e);
            return false;
        }}
    }};

    const wireAllMicIframes = () => {{
        document.querySelectorAll('div[data-purpose="mic-wrap"] iframe').forEach(iframe => {{
            if (iframe._megaWired) return;
            const ok = wireMicIframe(iframe);
            if (ok) {{
                iframe._megaWired = true;
            }} else {{
                iframe.addEventListener('load', () => {{
                    if (wireMicIframe(iframe)) iframe._megaWired = true;
                }}, {{ once:true }});
            }}
        }});
    }};

    const tick = () => {{ setUploadLabel(); wireAllMicIframes(); }};
    new MutationObserver(tick).observe(document.body, {{ childList:true, subtree:true }});
    tick();
    </script>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 系统指标
# ──────────────────────────────────────────────
def read_cpu_percent():
    if not HAS_PSUTIL: return None
    try: return psutil.cpu_percent(interval=None)
    except Exception: return None

def read_mem_percent():
    if not HAS_PSUTIL: return None
    try: return psutil.virtual_memory().percent
    except Exception: return None

def read_gpu():
    if HAS_NVML:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            return util, mem.used / mem.total * 100
        except Exception:
            return None, None
    return None, None

def read_disk_percent():
    if not HAS_PSUTIL: return None
    try:
        path = "C:\\" if OS_NAME == "Windows" else "/"
        return psutil.disk_usage(path).percent
    except Exception:
        return None

def read_net_kbps():
    if not HAS_PSUTIL: return None
    try:
        now = psutil.net_io_counters()
        ts = time.time()
        prev = st.session_state.get("_net_prev")
        st.session_state._net_prev = (now.bytes_sent + now.bytes_recv, ts)
        if prev is None: return 0.0
        dbytes = (now.bytes_sent + now.bytes_recv) - prev[0]
        dt = max(ts - prev[1], 1e-6)
        return max(dbytes / dt / 1024, 0)
    except Exception:
        return None


def render_metric(label, value, unit="%", bar=True, gold=False, max_bar=100):
    if value is None:
        body = '<div class="metric-row"><span class="metric-na">—</span></div>'
    else:
        v_display = f"{value:.1f}" if isinstance(value, float) else str(value)
        body = (
            f'<div class="metric-row">'
            f'<span class="metric-val">{v_display}</span>'
            f'<span class="metric-unit">{unit}</span>'
            f"</div>"
        )
        if bar:
            pct = min(max(value / max_bar * 100, 0), 100)
            cls = "metric-bar gold" if gold else "metric-bar"
            body += f'<div class="{cls}"><span style="width:{pct:.1f}%"></span></div>'
    return f'<div class="metric"><div class="metric-label">{label}</div>{body}</div>'


# ──────────────────────────────────────────────
# 频谱（保留原实现，未改动）
# ──────────────────────────────────────────────
def decode_audio(audio_bytes: bytes, mime_hint: str = "") -> tuple:
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def compute_spectrogram(samples: np.ndarray, sr: int,
                         n_bins: int = 32, n_frames: int = 128,
                         f_max: int = 4000) -> np.ndarray:
    if samples is None or len(samples) < 64:
        return np.zeros((n_bins, n_frames), dtype=np.float32)
    samples = samples - samples.mean()
    peak = float(np.max(np.abs(samples))) or 1.0
    samples = samples / peak
    win_size = max(256, min(2048, len(samples) // n_frames))
    starts = np.linspace(0, max(0, len(samples) - win_size), n_frames).astype(int)
    window = np.hanning(win_size).astype(np.float32)
    spec_lin = np.zeros((win_size // 2 + 1, n_frames), dtype=np.float32)
    for i, s in enumerate(starts):
        seg = samples[s:s + win_size]
        if len(seg) < win_size:
            seg = np.pad(seg, (0, win_size - len(seg)))
        fft = np.fft.rfft(seg * window)
        spec_lin[:, i] = np.abs(fft)
    freqs = np.fft.rfftfreq(win_size, 1 / sr)
    cap = np.searchsorted(freqs, f_max)
    spec_lin = spec_lin[:cap, :] if cap > n_bins else spec_lin
    n_freq = spec_lin.shape[0]
    log_edges = np.logspace(np.log10(1), np.log10(n_freq), n_bins + 1).astype(int)
    log_edges = np.clip(log_edges, 0, n_freq)
    binned = np.zeros((n_bins, n_frames), dtype=np.float32)
    for b in range(n_bins):
        lo, hi = log_edges[b], max(log_edges[b + 1], log_edges[b] + 1)
        binned[b] = spec_lin[lo:hi, :].mean(axis=0) if hi > lo else spec_lin[lo, :]
    eps = 1e-6
    db = 20.0 * np.log10(binned + eps)
    db -= db.max()
    db = np.clip(db, -60.0, 0.0)
    norm = (db + 60.0) / 60.0
    norm = np.power(norm, 1.4)
    return norm.astype(np.float32)


def spectrogram_svg(spec: np.ndarray, height: int = 96) -> str:
    n_bins, n_frames = spec.shape
    cell_w = 5
    width = n_frames * cell_w
    cell_h = height / n_bins
    rects = []
    for b in range(n_bins):
        row = n_bins - 1 - b
        y = row * cell_h
        for f in range(n_frames):
            v = float(spec[b, f])
            if v < 0.04:
                continue
            if v < 0.35:
                k = v / 0.35
                r = int(0 + (90 - 0) * k)
                g = int(0 + (61 - 0) * k)
                bl = 0
            elif v < 0.75:
                k = (v - 0.35) / 0.4
                r = int(90 + (212 - 90) * k)
                g = int(61 + (175 - 61) * k)
                bl = int(0 + (55 - 0) * k)
            else:
                k = (v - 0.75) / 0.25
                r = int(212 + (255 - 212) * k)
                g = int(175 + (216 - 175) * k)
                bl = int(55 + (100 - 55) * k)
            color = f"rgb({r},{g},{bl})"
            x = f * cell_w
            rects.append(
                f'<rect x="{x}" y="{y:.2f}" width="{cell_w}" '
                f'height="{cell_h:.2f}" fill="{color}"/>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none" '
        f'style="display:block;background:#000;">'
        + "".join(rects) +
        "</svg>"
    )


def audio_data_uri(wav_bytes: bytes, mime: str = "audio/wav") -> str:
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:{mime};base64,{b64}"


def estimate_wav_duration(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return min(max(len(wav_bytes) / 32000, 1.0), 30.0)


# ──────────────────────────────────────────────
# 后端模型加载（全局只加载一次）
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_megaasr_model():
    """
    返回 MegaASR 实例。第一次调用时会加载权重，之后被 Streamlit 缓存复用。
    若加载失败，返回 (None, error_message)；成功返回 (model, None)。
    """
    try:
        from MegaASR.model.megaASR import MegaASR
    except Exception as e:
        return None, f"Failed to import MegaASR: {e}"

    try:
        kwargs = dict(
            model_path=CKPT_DIR / "Qwen3-ASR-1.7B",
            lora_dir=CKPT_DIR / "mega-asr-merged",
            router_checkpoint=CKPT_DIR / "audio_quality_router/best_acc_model.safetensors",
            routing_enabled=ROUTING_ENABLED,
            quality_threshold=ROUTING_THRESHOLD,
        )
        if DEVICE_MAP is not None:
            kwargs["device_map"] = DEVICE_MAP
        model = MegaASR(**kwargs)
        return model, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _bytes_to_wav_path(audio_bytes: bytes, mime: str = "audio/wav") -> Path:
    """
    把音频 bytes 落盘成 wav（MegaASR.infer 接收文件路径）。
    非 wav 来源（mp3/m4a/flac/ogg）会用 soundfile 重新解码再写成 wav，
    以保证兼容性。
    """
    tmp_dir = Path(tempfile.gettempdir()) / "mega-asr-webui"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"clip_{int(time.time()*1000)}_{os.getpid()}.wav"

    try:
        # soundfile 支持 wav/flac/ogg；mp3/m4a 在多数环境也能通过 libsndfile 读
        samples, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        sf.write(str(out_path), samples, sr, subtype="PCM_16")
    except Exception:
        # 解码失败 fallback：直接把原始 bytes 写盘（wav 录音肯定能走这条）
        out_path.write_bytes(audio_bytes)
    return out_path


def run_inference(audio_bytes: bytes, mime: str) -> dict:
    """
    调用真实后端。返回:
      {"text": str, "route": optional, "raw": original_result,
       "infer_time": float, "audio_duration": float, "rtf": float, "rtfx": float}
    抛出异常时由上层捕获并展示。
    """
    model, err = load_megaasr_model()
    if model is None:
        raise RuntimeError(err or "model not loaded")

    wav_path = _bytes_to_wav_path(audio_bytes, mime)
    try:
        audio_duration = sf.info(str(wav_path)).duration
        t0 = time.perf_counter()
        result = model.infer(wav_path, return_route=True)
        infer_time = time.perf_counter() - t0
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass

    # 兼容多种返回形态:
    # - dict, 含 "text" / "transcription" / "result"
    # - (text, route) 元组
    # - 单纯字符串
    text, route = "", None
    if isinstance(result, dict):
        text = (
            result.get("text")
            or result.get("transcription")
            or result.get("result")
            or result.get("output")
            or ""
        )
        route = result.get("route", result.get("router", None))
    elif isinstance(result, (list, tuple)):
        if len(result) >= 1:
            text = result[0] if isinstance(result[0], str) else str(result[0])
        if len(result) >= 2:
            route = result[1]
    else:
        text = str(result)

    rtf = infer_time / audio_duration if audio_duration > 0 else 0.0
    rtfx = 1.0 / rtf if rtf > 0 else 0.0

    return {"text": text, "route": route, "raw": result,
            "infer_time": infer_time, "audio_duration": audio_duration,
            "rtf": rtf, "rtfx": rtfx}


# ──────────────────────────────────────────────
# 状态
# ──────────────────────────────────────────────
if "records" not in st.session_state:
    st.session_state.records = []
if "seed_counter" not in st.session_state:
    st.session_state.seed_counter = 1
if "pending" not in st.session_state:
    st.session_state.pending = None
if "_upload_token" not in st.session_state:
    st.session_state._upload_token = 0
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "last_error" not in st.session_state:
    st.session_state.last_error = None


def commit_pending_as_record():
    """把 pending 转成正式 record（调用真实后端识别）"""
    p = st.session_state.pending
    if not p:
        return

    # 真实推理
    try:
        infer_out = run_inference(p["audio"], p.get("mime", "audio/wav"))
        text = infer_out["text"] or ""
        route = infer_out["route"]
        rtf = infer_out.get("rtf", 0.0)
        rtfx = infer_out.get("rtfx", 0.0)
        infer_time = infer_out.get("infer_time", 0.0)
    except Exception as e:
        st.session_state.last_error = t("infer_error").format(err=str(e))
        # 不落盘记录，pending 保留让用户可以重试
        return

    # 真实频谱
    try:
        samples, sr = decode_audio(p["audio"], p.get("mime", "audio/wav"))
        spec = compute_spectrogram(samples, sr or 16000)
    except Exception:
        spec = np.zeros((32, 128), dtype=np.float32)

    record = {
        "id": p["seed"],
        "time": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "language": UI_LANG_LABELS[st.session_state.ui_lang],
        "duration": p["duration"],
        "text": text,
        "route": route,
        "rtf": rtf,
        "rtfx": rtfx,
        "infer_time": infer_time,
        "audio": p["audio"],
        "audio_mime": p.get("mime", "audio/wav"),
        "spectrogram": spec,
        "source": p["source"],
        "file_name": p.get("name"),
    }
    st.session_state.records.insert(0, record)
    st.session_state.pending = None
    st.session_state.last_error = None


# ──────────────────────────────────────────────
# Logo + 标题
# ──────────────────────────────────────────────
logo_path = ROOT_DIR / "logo.png"
if logo_path.exists():
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" alt="logo"/>'
else:
    logo_html = '<div style="width:38px;height:38px;background:#1a1a1a;border-radius:6px;"></div>'

st.markdown(
    f"""
    <div class="brand">
        {logo_html}
        <div>
            <div class="brand-title">Mega-ASR Console</div>
            <div class="brand-sub">{t("brand_sub")}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# 启动时预热模型（首次访问时显示加载提示）
# ──────────────────────────────────────────────
if "model_warmed" not in st.session_state:
    with st.spinner(t("model_loading")):
        _, _err = load_megaasr_model()
    st.session_state.model_warmed = True
    if _err:
        st.session_state.last_error = _err

# ──────────────────────────────────────────────
# 三栏布局
# ──────────────────────────────────────────────
col_sys, col_ctrl, col_log = st.columns([1, 1, 3], gap="large")

# ============== 左栏：系统监控 ==============
with col_sys:
    st.markdown(f'<div class="section-label">{t("sec_system")}</div>', unsafe_allow_html=True)

    cpu = read_cpu_percent()
    mem = read_mem_percent()
    gpu_util, gpu_mem = read_gpu()
    disk = read_disk_percent()
    net = read_net_kbps()

    html = '<div class="panel">'
    html += render_metric(t("m_cpu"), cpu, "%", gold=True)
    html += render_metric(t("m_mem"), mem, "%")
    html += render_metric(t("m_gpu"), gpu_util, "%", gold=True)
    html += render_metric(t("m_vram"), gpu_mem, "%")
    html += render_metric(t("m_disk"), disk, "%")
    html += render_metric(t("m_net"), net, "KB/s", max_bar=512)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:0.68rem;color:#b0b0b0;margin-top:0.8rem;'
        f'letter-spacing:0.1em;text-align:center;">'
        f'{OS_NAME.upper()} · {platform.machine()}</div>',
        unsafe_allow_html=True,
    )

# ============== 中栏：控制台 ==============
with col_ctrl:
    st.markdown(f'<div class="section-label">{t("sec_console")}</div>', unsafe_allow_html=True)

    if st.session_state.is_processing:
        dot_cls, status_label = "proc", t("status_proc")
    elif st.session_state.pending is not None:
        dot_cls, status_label = "pending", t("status_pending")
    else:
        dot_cls, status_label = "", t("status_idle")
    st.markdown(
        f'<div class="status-pill"><span class="dot {dot_cls}"></span>{status_label}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.7rem;color:{MUTE};letter-spacing:0.12em;'
        f'text-transform:uppercase;margin-bottom:0.3rem;">{t("label_model")}</div>'
        f'<div style="font-size:1rem;color:{INK};font-weight:600;'
        f'margin-bottom:1rem;">Mega-ASR</div>',
        unsafe_allow_html=True,
    )

    lang_codes = list(UI_LANG_LABELS.keys())
    cur_idx = lang_codes.index(st.session_state.ui_lang)
    chosen_label = st.selectbox(
        t("label_lang"),
        [UI_LANG_LABELS[c] for c in lang_codes],
        index=cur_idx,
        key="ui_lang_select",
    )
    chosen_code = lang_codes[[UI_LANG_LABELS[c] for c in lang_codes].index(chosen_label)]
    if chosen_code != st.session_state.ui_lang:
        st.session_state.ui_lang = chosen_code
        st.rerun()

    # ── Qwen3-ASR 对比：禁用 ─────────────────
    st.markdown('<div data-purpose="toggle-disabled">', unsafe_allow_html=True)
    st.toggle(t("toggle_compare"), value=False, key="compare_toggle", disabled=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 录音 ───────────────────────────────────
    st.markdown(
        f'<div data-purpose="mic-wrap" '
        f'data-start-label="{t("btn_start")}" '
        f'data-stop-label="{t("btn_stop")}">',
        unsafe_allow_html=True,
    )
    mic_value = mic_recorder(
        start_prompt=t("btn_start"),
        stop_prompt=t("btn_stop"),
        just_once=True,
        use_container_width=True,
        format="wav",
        key="mic_rec",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if mic_value is not None and mic_value.get("id") != st.session_state.get("_last_mic_id"):
        st.session_state._last_mic_id = mic_value.get("id")
        audio_bytes = mic_value["bytes"]
        sr_in = mic_value.get("sample_rate", 16000)
        sw_in = mic_value.get("sample_width", 2)
        try:
            duration = round(len(audio_bytes) / (sr_in * sw_in), 1)
        except Exception:
            duration = round(estimate_wav_duration(audio_bytes), 1)
        duration = max(duration, 0.5)
        seed = st.session_state.seed_counter
        st.session_state.seed_counter += 1
        st.session_state.pending = {
            "source": "record",
            "audio": audio_bytes,
            "mime": "audio/wav",
            "duration": duration,
            "seed": seed,
        }
        st.session_state.last_error = None
        st.rerun()

    # ── 上传 ───────────────────────────────────
    st.markdown(
        f'<div data-purpose="btn-upload" data-label="{t("btn_upload")}" '
        f'style="margin-top:0.55rem;">',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        t("btn_upload"),
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        key=f"uploader_{st.session_state._upload_token}",
        label_visibility="collapsed",
        accept_multiple_files=False,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        mime = uploaded.type or "audio/wav"
        duration = estimate_wav_duration(audio_bytes) if mime.endswith("wav") else \
                   min(max(len(audio_bytes) / 32000, 1.0), 30.0)
        seed = st.session_state.seed_counter
        st.session_state.seed_counter += 1
        st.session_state.pending = {
            "source": "upload",
            "audio": audio_bytes,
            "mime": mime,
            "duration": round(duration, 1),
            "name": uploaded.name,
            "seed": seed,
        }
        st.session_state._upload_token += 1
        st.session_state.last_error = None
        st.rerun()

    # ── 开始识别 按钮 ─────────────────────────
    has_pending = st.session_state.pending is not None
    rec_purpose = "btn-recognize-active" if has_pending else "btn-recognize-disabled"
    rec_label = t("btn_recognizing") if st.session_state.is_processing else t("btn_recognize")
    st.markdown(f'<div data-purpose="{rec_purpose}" style="margin-top:0.6rem;">',
                unsafe_allow_html=True)
    recognize_clicked = st.button(
        rec_label,
        key="recognize_btn",
        use_container_width=True,
        disabled=not has_pending or st.session_state.is_processing,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if recognize_clicked and has_pending:
        st.session_state.is_processing = True
        st.rerun()

    # ── 待识别提示卡 ──────────────────────────
    if has_pending and not st.session_state.is_processing:
        p = st.session_state.pending
        if p["source"] == "record":
            desc = t("pending_recorded").format(dur=f'{p["duration"]:.1f}')
        else:
            desc = t("pending_uploaded").format(name=p.get("name", ""))
        st.markdown(
            f'<div class="pending-card">'
            f'<div class="pending-title">{t("status_pending")}</div>'
            f'{desc}<br>{t("pending_note")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 错误提示卡 ────────────────────────────
    if st.session_state.last_error:
        st.markdown(
            f'<div class="error-card">{st.session_state.last_error}</div>',
            unsafe_allow_html=True,
        )

    # ── 清空 ───────────────────────────────────
    st.markdown('<div data-purpose="btn-clear" style="margin-top:0.6rem;">',
                unsafe_allow_html=True)
    if st.button(t("btn_clear"), key="clr_btn", use_container_width=True):
        st.session_state.records = []
        st.session_state.pending = None
        st.session_state.is_processing = False
        st.session_state.last_error = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ============== 右栏：转写记录 ==============
with col_log:
    head_l, head_r = st.columns([3, 1])
    with head_l:
        st.markdown(f'<div class="section-label">{t("sec_log")}</div>', unsafe_allow_html=True)
    with head_r:
        st.markdown(
            f'<div style="text-align:right;font-size:0.72rem;color:{MUTE};'
            f'margin-top:2px;">{len(st.session_state.records)} {t("count_suffix")}</div>',
            unsafe_allow_html=True,
        )

    if not st.session_state.records:
        st.markdown(
            f'<div class="panel"><div class="empty">'
            f'<div class="empty-mark">◌</div>'
            f'{t("empty")}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        for r in st.session_state.records:
            duration_disp = f"{r['duration']:.1f}s"

            src_tag = ""
            if r.get("source") == "upload" and r.get("file_name"):
                src_tag = (f'<span class="sep">·</span>'
                           f'<span>↑ {r["file_name"]}</span>')

            # 路由信息显示
            route_tag = ""
            route = r.get("route")
            if route is not None:
                if isinstance(route, dict):
                    # 取出最常见的字段:决定 / score
                    decision = (
                        route.get("decision")
                        or route.get("route")
                        or route.get("label")
                        or route.get("model")
                    )
                    score = route.get("score") or route.get("quality") or route.get("prob")
                    parts = []
                    if decision is not None:
                        parts.append(str(decision))
                    if isinstance(score, (int, float)):
                        parts.append(f"{float(score):.2f}")
                    txt = " ".join(parts) if parts else str(route)
                else:
                    txt = str(route)
                route_tag = (
                    f'<span class="sep">·</span>'
                    f'<span class="badge gold">{t("router_tag")}: {txt}</span>'
                )

            # RTF/RTFx 显示
            rtf_tag = ""
            if r.get("rtf") and r.get("rtfx"):
                rtf_tag = (
                    f'<span class="sep">·</span>'
                    f'<span class="badge">RTF {r["rtf"]:.3f} · {r["rtfx"]:.1f}x</span>'
                )

            meta = (
                f'<div class="entry-meta">'
                f'<span>{r["date"]} {r["time"]}</span>'
                f'<span class="sep">·</span>'
                f'<span>{r["language"]}</span>'
                f'<span class="sep">·</span>'
                f'<span>{duration_disp}</span>'
                f'{src_tag}'
                f'{route_tag}'
                f'{rtf_tag}'
                f"</div>"
            )

            svg = spectrogram_svg(r["spectrogram"])
            spec = f'<div class="spec-wrap">{svg}</div>'

            uri = audio_data_uri(r["audio"], r.get("audio_mime", "audio/wav"))
            ext = "wav"
            if r.get("audio_mime", "").endswith("mpeg"): ext = "mp3"
            elif r.get("audio_mime", "").endswith("mp4"): ext = "m4a"
            elif r.get("audio_mime", "").endswith("flac"): ext = "flac"
            elif r.get("audio_mime", "").endswith("ogg"): ext = "ogg"
            player = (
                f'<div style="display:flex;align-items:center;gap:0.5rem;'
                f'margin-bottom:0.3rem;">'
                f'<audio controls preload="none" src="{uri}"></audio>'
                f'<a class="dl-link" href="{uri}" '
                f'download="mega-asr-{r["id"]}.{ext}">{t("download")}</a>'
                f"</div>"
            )

            body = f'<div class="entry-text">{r["text"]}</div>'

            st.markdown(
                f'<div class="entry">{meta}{spec}{player}{body}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 后台逻辑：执行识别 / 监控刷新
# ──────────────────────────────────────────────
if st.session_state.is_processing:
    # 真正阻塞调用后端
    commit_pending_as_record()
    st.session_state.is_processing = False
    st.rerun()
else:
    # 监控刷新：放慢节奏，避免在录音过程中频繁 rerun 干扰 mic_recorder
    time.sleep(2.0)
    st.rerun()
