#!/usr/bin/env python3
"""
Telegram Bot for Chinese SRT Extractor & Translator
Gửi video hoặc URL → nhận file SRT + audio TTS
"""

import os
import json
import sys
import asyncio
import uuid
import time
import threading
from pathlib import Path

# Ensure we can import from app.py
sys.path.insert(0, os.path.dirname(__file__))

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ── Import shared processing functions from modular structure ────────────────
from services.douyin_monitor import (
    get_channels, add_channel, remove_channel, toggle_channel,
    get_downloaded_history, is_monitor_running, get_monitor_status,
    start_monitor, stop_monitor, scan_now, resolve_channel_sec_uid,
    set_notify_chat_id
)

from config import (
    UPLOAD_FOLDER,
    OUTPUT_FOLDER,
    LANGUAGES,
    TTS_VOICES,
    jobs, create_job,
    safe_stem,
    start_cleanup_worker,
    DEVICE,
    COMPUTE_TYPE,
    load_presets, save_presets, validate_region,
    GEMINI_MODELS, GEMINI_DEFAULT_MODEL,
    AI_TRANSLATE_MODELS, AI_DEFAULT_MODEL,
    get_gemini_api_key, set_gemini_api_key,
    TRANSLATION_MODES, DEFAULT_TRANSLATION_MODE,
)
from services.whisper_engine import process_video
from services.downloader import download_from_url
from services.tts import generate_tts_audio
from services.burn_sub import burn_sub_video
from services.hardsub_gemini import hardsub_worker

# ── Config ─────────────────────────────────────────────────────────────────
from config import TELEGRAM_BOT_TOKEN as BOT_TOKEN

# User preferences (in-memory, per chat_id)
PREFS_FILE = Path(os.getenv("TELEGRAM_PREFS_FILE", str(Path(__file__).parent / "config" / "user_prefs.json")))
_PREFS_LOCK = threading.RLock()


def _load_user_prefs():
    try:
        if PREFS_FILE.exists():
            data = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as exc:
        print(f"⚠️ Không đọc được user preferences: {exc}")
    return {}


def _save_user_prefs():
    try:
        PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PREFS_FILE.with_suffix(".tmp")
        with _PREFS_LOCK:
            tmp.write_text(json.dumps(user_prefs, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(PREFS_FILE)
    except OSError as exc:
        print(f"⚠️ Không lưu được user preferences: {exc}")


user_prefs = _load_user_prefs()

DEFAULT_LANGS = ["vi", "en", "id"]
DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_TTS = False


def get_prefs(chat_id):
    """Get or create user preferences"""
    if chat_id not in user_prefs:
        user_prefs[chat_id] = {
            "langs": list(DEFAULT_LANGS),
            "model": DEFAULT_MODEL,
            "tts": DEFAULT_TTS,
            "burn": False,
            "preset": "default",
            "hardsub": False,
            "gemini_model": GEMINI_DEFAULT_MODEL,
            "tts_engine": "edge",
        }
    # Migration: add new keys if missing
    p = user_prefs[chat_id]
    if "hardsub" not in p:
        p["hardsub"] = False
    if "gemini_model" not in p:
        p["gemini_model"] = GEMINI_DEFAULT_MODEL
    if "translation_mode" not in p:
        p["translation_mode"] = DEFAULT_TRANSLATION_MODE
    if "vmode" not in p:
        p["vmode"] = "blur"
    if "tts_engine" not in p:
        p["tts_engine"] = "edge"
    if "ai_model" not in p:
        p["ai_model"] = AI_DEFAULT_MODEL
    if "trim_intro" not in p:
        p["trim_intro"] = "auto"
    if "bgm_mode" not in p:
        p["bgm_mode"] = "auto"
    if "bgm_volume" not in p:
        p["bgm_volume"] = 0.8
    if "clean_hardsub" not in p:
        p["clean_hardsub"] = True
    if "clean_logo" not in p:
        p["clean_logo"] = False
    if "clean_title" not in p:
        p["clean_title"] = False
    if "translate_title" not in p:
        p["translate_title"] = False
    if "brand_name" not in p:
        p["brand_name"] = ""
    _save_user_prefs()
    return p


# ── Command Handlers ──────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    try:
        set_notify_chat_id(update.effective_chat.id)
    except Exception:
        pass
    welcome = (
        "🎬 *Chinese SRT Extractor & Translator Bot*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Gửi cho tôi *video* hoặc *link* tiếng Trung, tôi sẽ:\n"
        "1️⃣ Nhận dạng giọng nói → file .srt tiếng Trung\n"
        "2️⃣ Dịch sang tiếng Việt & English\n"
        "3️⃣ Tạo audio TTS từ phụ đề\n\n"
        "📎 *Cách dùng:*\n"
        "• Gửi trực tiếp file video\n"
        "• Hoặc dán link (Douyin, YouTube, Bilibili...)\n\n"
        "⚙️ *Cài đặt:*\n"
        "/lang - Chọn ngôn ngữ dịch\n"
        "/tts - Bật/tắt tạo audio\n"
        "/model - Chọn AI model\n"
        "/mode - 🎯 Phong cách dịch (movie: Phim | driving: Dạy lái xe)\n"
        "/hardsub - 🔍 Bật/tắt trích hardsub (Gemini)\n"
        "/gemini - 🔑 Cài đặt Gemini API\n"
        "/status - Xem cài đặt hiện tại\n"
        "/help - Xem hướng dẫn\n\n"
        f"🖥️ GPU: *{DEVICE.upper()}* | Compute: *{COMPUTE_TYPE}*"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command — display comprehensive user manual."""
    help_text = (
        "📖 **HƯỚNG DẪN SỬ DỤNG BOT**\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "🎬 **1. GỬI FILE / LINK ĐỂ XỬ LÝ:**\n"
        "• **Gửi video trực tiếp:** File MP4, MOV, MKV... (tối đa 50MB)\n"
        "• **Gửi audio:** MP3, WAV, M4A, Voice message...\n"
        "• **Gửi link video:** Douyin, YouTube, Bilibili, TikTok...\n\n"
        "🌍 **2. DỊCH THUẬT & PHỤ ĐỀ:**\n"
        "• `/lang [vi|en|id]` — Chọn ngôn ngữ dịch đích\n"
        "• `/mode [driving|movie|literal|fun]` — Chọn phong cách dịch AI\n"
        "• `/aimodel [tên_model]` — Chọn model AI dịch thuật\n"
        "• `/model [large-v3-turbo|large-v3]` — Chọn model Whisper\n"
        "• `/hardsub [on|off]` — Bật/tắt trích chữ hardsub (Gemini OCR)\n"
        "• `/gemini [key|model]` — Cài đặt Gemini API Key\n\n"
        "🎙️ **3. LỒNG TIẾNG & NHẠC NỀN (TTS & BGM):**\n"
        "• `/tts [on|off]` — Bật/tắt lồng tiếng tự động\n"
        "• `/voice [edge|omnivoice|gemini]` — Chọn giọng đọc AI\n"
        "• `/bgm [auto|ai|off|<5-100>]` — Chế độ nhạc nền & âm lượng\n\n"
        "🧹 **4. XỬ LÝ VIDEO & CLEAN PLATE:**\n"
        "• `/vmode [clean|inpaint_burn|blur|pure_burn]` — Chế độ render video\n"
        "• `/clean [sub|logo|title|all]` — Tùy chọn xóa chữ & logo cũ\n"
        "• `/logo [on|off]` — Bật/tắt xóa logo kênh gốc\n"
        "• `/title [on|off]` — Bật/tắt dịch & thay banner tiêu đề\n"
        "• `/brand [tên_kênh|off]` — Thêm watermark thương hiệu riêng\n"
        "• `/trim [auto|off|<giây>]` — Tự động cắt ảnh bìa đầu video\n"
        "• `/burn [on|off]` — Bật/tắt tự động in phụ đề mới\n"
        "• `/preset [tên_preset]` — Chọn preset vùng blur\n\n"
        "🛰️ **5. THEO DÕI KÊNH DOUYIN TỰ ĐỘNG:**\n"
        "• `/follow <id_hoặc_link> [vi|en] [style]` — Thêm kênh theo dõi\n"
        "• `/unfollow <id>` — Hủy theo dõi kênh\n"
        "• `/channels` — Danh sách các kênh đang theo dõi\n"
        "• `/monitor [on|off|scan]` — Quản lý tiến trình quét ngầm\n\n"
        "⚙️ **6. HỆ THỐNG:**\n"
        "• `/status` — Xem toàn bộ cấu hình hiện tại\n"
        "• `/help` — Xem bảng hướng dẫn này"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def cmd_lang(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /lang command — select target translation languages."""
    prefs = get_prefs(update.effective_chat.id)
    
    if not context.args:
        curr_langs = prefs.get("langs", ["vi"])
        curr_str = ", ".join(f"{LANGUAGES[l]['flag']} {LANGUAGES[l]['name']}" for l in curr_langs if l in LANGUAGES)
        lines = [
            f"🌍 **Ngôn ngữ dịch hiện tại:** {curr_str}\n",
            "**Các ngôn ngữ hỗ trợ:**",
        ]
        for code, info in LANGUAGES.items():
            lines.append(f"• `{code}` — {info['flag']} {info['name']}")
        lines.append("\n**Ví dụ đổi ngôn ngữ:**")
        lines.append("• `/lang vi` — Chỉ dịch tiếng Việt")
        lines.append("• `/lang vi en` — Dịch song ngữ Việt + Anh")
        lines.append("• `/lang vi en id` — Dịch 3 ngôn ngữ")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    valid_langs = []
    for lang in context.args:
        lang = lang.lower().strip()
        if lang in LANGUAGES and lang not in valid_langs:
            valid_langs.append(lang)
    
    if not valid_langs:
        await update.message.reply_text("❌ Không có ngôn ngữ hợp lệ. Dùng: `vi`, `en`, `id`", parse_mode="Markdown")
        return
    
    prefs["langs"] = valid_langs
    _save_user_prefs()
    flags = " ".join(LANGUAGES[l]["flag"] for l in valid_langs)
    names = ", ".join(LANGUAGES[l]["name"] for l in valid_langs)
    await update.message.reply_text(f"✅ Đã chọn ngôn ngữ dịch: {flags} **{names}**", parse_mode="Markdown")


async def cmd_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /tts command"""
    prefs = get_prefs(update.effective_chat.id)
    
    if not context.args:
        status = "🔊 BẬT" if prefs["tts"] else "🔇 TẮT"
        await update.message.reply_text(
            f"TTS hiện tại: {status}\n\nDùng `/tts on` hoặc `/tts off`",
            parse_mode="Markdown",
        )
        return
    
    val = context.args[0].lower()
    if val in ("on", "1", "bat", "bật"):
        prefs["tts"] = True
        _save_user_prefs()
        await update.message.reply_text("🔊 Đã BẬT tạo audio TTS")
    elif val in ("off", "0", "tat", "tắt"):
        prefs["tts"] = False
        _save_user_prefs()
        await update.message.reply_text("🔇 Đã TẮT tạo audio TTS")
    else:
        await update.message.reply_text("❌ Dùng `/tts on` hoặc `/tts off`", parse_mode="Markdown")


async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /voice command to select TTS engine"""
    prefs = get_prefs(update.effective_chat.id)
    
    if not context.args:
        engine = prefs.get("tts_engine", "edge")
        names = {
            "edge": "Edge-TTS (Mặc định)",
            "omnivoice": "OmniVoice (AI Voice Clone)",
            "gemini": "Gemini 3.1 Flash TTS (Biểu cảm)",
        }
        name = names.get(engine, engine)
        await update.message.reply_text(
            f"🎙️ **Engine TTS hiện tại:** `{engine}` — {name}\n\n"
            "Dùng lệnh để đổi:\n"
            "• `/voice edge` - Dùng Microsoft Edge (Nhanh/Nhẹ)\n"
            "• `/voice omnivoice` - Dùng OmniVoice (Chất lượng cao, clone giọng)\n"
            "• `/voice gemini` - Dùng Gemini TTS (Biểu cảm, điện ảnh)\n",
            parse_mode="Markdown"
        )
        return
        
    val = context.args[0].lower()
    if val in ("edge", "edge-tts"):
        prefs["tts_engine"] = "edge"
        _save_user_prefs()
        await update.message.reply_text("✅ Đã chọn **Edge-TTS** (Nhanh, nhẹ).", parse_mode="Markdown")
    elif val in ("omnivoice", "omni"):
        prefs["tts_engine"] = "omnivoice"
        _save_user_prefs()
        await update.message.reply_text("✅ Đã chọn **OmniVoice** (Chất lượng cao, voice clone).", parse_mode="Markdown")
    elif val in ("gemini", "google"):
        prefs["tts_engine"] = "gemini"
        _save_user_prefs()
        await update.message.reply_text("✅ Đã chọn **Gemini TTS** (Biểu cảm, điện ảnh).", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Engine không hợp lệ. Chọn: `edge`, `omnivoice` hoặc `gemini`.", parse_mode="Markdown")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /model command"""
    prefs = get_prefs(update.effective_chat.id)
    valid_models = ["large-v3-turbo", "large-v3", "large-v2", "large-v1"]
    
    if not context.args:
        await update.message.reply_text(
            f"🧠 Model hiện tại: `{prefs['model']}`\n\n"
            f"Các model: {', '.join(f'`{m}`' for m in valid_models)}\n\n"
            "• `large-v3-turbo` - ⚡ Khuyên dùng (Siêu nhanh 6x, chính xác cao)\n" +
            "• `large-v3` - 🧠 Large V3 (Chính xác tuyệt đối, đa dạng chất giọng)\n" +
            "• `large-v2` - 🛡️ Large V2 (Rất ổn định, chống lặp từ khi có nhạc nền)\n" +
            "• `large-v1` - 📦 Large V1 (Phiên bản tiêu chuẩn)",
            parse_mode="Markdown",
        )
        return
    
    model = context.args[0].lower()
    if model not in valid_models:
        await update.message.reply_text(f"❌ Model không hợp lệ. Chọn: {', '.join(valid_models)}")
        return
    
    prefs["model"] = model
    _save_user_prefs()
    await update.message.reply_text(f"✅ Đã chọn model: `{model}`", parse_mode="Markdown")


async def cmd_aimodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /aimodel command — select AI translation model."""
    prefs = get_prefs(update.effective_chat.id)
    curr_ai = prefs.get("ai_model", AI_DEFAULT_MODEL)
    if not context.args:
        lines = [
            f"🤖 **Model Dịch AI hiện tại:** `{curr_ai}`\n",
            "Các model khả dụng:",
        ]
        for mid, info in AI_TRANSLATE_MODELS.items():
            m_name = info.get("name", mid)
            lines.append(f"• `{mid}` — {m_name}")
        lines.append("\nDùng: `/aimodel tên_model` để đổi")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    chosen = context.args[0].strip()
    prefs["ai_model"] = chosen
    _save_user_prefs()
    await update.message.reply_text(f"✅ Đã chọn model dịch AI: `{chosen}`", parse_mode="Markdown")




async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /mode or /style command — switch between movie dialogue and driving tutorial styles"""
    prefs = get_prefs(update.effective_chat.id)
    args = context.args
    if not args:
        curr = prefs.get("translation_mode", DEFAULT_TRANSLATION_MODE)
        options = []
        for k, v in TRANSLATION_MODES.items():
            check = "✅ " if k == curr else "   "
            options.append(f"{check}`/mode {k}` : {v['icon']} *{v['name']}*\n   _{v['description']}_")
        await update.message.reply_text(
            "🎯 *Tùy chọn phong cách dịch thuật:*\n\n"
            + "\n\n".join(options)
            + "\n\n💡 Ví dụ: gõ `/mode driving` để chọn dạy lái xe, `/mode movie` để chọn dịch phim.",
            parse_mode="Markdown",
        )
        return

    chosen = args[0].lower().strip()
    if chosen not in TRANSLATION_MODES:
        valid_modes = ", ".join(f"`{k}`" for k in TRANSLATION_MODES.keys())
        await update.message.reply_text(
            f"❌ Chế độ không hợp lệ. Các chế độ hợp lệ: {valid_modes}",
            parse_mode="Markdown",
        )
        return

    prefs["translation_mode"] = chosen
    _save_user_prefs()
    info = TRANSLATION_MODES[chosen]
    await update.message.reply_text(
        f"✅ Đã chuyển phong cách dịch sang: {info['icon']} *{info['name']}*\n_{info['description']}_",
        parse_mode="Markdown",
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    prefs = get_prefs(update.effective_chat.id)
    langs_str = ", ".join(
        f"{LANGUAGES[l]['flag']} {LANGUAGES[l]['name']}" for l in prefs["langs"]
    ) or "Không có"
    tts_str = "🔊 BẬT" if prefs["tts"] else "🔇 TẮT"
    burn_str = "🎬 BẬT" if prefs.get("burn", True) else "⏸️ TẮT"
    preset_key = prefs.get("preset", "default")
    all_presets = load_presets()
    preset_name = all_presets.get(preset_key, {}).get("name", preset_key)
    
    hardsub_str = "🔍 BẬT (Gemini OCR)" if prefs.get("hardsub") else "⏸️ TẮT (Whisper)"
    gemini_key = get_gemini_api_key()
    key_str = f"✅ {gemini_key[:8]}..." if gemini_key else "❌ Chưa có"
    gemini_model = prefs.get("gemini_model", GEMINI_DEFAULT_MODEL)
    tts_engine = prefs.get("tts_engine", "edge")
    
    mode_key = prefs.get("translation_mode", DEFAULT_TRANSLATION_MODE)
    mode_info = TRANSLATION_MODES.get(mode_key, {})
    mode_str = f"{mode_info.get('icon', '🎯')} {mode_info.get('name', mode_key)}"

    await update.message.reply_text(
        f"⚙️ **Cài đặt hiện tại:**\n\n"
        f"🌍 Ngôn ngữ dịch: {langs_str}\n"
        f"🎯 Phong cách dịch: {mode_str}\n"
        f"🧠 AI Model: `{prefs['model']}`\n"
        f"🔊 TTS Audio: {tts_str} (`{tts_engine}`)\n"
        f"🎬 Auto Burn: {burn_str}\n"
        f"✂️ Cắt bìa đầu: `{'Tự động (nếu có bìa chữ Trung)' if prefs.get('trim_intro', 'auto') == 'auto' else ('TẮT' if prefs.get('trim_intro') == 'off' else str(prefs.get('trim_intro')) + 's')}`\n"
        f"🎵 Nhạc nền (BGM): `{prefs.get('bgm_mode', 'auto')}` ({int(float(prefs.get('bgm_volume', 0.8))*100)}% vol)\n"
        f"🧹 Xóa Hardsub chân: `{'BẬT' if prefs.get('clean_hardsub', True) else 'TẮT'}`\n"
        f"🏷️ Xóa Logo watermark: `{'BẬT' if prefs.get('clean_logo', False) else 'TẮT'}`\n"
        f"✨ Banner tiêu đề: `{'DỊCH' if prefs.get('translate_title') else ('XÓA' if prefs.get('clean_title') else 'TẮT')}`\n"
        f"🏷️ Watermark: `'{prefs.get('brand_name')}'`\n" 
        f"🎞️ Render Mode: `{prefs.get('vmode', 'blur')}` ({VMODES.get(prefs.get('vmode', 'blur'), 'blur')})\n"
        f"📐 Preset: `{preset_key}` ({preset_name})\n"
        f"🔍 Hardsub: {hardsub_str}\n"
        f"🔑 Gemini Key: {key_str}\n"
        f"🤖 Gemini Model: `{gemini_model}`\n"
        f"🖥️ GPU: `{DEVICE.upper()} ({COMPUTE_TYPE})`\n"
        f"🛰️ Giám sát Douyin: `{'🟢 BẬT (' + str(len([c for c in get_channels() if c.get('enabled')])) + ' kênh)' if is_monitor_running() else '⚪ TẮT'}` (dùng `/channels`)",
        parse_mode="Markdown",
    )


VMODES = {
    "clean": "🧹 1. Chỉ xóa hardsub cũ (Video sạch)",
    "inpaint_burn": "🌟 Xóa cũ + In sub mới (Inpaint AI)",
    "blur": "⚡ Che mờ + In sub mới (Dynamic Blur)",
    "pure_burn": "📝 2. Chỉ burn sub mới (In trực tiếp)",
}

async def cmd_vmode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /vmode command — select video processing / inpainting mode"""
    prefs = get_prefs(update.effective_chat.id)
    if not context.args:
        curr = prefs.get("vmode", "blur")
        lines = [f"🎬 *Chế độ xử lý video hiện tại:* `{curr}`\n"]
        for k, name in VMODES.items():
            check = "✅ " if k == curr else "   "
            lines.append(f"{check}`/vmode {k}` — {name}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    val = context.args[0].lower().strip()
    if val in VMODES:
        prefs["vmode"] = val
        _save_user_prefs()
        await update.message.reply_text(f"✅ Đã đổi chế độ xử lý video sang: *{VMODES[val]}*", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Chế độ không hợp lệ. Chọn: `clean`, `inpaint_burn`, `blur`, hoặc `pure_burn`", parse_mode="Markdown")


async def cmd_clean(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clean command — modular clean options (sub, logo, title, all)"""
    prefs = get_prefs(update.effective_chat.id)
    arg = context.args[0].lower().strip() if context.args else "all"

    if arg in ("sub", "hardsub"):
        prefs["vmode"] = "clean"
        prefs["clean_hardsub"] = True
        prefs["clean_logo"] = False
        prefs["clean_title"] = False
        _save_user_prefs()
        await update.message.reply_text("🧹 Đã chọn: *Chỉ xóa hardsub dưới chân video* (Giữ nguyên logo & banner trên)", parse_mode="Markdown")
    elif arg in ("logo", "watermark"):
        prefs["vmode"] = "clean"
        prefs["clean_hardsub"] = False
        prefs["clean_logo"] = True
        prefs["clean_title"] = False
        _save_user_prefs()
        await update.message.reply_text("🏷️ Đã chọn: *Chỉ xóa logo & watermark góc*", parse_mode="Markdown")
    elif arg in ("all", "full"):
        prefs["vmode"] = "clean"
        prefs["clean_hardsub"] = True
        prefs["clean_logo"] = True
        prefs["clean_title"] = True
        _save_user_prefs()
        await update.message.reply_text("✨ Đã chọn: *Xóa toàn diện* (Xóa sạch hardsub + logo + banner tiêu đề trên)", parse_mode="Markdown")
    elif arg in ("off", "tat", "tắt"):
        prefs["vmode"] = "inpaint_burn"
        _save_user_prefs()
        await update.message.reply_text("🔄 Đã tắt chế độ Clean Plate (Quay lại chế độ in sub mới)", parse_mode="Markdown")
    else:
        msg = (
            "🧹 *Tùy chọn xóa chữ & logo (Clean Plate):*\n\n"
            "• `/clean sub` — Chỉ xóa hardsub dưới chân\n"
            "• `/clean logo` — Chỉ xóa logo & watermark góc\n"
            "• `/clean all` — Xóa sạch toàn bộ (hardsub + logo + tiêu đề trên)\n"
            "• `/clean off` — Tắt clean plate (quay lại in sub mới)"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_logo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /logo command — toggle logo/watermark removal"""
    prefs = get_prefs(update.effective_chat.id)
    if not context.args:
        curr = "BẬT" if prefs.get("clean_logo", False) else "TẮT"
        await update.message.reply_text(f"🏷️ Xóa logo kênh gốc hiện tại: *{curr}*\nDùng `/logo on` hoặc `/logo off`", parse_mode="Markdown")
        return
    val = context.args[0].lower().strip()
    if val in ("on", "1", "bat", "bật"):
        prefs["clean_logo"] = True
        _save_user_prefs()
        await update.message.reply_text("🏷️ Đã BẬT xóa logo & watermark kênh gốc")
    elif val in ("off", "0", "tat", "tắt"):
        prefs["clean_logo"] = False
        _save_user_prefs()
        await update.message.reply_text("⏸️ Đã TẮT xóa logo kênh gốc")


async def cmd_title(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /title command — top title banner handling"""
    prefs = get_prefs(update.effective_chat.id)
    if not context.args:
        msg = (
            "✨ *Cài đặt banner tiêu đề trên cùng:*\n\n"
            "• `/title trans` — Dịch và in đè banner tiêu đề mới\n"
            "• `/title clean` — Chỉ xóa banner tiêu đề (không in mới)\n"
            "• `/title off` — Giữ nguyên banner gốc"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return
    val = context.args[0].lower().strip()
    if val in ("trans", "translate", "dich", "dịch"):
        prefs["translate_title"] = True
        prefs["clean_title"] = True
        _save_user_prefs()
        await update.message.reply_text("✨ Đã BẬT dịch & thay thế banner tiêu đề trên cùng")
    elif val in ("clean", "xoa", "xóa"):
        prefs["translate_title"] = False
        prefs["clean_title"] = True
        _save_user_prefs()
        await update.message.reply_text("🧹 Đã BẬT chỉ xóa banner tiêu đề trên cùng")
    elif val in ("off", "0", "tat", "tắt"):
        prefs["translate_title"] = False
        prefs["clean_title"] = False
        _save_user_prefs()
        await update.message.reply_text("⏸️ Đã TẮT xử lý banner tiêu đề trên cùng")


async def cmd_brand(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /brand or /watermark command — set channel watermark"""
    prefs = get_prefs(update.effective_chat.id)
    if not context.args:
        curr = prefs.get("brand_name", "") or "Chưa đặt"
        await update.message.reply_text(f"🏷️ Tên kênh / Watermark hiện tại: `{curr}`\nDùng `/brand <tên_kênh>` hoặc `/brand off`", parse_mode="Markdown")
        return
    val = " ".join(context.args).strip()
    if val.lower() in ("off", "none", "0"):
        prefs["brand_name"] = ""
        _save_user_prefs()
        await update.message.reply_text("🗑️ Đã xóa tên kênh / watermark")
    else:
        prefs["brand_name"] = val[:50]
        _save_user_prefs()
        await update.message.reply_text(f"🏷️ Đã đặt tên kênh / watermark thành: *{val[:50]}*", parse_mode="Markdown")


async def cmd_burn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /burn command — toggle auto burn on/off"""
    prefs = get_prefs(update.effective_chat.id)
    
    if not context.args:
        status = "🎬 BẬT" if prefs.get("burn", True) else "⏸️ TẮT"
        await update.message.reply_text(
            f"Auto burn sub hiện tại: {status}\n\nDùng `/burn on` hoặc `/burn off`",
            parse_mode="Markdown",
        )
        return
    
    val = context.args[0].lower()
    if val in ("on", "1", "bat", "bật"):
        prefs["burn"] = True
        _save_user_prefs()
        await update.message.reply_text("🎬 Đã BẬT auto burn sub\n(Video sẽ tự động được burn sub sau khi xử lý)")
    elif val in ("off", "0", "tat", "tắt"):
        prefs["burn"] = False
        _save_user_prefs()
        await update.message.reply_text("⏸️ Đã TẮT auto burn sub")
    else:
        await update.message.reply_text("❌ Dùng `/burn on` hoặc `/burn off`", parse_mode="Markdown")

async def cmd_trim(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /trim command - toggle auto intro cover trimming or set seconds"""
    prefs = get_prefs(update.effective_chat.id)

    if not context.args:
        status = prefs.get("trim_intro", "auto")
        status_str = "Tự động (nếu có bìa chữ Trung)" if status == "auto" else ("TẮT" if status == "off" else f"Cắt {status}s")
        msg = (
            f"✂️ *Cắt bìa đầu video hiện tại:* `{status}` ({status_str})\n\n"
            "Dùng:\n"
            "• `/trim auto` hoặc `/trim on` — Tự động phát hiện & cắt bìa chữ Trung\n"
            "• `/trim off` — Tắt cắt bìa đầu\n"
            "• `/trim 1.5` — Cắt cố định 1.5 giây đầu"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    val = context.args[0].lower()
    if val in ("auto", "on", "1", "bat", "bật"):
        prefs["trim_intro"] = "auto"
        _save_user_prefs()
        await update.message.reply_text("✂️ Đã BẬT tự động phát hiện & cắt bìa tiếng Trung đầu video")
    elif val in ("off", "0", "tat", "tắt"):
        prefs["trim_intro"] = "off"
        _save_user_prefs()
        await update.message.reply_text("⏸️ Đã TẮT cắt bìa đầu video")
    else:
        try:
            sec = float(val)
            if 0 < sec <= 10:
                prefs["trim_intro"] = str(round(sec, 2))
                _save_user_prefs()
                await update.message.reply_text(f"✂️ Đã cài đặt cắt cố định *{sec:.2f}s* đầu video", parse_mode="Markdown")
            else:
                await update.message.reply_text("❌ Số giây cắt phải từ 0.1 đến 10 giây", parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text("❌ Dùng: `/trim auto`, `/trim off`, hoặc `/trim <số_giây>`", parse_mode="Markdown")


async def cmd_bgm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /bgm command — configure background music preservation & volume"""
    prefs = get_prefs(update.effective_chat.id)

    if not context.args:
        mode = prefs.get("bgm_mode", "auto")
        vol = int(float(prefs.get("bgm_volume", 0.8)) * 100)
        mode_label = "⚡ Giảm âm tự động (Ducking)" if mode in ("auto", "duck") else ("🌟 Tách nhạc nền AI (Demucs)" if mode == "ai" else "🔇 TẮT (Chỉ giọng đọc)")
        lines = [
            f"🎵 *Cài đặt nhạc nền (BGM):* `{mode}` ({mode_label})",
            f"🔊 *Âm lượng BGM:* `{vol}%`",
            "",
            "Dùng:",
            "• `/bgm auto` hoặc `/bgm duck` — Bật giảm âm tự động (FFmpeg Ducking - Nhanh)",
            "• `/bgm ai` — Bật tách nhạc nền AI (Demucs - Tách sạch giọng Hán)",
            "• `/bgm off` — Tắt nhạc nền (chỉ lấy giọng đọc TTS)",
            "• `/bgm 70` — Đặt âm lượng nhạc nền 70%",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    val = context.args[0].lower().replace("%", "")
    if val in ("auto", "duck", "on", "1", "bat", "bật"):
        prefs["bgm_mode"] = "auto"
        _save_user_prefs()
        await update.message.reply_text("🎵 Đã BẬT giữ nhạc nền BGM (chế độ Ducking tự động giảm âm khi nói)")
    elif val in ("ai", "demucs", "sota"):
        prefs["bgm_mode"] = "ai"
        _save_user_prefs()
        await update.message.reply_text("🌟 Đã BẬT tách nhạc nền AI (Demucs — tách sạch tiếng Trung)")
    elif val in ("off", "none", "0", "tat", "tắt", "mute"):
        prefs["bgm_mode"] = "none"
        _save_user_prefs()
        await update.message.reply_text("🔇 Đã TẮT giữ nhạc nền (chỉ phát giọng đọc TTS mới)")
    else:
        try:
            num = float(val)
            vol = num / 100.0 if num > 1.0 else num
            if 0.05 <= vol <= 1.5:
                prefs["bgm_volume"] = round(vol, 2)
                _save_user_prefs()
                await update.message.reply_text(f"🔊 Đã đặt âm lượng nhạc nền BGM thành *{int(vol*100)}%*", parse_mode="Markdown")
            else:
                await update.message.reply_text("❌ Âm lượng phải từ 5% đến 150%", parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text("❌ Dùng: `/bgm auto`, `/bgm ai`, `/bgm off`, hoặc `/bgm <5-100>`", parse_mode="Markdown")


async def cmd_preset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /preset command — view and select presets"""
    prefs = get_prefs(update.effective_chat.id)
    all_presets = load_presets()
    current = prefs.get("preset", "default")
    
    if not context.args:
        lines = ["📐 **Danh sách preset:**\n"]
        for key, p in all_presets.items():
            mark = "✅" if key == current else "⬜"
            sr = p.get("sub_region", {})
            extras = len(p.get("extra_regions", []))
            lines.append(
                f"{mark} `{key}` — {p['name']}\n"
                f"   Sub: Y{sr.get('y_ratio',0):.0%} H{sr.get('h_ratio',0):.0%}"
                f"{f' + {extras} extra' if extras else ''}"
            )
        lines.append(f"\nDùng `/preset tên` để chọn preset")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return
    
    key = context.args[0].lower().strip()
    if key not in all_presets:
        await update.message.reply_text(
            f"❌ Preset `{key}` không tồn tại.\nDùng `/preset` để xem danh sách.",
            parse_mode="Markdown"
        )
        return
    
    prefs["preset"] = key
    _save_user_prefs()
    p = all_presets[key]
    extras = len(p.get("extra_regions", []))
    await update.message.reply_text(
        f"✅ Đã chọn preset: `{key}` — {p['name']}"
        f"\n{'📐 ' + str(1 + extras) + ' vùng blur' if extras else '📐 1 vùng blur'}",
        parse_mode="Markdown"
    )


# ── Hardsub & Gemini Commands ────────────────────────────────────────────

async def cmd_hardsub(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /hardsub command — toggle hardsub OCR mode"""
    prefs = get_prefs(update.effective_chat.id)

    if not context.args:
        status = "🔍 BẬT (Gemini OCR)" if prefs["hardsub"] else "⏸️ TẮT (Whisper)"
        await update.message.reply_text(
            f"Chế độ Hardsub: {status}\n\n"
            "🔍 `/hardsub on` — Bật trích sub cứng bằng Gemini AI\n"
            "⏸️ `/hardsub off` — Quay lại Whisper (nhận dạng giọng nói)\n\n"
            "💡 *Hardsub* dùng cho video có phụ đề cứng burn vào hình ảnh,\n"
            "không cần âm thanh. Cần có Gemini API key (`/gemini`).",
            parse_mode="Markdown",
        )
        return

    val = context.args[0].lower()
    if val in ("on", "1", "bat", "bật"):
        key = get_gemini_api_key()
        if not key:
            await update.message.reply_text(
                "❌ Chưa có Gemini API key!\n"
                "Dùng `/gemini key AIza...` để cài key trước.",
                parse_mode="Markdown",
            )
            return
        prefs["hardsub"] = True
        _save_user_prefs()
        model = prefs.get("gemini_model", GEMINI_DEFAULT_MODEL)
        await update.message.reply_text(
            f"🔍 Đã BẬT chế độ Hardsub (Gemini OCR)\n"
            f"🤖 Model: `{model}`\n\n"
            "Gửi video → Gemini sẽ đọc sub cứng trên hình ảnh.",
            parse_mode="Markdown",
        )
    elif val in ("off", "0", "tat", "tắt"):
        prefs["hardsub"] = False
        _save_user_prefs()
        await update.message.reply_text(
            "⏸️ Đã TẮT Hardsub → quay lại Whisper (nhận dạng giọng nói)"
        )
    else:
        await update.message.reply_text(
            "❌ Dùng `/hardsub on` hoặc `/hardsub off`",
            parse_mode="Markdown",
        )


async def cmd_gemini(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /gemini command — manage Gemini API key and model"""
    prefs = get_prefs(update.effective_chat.id)

    if not context.args:
        key = get_gemini_api_key()
        key_str = f"✅ `{key[:8]}...`" if key else "❌ Chưa có"
        model = prefs.get("gemini_model", GEMINI_DEFAULT_MODEL)
        model_info = GEMINI_MODELS.get(model, {})
        model_name = model_info.get("name", model)

        models_list = "\n".join(
            f"  {'✅' if k == model else '⬜'} `{k}` — {v['name']}"
            for k, v in GEMINI_MODELS.items()
        )

        await update.message.reply_text(
            f"🔑 **Gemini API Config**\n\n"
            f"API Key: {key_str}\n"
            f"Model: `{model}` ({model_name})\n\n"
            f"**Models:**\n{models_list}\n\n"
            "**Lệnh:**\n"
            "`/gemini key AIza...` — Lưu API key\n"
            "`/gemini model tên` — Chọn model",
            parse_mode="Markdown",
        )
        return

    subcmd = context.args[0].lower()

    if subcmd == "key":
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Thiếu key. Dùng: `/gemini key AIzaSy...`",
                parse_mode="Markdown",
            )
            return

        api_key = context.args[1].strip()
        # Verify key
        status_msg = await update.message.reply_text("🔄 Đang xác thực API key...")
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            models = list(client.models.list())
            if not models:
                await status_msg.edit_text("❌ API key không hợp lệ.")
                return
        except Exception as e:
            err = str(e).lower()
            if "api_key" in err or "invalid" in err or "permission" in err:
                await status_msg.edit_text("❌ API key không hợp lệ. Kiểm tra lại.")
                return
            # Network error — save anyway
            pass

        set_gemini_api_key(api_key)
        await status_msg.edit_text(
            f"✅ Đã lưu API key: `{api_key[:8]}...`\n"
            "Giờ bạn có thể dùng `/hardsub on`",
            parse_mode="Markdown",
        )

    elif subcmd == "model":
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Thiếu tên model. Dùng: `/gemini model gemini-2.5-flash`",
                parse_mode="Markdown",
            )
            return

        model_name = context.args[1].strip()
        if model_name not in GEMINI_MODELS:
            valid = ", ".join(f"`{k}`" for k in GEMINI_MODELS)
            await update.message.reply_text(
                f"❌ Model không hợp lệ.\nCác model: {valid}",
                parse_mode="Markdown",
            )
            return

        prefs["gemini_model"] = model_name
        _save_user_prefs()
        info = GEMINI_MODELS[model_name]
        await update.message.reply_text(
            f"✅ Đã chọn: `{model_name}`\n{info['name']}",
            parse_mode="Markdown",
        )

    else:
        await update.message.reply_text(
            "❌ Dùng `/gemini key ...` hoặc `/gemini model ...`",
            parse_mode="Markdown",
        )


async def cmd_omni(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /omni command to generate standalone TTS audio"""
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text(
            "❌ Thiếu nội dung.\n\n"
            "Dùng: `/omni [Nội dung muốn đọc]`\n"
            "Ví dụ: `/omni Xin chào các bạn`",
            parse_mode="Markdown"
        )
        return

    status_msg = await update.message.reply_text("🎤 Đang khởi tạo OmniVoice AI...")
    
    import json
    import subprocess
    import uuid
    from pathlib import Path
    
    # Using the same fixed reference audio as the main project
    OMNIVOICE_PYTHON = r"D:\Naldo\omnivoice-env\Scripts\python.exe"
    OMNIVOICE_WORKER = str(Path("services/omnivoice_worker.py").resolve())
    DEFAULT_CLONE_AUDIO = r"D:\Naldo\flowkit\output\_shared\tts_templates\adam_ref.wav"
    DEFAULT_CLONE_TEXT = "In high-pressure corporate settings, true strength isn't measured by volume, but by psychological leverage. When faced with blame, avoid the explanation trap, the reactive urge to justify yourself only reinforces a position of weakness."
    
    job_id = uuid.uuid4().hex[:8]
    from config import UPLOAD_FOLDER
    temp_dir = UPLOAD_FOLDER / f"omni_{job_id}"
    temp_dir.mkdir(exist_ok=True)
    
    input_data = {
        "segments": [text],
        "lang": "vi",
        "ref_audio": DEFAULT_CLONE_AUDIO,
        "ref_text": DEFAULT_CLONE_TEXT,
        "speed": 1.1,
        "out_dir": str(temp_dir)
    }
    
    input_json_path = temp_dir / "omnivoice_input.json"
    with open(input_json_path, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, ensure_ascii=False)
        
    await status_msg.edit_text("🎤 Đang đọc văn bản bằng giọng Clone (OmniVoice)...\nCó thể mất vài chục giây.")
    
    def run_omni():
        try:
            subprocess.run(
                [OMNIVOICE_PYTHON, OMNIVOICE_WORKER, str(input_json_path)],
                capture_output=True, text=True, timeout=300
            )
            return True
        except Exception as e:
            return str(e)
            
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_omni)
    
    if result is not True:
        await status_msg.edit_text(f"❌ Lỗi OmniVoice: {result}")
        return
        
    result_json = temp_dir / "result.json"
    audio_file = temp_dir / "seg_0000.wav"
    
    if not result_json.exists() or not audio_file.exists():
        await status_msg.edit_text("❌ Không tạo được audio (OmniVoice crash hoặc hết VRAM).")
        return
        
    await status_msg.edit_text("📤 Đang gửi file audio...")
    
    import os
    with open(audio_file, "rb") as f:
        await context.bot.send_audio(
            update.effective_chat.id,
            audio=f,
            title="OmniVoice Audio",
            filename=f"omnivoice_{job_id}.wav",
            caption=f"🎙️ **OmniVoice Clone**\n📝 `{text[:100]}{'...' if len(text) > 100 else ''}`",
            parse_mode="Markdown"
        )
        
    await status_msg.delete()
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(str(temp_dir))
    except:
        pass


# ── Processing Handlers ──────────────────────────────────────────────────

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle video/audio file uploads"""
    msg = update.message
    prefs = get_prefs(msg.chat_id)
    
    # Determine file type
    if msg.video:
        file_obj = msg.video
        file_name = msg.video.file_name or "video.mp4"
        file_size = msg.video.file_size
        emoji = "🎬"
    elif msg.audio:
        file_obj = msg.audio
        file_name = msg.audio.file_name or "audio.mp3"
        file_size = msg.audio.file_size
        emoji = "🎵"
    elif msg.voice:
        file_obj = msg.voice
        file_name = "voice.ogg"
        file_size = msg.voice.file_size
        emoji = "🎤"
    elif msg.document:
        file_obj = msg.document
        file_name = msg.document.file_name or "file"
        file_size = msg.document.file_size
        emoji = "📎"
        # Check if it's a media file
        ext = Path(file_name).suffix.lower()
        if ext not in ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac'):
            await msg.reply_text("❌ File không phải video/audio. Vui lòng gửi file video hoặc audio.")
            return
    else:
        return
    
    size_mb = (file_size or 0) / 1048576
    status_msg = await msg.reply_text(
        f"{emoji} Đã nhận file: *{file_name}* ({size_mb:.1f}MB)\n"
        f"⏳ Đang tải file...",
        parse_mode="Markdown",
    )
    
    # Download file from Telegram
    job_id = uuid.uuid4().hex[:8]
    safe_name = safe_stem(file_name, fallback="upload") + Path(file_name).suffix.lower()
    local_path = str(UPLOAD_FOLDER / job_id / safe_name)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        tg_file = await file_obj.get_file()
        await tg_file.download_to_drive(custom_path=local_path)
    except Exception as e:
        await status_msg.edit_text(f"❌ Lỗi tải file: {e}")
        return
    
    # Process in background
    await _process_and_reply(update, context, status_msg, job_id, local_path, prefs, file_name)


async def handle_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle URL messages - supports batch (multiple URLs, one per line)"""
    msg = update.message
    text = msg.text.strip()
    prefs = get_prefs(msg.chat_id)
    
    # Extract all URLs from message
    import re
    urls = re.findall(r'https?://\S+', text)
    urls = list(dict.fromkeys(urls))  # Remove duplicates, preserve order
    
    if not urls:
        await msg.reply_text("❌ Không tìm thấy URL hợp lệ")
        return
    
    total = len(urls)
    
    if total == 1:
        # Single URL - original flow
        await _process_single_url(update, context, urls[0], prefs)
    else:
        # Batch mode
        await msg.reply_text(
            f"📋 *Batch Processing*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Đã nhận *{total}* URLs, xử lý lần lượt...\n",
            parse_mode="Markdown",
        )
        
        success = 0
        failed = 0
        for i, url in enumerate(urls):
            await context.bot.send_message(
                msg.chat_id,
                f"📌 *[{i+1}/{total}]* Đang xử lý...\n`{url[:60]}...`",
                parse_mode="Markdown",
            )
            try:
                await _process_single_url(update, context, url, prefs)
                success += 1
            except Exception as e:
                failed += 1
                await context.bot.send_message(
                    msg.chat_id, f"❌ [{i+1}/{total}] Lỗi: {e}"
                )
        
        # Batch summary
        await context.bot.send_message(
            msg.chat_id,
            f"📋 *Batch hoàn tất!*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ Thành công: {success}/{total}\n"
            f"{'❌ Thất bại: ' + str(failed) if failed else '🎉 Tất cả thành công!'}",
            parse_mode="Markdown",
        )


async def _process_single_url(update, context, url, prefs):
    """Process a single URL (used by both single and batch mode)"""
    chat_id = update.effective_chat.id
    
    status_msg = await context.bot.send_message(
        chat_id,
        f"🔗 {url[:50]}...\n⏳ Đang tải video..."
    )
    
    job_id = uuid.uuid4().hex[:8]
    create_job(
        job_id,
        message="Đang tải video từ URL...",
        original_name=url[:120],
        trim_intro=prefs.get("trim_intro", "auto"),
    )
    
    # Download video from URL
    try:
        video_path = await asyncio.get_event_loop().run_in_executor(
            None, download_from_url, job_id, url
        )
    except Exception as e:
        await status_msg.edit_text(f"❌ Lỗi tải video: {e}")
        raise
    
    original_name = jobs[job_id].get("original_name", "video")
    await status_msg.edit_text(
        f"✅ Đã tải: *{original_name[:40]}*\n"
        f"🎤 Đang xử lý...",
        parse_mode="Markdown",
    )
    
    await _process_and_reply(update, context, status_msg, job_id, video_path, prefs, original_name)


async def _process_and_reply(update, context, status_msg, job_id, file_path, prefs, original_name):
    """Process video/audio and send results back"""
    chat_id = update.effective_chat.id
    model_size = prefs["model"]
    translate_langs = prefs["langs"]
    
    # Initialize job
    if job_id not in jobs:
        create_job(job_id, message="Khởi tạo...", trim_intro=prefs.get("trim_intro", "auto"))
    jobs[job_id]["original_name"] = original_name
    jobs[job_id]["trim_intro"] = prefs.get("trim_intro", "auto")
    
    # Run processing in background thread with progress updates
    loop = asyncio.get_event_loop()
    use_hardsub = prefs.get("hardsub", False)

    # Start processing in executor
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    if use_hardsub:
        # Hardsub mode: use Gemini OCR
        gemini_key = get_gemini_api_key()
        if not gemini_key:
            await status_msg.edit_text(
                "❌ Chưa có Gemini API key! Dùng /gemini key AIza... để cài."
            )
            return

        await status_msg.edit_text("🔍 Đang trích hardsub bằng Gemini AI...")
        job = jobs[job_id]
        job["video_path"] = file_path
        job["gemini_api_key"] = gemini_key
        job["gemini_model"] = prefs.get("gemini_model", GEMINI_DEFAULT_MODEL)
        job["translate_langs"] = translate_langs
        job["translate_method"] = "ai"
        job["translation_mode"] = prefs.get("translation_mode", DEFAULT_TRANSLATION_MODE)
        job["mode"] = "hardsub"
        future = executor.submit(hardsub_worker, job)
    else:
        # Normal mode: Whisper speech recognition
        await status_msg.edit_text("🎤 Đang trích xuất và nhận dạng giọng nói...")
        jobs[job_id]["ai_model"] = prefs.get("ai_model", AI_DEFAULT_MODEL)
        jobs[job_id]["translation_mode"] = prefs.get("translation_mode", DEFAULT_TRANSLATION_MODE)
        future = executor.submit(process_video, job_id, file_path, model_size, translate_langs, "ai", prefs.get("translation_mode", DEFAULT_TRANSLATION_MODE))
    
    # Progress update loop: edit message every 10s with current status
    TIMEOUT_SECONDS = 1800  # 30 minutes max
    start_time = time.time()
    last_msg = ""
    
    while not future.done():
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_SECONDS:
            jobs[job_id]["cancel"] = True
            await status_msg.edit_text("⏰ Timeout: Quá 30 phút, đã tự động hủy.")
            executor.shutdown(wait=False)
            return
        
        # Update progress message every 10s
        job = jobs.get(job_id, {})
        current_msg = job.get("message", "")
        progress = job.get("progress", 0)
        
        if current_msg and current_msg != last_msg:
            try:
                status_text = f"⏳ [{progress}%] {current_msg}"
                await status_msg.edit_text(status_text)
                last_msg = current_msg
            except Exception:
                pass  # Ignore edit errors (message not modified, etc.)
        
        await asyncio.sleep(8)
    
    # Check for exceptions
    try:
        future.result()  # Raises if processing failed
    except Exception as e:
        await status_msg.edit_text(f"❌ Lỗi xử lý: {e}")
        return
    
    job = jobs[job_id]
    
    if job["status"] == "error":
        await status_msg.edit_text(f"❌ {job['message']}")
        return
    
    # ── Send results ──
    total_time = job.get("total_time", 0)
    segment_count = job.get("segment_count", 0)
    duration = job.get("duration", 0)
    
    if use_hardsub:
        gemini_model = prefs.get("gemini_model", GEMINI_DEFAULT_MODEL)
        result_text = (
            f"✅ *Hoàn thành!*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔍 Hardsub OCR | 📊 {segment_count} câu | 🤖 {gemini_model}\n"
        )
    else:
        dur_min = int(duration // 60)
        dur_sec = int(duration % 60)
        result_text = (
            f"✅ *Hoàn thành trong {total_time:.0f}s!*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 {segment_count} câu | ⏱ {dur_min}:{dur_sec:02d} | 🧠 {model_size}\n"
        )
    await status_msg.edit_text(result_text, parse_mode="Markdown")
    
    # Send SRT files
    srt_files = job.get("srt_files", {})
    for lang, info in srt_files.items():
        if "error" in info:
            await context.bot.send_message(
                chat_id, f"⚠️ Lỗi dịch {info.get('flag', '')} {info.get('lang_name', lang)}: {info['error']}"
            )
            continue
        
        srt_path = info.get("path", "")
        if srt_path and os.path.exists(srt_path):
            with open(srt_path, "rb") as f:
                await context.bot.send_document(
                    chat_id,
                    document=f,
                    filename=info["filename"],
                    caption=f"{info.get('flag', '📝')} {info.get('lang_name', lang)}",
                )
    
    # ── Send video file if available ──
    video_info = job.get("video_file", {})
    video_path = video_info.get("path", "")
    if video_path and os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) / 1048576
        if size_mb <= 50:  # Telegram limit 50MB
            await context.bot.send_message(chat_id, f"🎬 Đang gửi video gốc ({size_mb:.1f}MB)...")
            try:
                with open(video_path, "rb") as f:
                    await context.bot.send_document(
                        chat_id,
                        document=f,
                        filename=video_info.get("filename", "video.mp4"),
                        caption=f"🎬 Video gốc - chất lượng đầy đủ ({size_mb:.1f}MB)",
                    )
            except Exception as e:
                await context.bot.send_message(chat_id, f"⚠️ Không gửi được video: {e}")
        else:
            await context.bot.send_message(chat_id, f"⚠️ Video quá lớn ({size_mb:.1f}MB > 50MB), không gửi qua Telegram được")
    
    # ── TTS if enabled ──
    if prefs["tts"]:
        for lang, info in srt_files.items():
            if "error" in info or lang == "zh":
                continue
            
            srt_path = info.get("path", "")
            if not srt_path or not os.path.exists(srt_path):
                continue
            
            lang_name = info.get("lang_name", lang)
            await context.bot.send_message(chat_id, f"🔊 Đang tạo audio TTS {info.get('flag', '')} {lang_name}...")
            
            try:
                with open(srt_path, "r", encoding="utf-8") as f:
                    srt_content = f.read()
                
                await loop.run_in_executor(
                    None, generate_tts_audio, job_id, lang, srt_content, prefs.get("tts_engine", "edge")
                )
                
                tts_key = f"tts_{lang}"
                tts_info = job.get(tts_key, {})
                
                if tts_info.get("status") == "done":
                    audio_path = tts_info["path"]
                    size_mb = tts_info.get("size", 0) / 1048576
                    duration_s = tts_info.get("duration", 0)
                    
                    with open(audio_path, "rb") as f:
                        await context.bot.send_audio(
                            chat_id,
                            audio=f,
                            title=f"TTS {lang_name}",
                            filename=tts_info.get("filename", f"tts_{lang}.mp3"),
                            caption=f"🔊 {info.get('flag', '')} {lang_name} ({size_mb:.1f}MB, {duration_s:.0f}s)",
                        )
                else:
                    await context.bot.send_message(
                        chat_id, f"⚠️ TTS {lang_name} thất bại: {tts_info.get('message', 'Unknown error')}"
                    )
            except Exception as e:
                await context.bot.send_message(chat_id, f"⚠️ TTS {lang_name} lỗi: {e}")

    # ── Auto Burn Sub if enabled ──
    if prefs.get("burn", True) and video_path and os.path.exists(video_path):
        preset_key = prefs.get("preset", "default")
        all_presets = load_presets()
        preset_data = all_presets.get(preset_key, all_presets.get("default"))

        if preset_data:
            for lang, info in srt_files.items():
                if "error" in info or lang == "zh":
                    continue

                srt_path = info.get("path", "")
                if not srt_path or not os.path.exists(srt_path):
                    continue

                lang_name = info.get("lang_name", lang)
                await context.bot.send_message(
                    chat_id,
                    f"🎬 Đang burn sub {info.get('flag', '')} {lang_name} (preset: {preset_data['name']})..."
                )

                try:
                    with open(srt_path, "r", encoding="utf-8") as f:
                        srt_content = f.read()

                    sub_region = preset_data.get("sub_region")
                    extra_regions = preset_data.get("extra_regions", [])

                    from functools import partial
                    burn_func = partial(
                        burn_sub_video,
                        job_id, lang, srt_content, sub_region, extra_regions,
                        render_mode=prefs.get("vmode", "blur"),
                        trim_intro=prefs.get("trim_intro", "auto"),
                        translate_title=prefs.get("translate_title", False),
                        title_lang=lang,
                        brand_name=prefs.get("brand_name", ""),
                        bgm_mode=prefs.get("bgm_mode", "auto"),
                        bgm_volume=float(prefs.get("bgm_volume", 0.8)),
                        clean_hardsub=prefs.get("clean_hardsub", True),
                        clean_logo=prefs.get("clean_logo", False),
                        clean_title=prefs.get("clean_title", False),
                        burn_new_sub=(prefs.get("vmode") != "clean"),
                    )
                    await loop.run_in_executor(None, burn_func)

                    burn_key = f"burn_{lang}"
                    burn_info = job.get(burn_key, {})

                    if burn_info.get("status") == "done":
                        burn_path = burn_info["path"]
                        size_mb = os.path.getsize(burn_path) / 1048576

                        if size_mb <= 50:
                            with open(burn_path, "rb") as f:
                                await context.bot.send_document(
                                    chat_id,
                                    document=f,
                                    filename=burn_info.get("filename", f"video_{lang}_sub.mp4"),
                                    caption=f"🎬 {info.get('flag', '')} {lang_name} sub ({size_mb:.1f}MB)",
                                )
                        else:
                            await context.bot.send_message(
                                chat_id, f"⚠️ Video burn quá lớn ({size_mb:.1f}MB > 50MB)"
                            )
                    else:
                        await context.bot.send_message(
                            chat_id,
                            f"⚠️ Burn {lang_name} thất bại: {burn_info.get('message', 'Unknown')}"
                        )
                except Exception as e:
                    await context.bot.send_message(chat_id, f"⚠️ Burn {lang_name} lỗi: {e}")


def is_url(text: str) -> bool:
    """Check if text is a URL"""
    return any(text.strip().startswith(p) for p in (
        "http://", "https://", "www.",
    ))


# ── Main ──────────────────────────────────────────────────────────────────


# ── Douyin Monitor Command Handlers ──────────────────────────────────────────

async def cmd_follow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /follow command to track a Douyin channel."""
    chat_id = update.effective_chat.id
    set_notify_chat_id(chat_id)

    if not context.args:
        await update.message.reply_text(
            "➕ **Cú pháp theo dõi kênh Douyin tự động:**\n\n"
            "`/follow <douyin_id_hoặc_link> [ngôn_ngữ: vi|en] [style: driving|movie]`\n\n"
            "**Ví dụ:**\n"
            "• `/follow Binbinbin9993` (Mặc định dịch Tiếng Việt, phong cách Lái xe)\n"
            "• `/follow Binbinbin9993 vi driving`\n"
            "• `/follow https://www.douyin.com/user/MS4wLjABAAAA... en movie`\n\n"
            "💡 *Khi kênh đăng video mới, bot sẽ tự động tải bản gốc Master, cắt bìa chữ Hán, dịch AI, tách nhạc nền Demucs, in sub mới và gửi thành phẩm về đây!*",
            parse_mode="Markdown",
        )
        return

    channel_input = context.args[0].strip()
    target_lang = "vi"
    style = "driving"

    if len(context.args) > 1:
        arg1 = context.args[1].lower()
        if arg1 in ["vi", "en", "id", "zh", "ja", "ko", "th"]:
            target_lang = arg1
        elif arg1 in ["driving", "movie", "literal", "fun"]:
            style = arg1

    if len(context.args) > 2:
        arg2 = context.args[2].lower()
        if arg2 in ["driving", "movie", "literal", "fun"]:
            style = arg2
        elif arg2 in ["vi", "en", "id", "zh", "ja", "ko", "th"]:
            target_lang = arg2

    msg = await update.message.reply_text(
        f"🔍 Đang tìm kiếm và phân giải thông tin kênh `{channel_input}`...",
        parse_mode="Markdown",
    )

    try:
        sec_uid, nickname, meta = resolve_channel_sec_uid(channel_input)
        if not sec_uid:
            await msg.edit_text(
                f"❌ Không tìm thấy kênh Douyin với ID: `{channel_input}`.\n"
                "Vui lòng kiểm tra lại Douyin ID hoặc dán trực tiếp link chia sẻ kênh!",
                parse_mode="Markdown",
            )
            return

        cid = meta.get("unique_id") or channel_input
        ch = add_channel(
            channel_id=cid,
            nickname=nickname or cid,
            sec_uid=sec_uid,
            target_lang=target_lang,
            style=style,
            bgm_mode="ai",
            auto_burn=True,
        )

        if not is_monitor_running():
            start_monitor(interval=3600)

        followers = meta.get("follower_count")
        follower_str = f"{followers:,}" if followers else "N/A"

        await msg.edit_text(
            f"✅ **Đã thêm kênh theo dõi thành công!**\n\n"
            f"👤 **Kênh:** *{nickname}*\n"
            f"🆔 **Douyin ID:** `{cid}`\n"
            f"👥 **Người theo dõi:** {follower_str}\n"
            f"🌐 **Dịch sang:** `{target_lang.upper()}` (Phong cách: *{style}*)\n"
            f"🎵 **Nhạc nền BGM:** Tách giọng AI (Demucs)\n"
            f"🎬 **Chế độ:** Tự động cắt bìa & in sub mới\n"
            f"🛰️ **Giám sát:** 🟢 Đang chạy (Quét mỗi 3 phút)\n\n"
            f"💡 Dùng `/channels` để xem danh sách hoặc `/unfollow {cid}` để hủy theo dõi.",
            parse_mode="Markdown",
        )

    except Exception as e:
        await msg.edit_text(
            f"❌ Lỗi khi thêm kênh: {e}",
            parse_mode="Markdown",
        )


async def cmd_unfollow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /unfollow command."""
    chat_id = update.effective_chat.id
    set_notify_chat_id(chat_id)

    if not context.args:
        await update.message.reply_text(
            "➖ **Cú pháp bỏ theo dõi:** `/unfollow <douyin_id>`\n"
            "Xem danh sách kênh đang theo dõi bằng `/channels`",
            parse_mode="Markdown",
        )
        return

    channel_id = context.args[0].strip()
    success = remove_channel(channel_id)
    if success:
        await update.message.reply_text(
            f"✅ Đã hủy theo dõi kênh `{channel_id}` thành công!",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"❌ Không tìm thấy kênh `{channel_id}` trong danh sách theo dõi.",
            parse_mode="Markdown",
        )


async def cmd_channels(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all monitored channels."""
    chat_id = update.effective_chat.id
    set_notify_chat_id(chat_id)

    channels = get_channels()
    if not channels:
        await update.message.reply_text(
            "📺 **Hiện chưa có kênh Douyin nào được theo dõi.**\n\n"
            "Thêm kênh mới bằng lệnh:\n"
            "`/follow <douyin_id> [vi|en] [style]`",
            parse_mode="Markdown",
        )
        return

    history = get_downloaded_history()
    status = get_monitor_status()
    daemon_icon = "🟢" if status.get("running") else "⚪"

    lines = [
        f"📺 **Danh sách kênh Douyin đang theo dõi ({len(channels)} kênh):**\n"
        f"🛰️ Tiến trình giám sát: {daemon_icon} *{'Đang chạy' if status.get('running') else 'Tạm dừng'}*\n"
        f"📦 Tổng video đã xử lý: *{len(history)} video*\n"
    ]

    for i, ch in enumerate(channels, 1):
        cid = ch.get("channel_id", "")
        nick = ch.get("nickname", cid)
        lang = ch.get("target_lang", "vi").upper()
        style = ch.get("style", "driving")
        en = "🟢 Bật" if ch.get("enabled", True) else "⚪ Tắt"
        last_check = ch.get("last_check", 0)
        last_check_str = time.strftime("%H:%M %d/%m", time.localtime(last_check)) if last_check else "Chưa quét"

        lines.append(
            f"**{i}. {nick}** (`{cid}`) [{en}]\n"
            f"   └ Ngôn ngữ: `{lang}` ({style}) | Lần quét: {last_check_str}"
        )

    lines.append(
        "\n💡 *Lệnh quản lý:*\n"
        "• `/follow <id>`: Thêm kênh\n"
        "• `/unfollow <id>`: Xóa kênh\n"
        "• `/monitor scan`: Quét tìm video mới ngay"
    )

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_monitor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manage monitor daemon (on/off/scan/status)."""
    chat_id = update.effective_chat.id
    set_notify_chat_id(chat_id)

    arg = context.args[0].lower() if context.args else "status"

    if arg in ["on", "start"]:
        start_monitor(interval=3600)
        await update.message.reply_text(
            "🟢 **Đã khởi động tiến trình giám sát kênh Douyin tự động!**\n"
            "Hệ thống sẽ định kỳ 1 tiếng kiểm tra video mới và gửi thành phẩm về Telegram.",
            parse_mode="Markdown",
        )
    elif arg in ["off", "stop"]:
        stop_monitor()
        await update.message.reply_text(
            "⚪ **Đã tạm dừng tiến trình giám sát kênh Douyin tự động.**",
            parse_mode="Markdown",
        )
    elif arg in ["scan", "now"]:
        scan_now()
        await update.message.reply_text(
            "⚡ **Đã kích hoạt quét kiểm tra video mới ngay lập tức!**\n"
            "Video mới (nếu có) sẽ được tự động tải, dịch và gửi về đây.",
            parse_mode="Markdown",
        )
    else:
        status = get_monitor_status()
        is_run = status.get("running")
        last_time = status.get("last_scan_time") or "Chưa có"
        last_stat = status.get("last_scan_status", "idle")
        total = status.get("total_scans", 0)
        task = status.get("current_task") or "Đang chờ chu kỳ tiếp theo"
        interval = status.get("interval", 180)
        history_len = len(get_downloaded_history())
        channels = get_channels()
        active_count = len([c for c in channels if c.get("enabled", True)])

        stat_text = (
            "🛰️ **TRẠNG THÁI GIÁM SÁT KÊNH DOUYIN**\n\n"
            f"• **Trạng thái:** {'🟢 ĐANG CHẠY' if is_run else '⚪ ĐÃ TẠM DỪNG'}\n"
            f"• **Chu kỳ quét:** `{interval}s` (Mỗi 1 tiếng)\n"
            f"• **Số kênh theo dõi:** {active_count}/{len(channels)} kênh đang bật\n"
            f"• **Tổng lượt đã quét:** {total} lượt\n"
            f"• **Lần quét gần nhất:** {last_time} ({last_stat})\n"
            f"• **Công việc hiện tại:** {task}\n"
            f"• **Lịch sử video đã xử lý:** {history_len} video\n\n"
            "**Các lệnh điều khiển:**\n"
            "• `/monitor on` - Bật giám sát ngầm\n"
            "• `/monitor off` - Tắt giám sát\n"
            "• `/monitor scan` - Quét ngay lập tức\n"
            "• `/channels` - Quản lý danh sách kênh"
        )
        await update.message.reply_text(stat_text, parse_mode="Markdown")

def main():
    """Start the Telegram bot"""
    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN chưa được set!")
        print("   export TELEGRAM_BOT_TOKEN='your_token'")
        sys.exit(1)
    
    print("=" * 60)
    print("  🤖 Chinese SRT Extractor - Telegram Bot")
    print("=" * 60)
    print(f"  🖥️  GPU: {DEVICE.upper()} | Compute: {COMPUTE_TYPE}")
    print(f"  🌍 Ngôn ngữ: {', '.join(DEFAULT_LANGS)}")
    print(f"  🔊 TTS: {'BẬT' if DEFAULT_TTS else 'TẮT'}")
    print("=" * 60)
    
    start_cleanup_worker()
    try:
        if not is_monitor_running():
            start_monitor(interval=3600)
            print("  🛰️ Douyin Monitor Daemon: Đã bật tự động quét ngầm (3 phút/lần)")
    except Exception as _e:
        print(f"  [DouyinMonitor] Notice: {_e}")
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("lang", cmd_lang))
    app.add_handler(CommandHandler("tts", cmd_tts))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(CommandHandler("burn", cmd_burn))
    app.add_handler(CommandHandler("trim", cmd_trim))
    app.add_handler(CommandHandler("bgm", cmd_bgm))
    app.add_handler(CommandHandler("vmode", cmd_vmode))
    app.add_handler(CommandHandler("clean", cmd_clean))
    app.add_handler(CommandHandler("logo", cmd_logo))
    app.add_handler(CommandHandler("title", cmd_title))
    app.add_handler(CommandHandler("brand", cmd_brand))
    app.add_handler(CommandHandler("watermark", cmd_brand))
    app.add_handler(CommandHandler("preset", cmd_preset))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("aimodel", cmd_aimodel))
    app.add_handler(CommandHandler("mode", cmd_mode))
    app.add_handler(CommandHandler("style", cmd_mode))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("hardsub", cmd_hardsub))
    app.add_handler(CommandHandler("gemini", cmd_gemini))
    app.add_handler(CommandHandler("omni", cmd_omni))
    app.add_handler(CommandHandler("follow", cmd_follow))
    app.add_handler(CommandHandler("unfollow", cmd_unfollow))
    app.add_handler(CommandHandler("channels", cmd_channels))
    app.add_handler(CommandHandler("monitor", cmd_monitor))
    
    # Media handlers
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.AUDIO | filters.VOICE | filters.Document.ALL,
        handle_video,
    ))
    
    # URL handler
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & filters.Regex(r'https?://'),
        handle_url,
    ))
    
    # Bot commands menu
    async def post_init(application):
        await application.bot.set_my_commands([
            BotCommand("start", "🏠 Bắt đầu"),
            BotCommand("help", "📖 Hướng dẫn"),
            BotCommand("lang", "🌍 Chọn ngôn ngữ dịch"),
            BotCommand("tts", "🔊 Bật/tắt TTS audio"),
            BotCommand("voice", "🎙️ Chọn engine TTS"),
            BotCommand("burn", "🎬 Bật/tắt auto burn sub"),
            BotCommand("clean", "🧹 Tùy chọn xóa chữ & logo"),
            BotCommand("logo", "🏷️ Bật/tắt xóa logo kênh gốc"),
            BotCommand("title", "✨ Xử lý banner tiêu đề trên"),
            BotCommand("brand", "🏷️ Đặt watermark / tên kênh"),
            BotCommand("trim", "✂️ Cắt bìa tiếng Trung đầu video"),
            BotCommand("bgm", "🎵 Cài đặt nhạc nền BGM lồng tiếng"),
            BotCommand("preset", "📐 Chọn preset vùng blur"),
            BotCommand("model", "🧠 Chọn AI model"),
            BotCommand("aimodel", "🤖 Chọn model dịch AI"),
            BotCommand("hardsub", "🔍 Bật/tắt trích hardsub"),
            BotCommand("gemini", "🔑 Cài đặt Gemini API"),
            BotCommand("omni", "🎤 Đọc văn bản bằng OmniVoice"),
            BotCommand("follow", "➕ Theo dõi kênh Douyin tự động"),
            BotCommand("unfollow", "➖ Bỏ theo dõi kênh Douyin"),
            BotCommand("channels", "📺 Danh sách kênh theo dõi"),
            BotCommand("monitor", "🛰️ Bật/tắt/quét tự động ngầm"),
            BotCommand("status", "⚙️ Xem cài đặt"),
        ])
        me = await application.bot.get_me()
        print(f"  🤖 Bot: @{me.username}")
        print(f"  💬 Gửi /start để bắt đầu")
        print("=" * 60)
    
    app.post_init = post_init
    
    print("  ⏳ Đang kết nối Telegram...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()



