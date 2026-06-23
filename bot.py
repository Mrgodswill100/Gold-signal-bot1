"""
Chima Dtrader AI — London Sweep Bot

Strategy:
1. 4H candle HIGH/LOW formed 1AM–2AM WAT  → Asia range
2. London session (8AM–12PM WAT) sweeps that HIGH or LOW
3. 1M structure break (CHoCH) or Inverse FVG after sweep
4. RSI(14) crosses 50 midline on 1M
5. HIGH swept → SELL | LOW swept → BUY

Runs Flask web server so Render free tier stays alive.
Pin with UptimeRobot on your Render URL.
"""

import os
import json
import logging
import threading
from datetime import datetime
import pytz

from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.constants import ParseMode

from analysis import analyze_pair, analyze_all, get_asia_range, PAIRS

# ─── CONFIG ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN")
CHECK_INTERVAL       = 60        # scan every 60 seconds
STATE_FILE           = "state.json"
PORT                 = int(os.environ.get("PORT", 10000))
WAT                  = pytz.timezone("Africa/Lagos")

# ─── FLASK (keeps Render free tier alive) ────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route("/")
def health():
    return "Chima Dtrader AI — London Sweep Bot is running."

def run_web_server():
    flask_app.run(host="0.0.0.0", port=PORT)

# ─── STATE ───────────────────────────────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"chat_ids": [], "fired_today": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# ─── MESSAGE FORMATTER ───────────────────────────────────────────────────────
def format_signal(result: dict) -> str:
    direction  = result["direction"]
    emoji      = "🟢 BUY" if direction == "BUY" else "🔴 SELL"
    sweep_icon = "⬇️" if direction == "BUY" else "⬆️"
    reasons    = "\n".join(f"  • {r}" for r in result["reasons"])

    return (
        f"🚨 *SIGNAL ALERT*\n\n"
        f"📌 *Pair:* `{result['symbol']}`\n"
        f"📍 *Direction:* {emoji}\n\n"
        f"✅ *Conditions Met:*\n"
        f"  {sweep_icon} Asia {result.get('sweep','').replace('_',' ')} @ `{result['price']:.4f}`\n"
        f"  🔀 1M Structure: `{result.get('structure','').upper()}`\n"
        f"  📊 RSI crossed `{result.get('rsi_cross','')}`  across 50\n\n"
        f"🏔 *Asia Range:*\n"
        f"  H: `{result['asia_high']}` | L: `{result['asia_low']}`\n\n"
        f"_Reasons:_\n{reasons}\n\n"
        f"⚠️ Not financial advice. Confirm before entering."
    )

def format_neutral(pair: str, result: dict) -> str:
    reasons = "\n".join(f"  • {r}" for r in result["reasons"])
    return (
        f"📊 *{pair} — NO SIGNAL*\n\n"
        f"Price: `{result['price']:.4f}`\n\n"
        f"_Status:_\n{reasons}"
    )

# ─── COMMANDS ────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state   = load_state()
    chat_id = update.effective_chat.id

    if chat_id not in state["chat_ids"]:
        state["chat_ids"].append(chat_id)
        save_state(state)

    pairs = "\n".join([f"  • {p}" for p in PAIRS])
    now_wat = datetime.now(WAT).strftime("%H:%M WAT")

    await update.message.reply_text(
        f"🚀 *Chima Dtrader AI — London Sweep Bot*\n\n"
        f"📊 *Monitoring:*\n{pairs}\n\n"
        f"📋 *Strategy Rules:*\n"
        f"  1️⃣ 4H candle 1AM–2AM WAT → marks HIGH & LOW\n"
        f"  2️⃣ London sweep (8AM–12PM WAT)\n"
        f"  3️⃣ 1M CHoCH or Inverse FVG\n"
        f"  4️⃣ RSI(14) crosses 50 on 1M\n\n"
        f"🎯 *Trade Logic:*\n"
        f"  HIGH swept → 🔴 SELL\n"
        f"  LOW swept  → 🟢 BUY\n\n"
        f"🕐 Current time: {now_wat}\n\n"
        f"You'll get alerts when ALL conditions align.\n"
        f"One signal per pair per day.\n\n"
        f"/signal — on-demand scan\n"
        f"/range  — today's Asia HIGH/LOW\n"
        f"/stop   — disable alerts",
        parse_mode=ParseMode.MARKDOWN
    )


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state   = load_state()
    chat_id = update.effective_chat.id
    if chat_id in state["chat_ids"]:
        state["chat_ids"].remove(chat_id)
        save_state(state)
    await update.message.reply_text("🔕 Alerts disabled. Send /start to re-enable.")


async def cmd_range(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show today's Asia range for all pairs."""
    lines = ["📐 *Today's Asia Ranges (1AM–2AM WAT):*\n"]
    found = False

    for display_name in PAIRS:
        level = get_asia_range(display_name)
        if level:
            found = True
            lines.append(
                f"*{display_name}*\n"
                f"  🔼 High: `{level['high']}`\n"
                f"  🔽 Low:  `{level['low']}`\n"
            )

    if not found:
        await update.message.reply_text(
            "⏳ No Asia range marked yet today.\n"
            "The 4H candle at 1AM–2AM WAT has not formed yet.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """On-demand scan of all pairs."""
    await update.message.reply_text("🔍 Scanning all pairs...")

    results = analyze_all()
    signals_found = False

    for pair, result in results.items():
        if result["direction"] in ("BUY", "SELL"):
            result["symbol"] = pair
            await update.message.reply_text(
                format_signal(result),
                parse_mode=ParseMode.MARKDOWN
            )
            signals_found = True

    if not signals_found:
        now_wat  = datetime.now(WAT)
        now_time = now_wat.time()

        from datetime import time
        london_start = time(8, 0)
        london_end   = time(12, 0)

        if not (london_start <= now_time < london_end):
            msg = (
                f"⏰ Outside London window.\n"
                f"Signals only fire 8AM–12PM WAT.\n"
                f"Current time: {now_wat.strftime('%H:%M WAT')}"
            )
        else:
            msg = "❌ No valid signals right now.\nConditions not fully aligned yet."

        await update.message.reply_text(msg)


# ─── BACKGROUND SCANNER ──────────────────────────────────────────────────────
async def background_scan(context: ContextTypes.DEFAULT_TYPE):
    """Runs every 60s. Fires signal only once per pair per day."""
    state    = load_state()
    chat_ids = state.get("chat_ids", [])

    if not chat_ids:
        return

    today = datetime.now(WAT).strftime("%Y-%m-%d")

    # Reset fired_today on new day
    fired_today = state.get("fired_today", {})
    if fired_today.get("date") != today:
        fired_today = {"date": today}
        state["fired_today"] = fired_today
        save_state(state)

    results = analyze_all()

    for pair, result in results.items():
        if result["direction"] not in ("BUY", "SELL"):
            continue

        # One signal per pair per day
        if fired_today.get(pair):
            continue

        result["symbol"] = pair
        msg = format_signal(result)

        for chat_id in chat_ids:
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                logger.error(f"Send error to {chat_id}: {e}")

        fired_today[pair] = True
        state["fired_today"] = fired_today
        save_state(state)
        logger.info(f"Signal fired: {pair} {result['direction']}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable not set")

    # Flask keeps Render free tier alive
    threading.Thread(target=run_web_server, daemon=True).start()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("stop",   cmd_stop))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("range",  cmd_range))

    # Scan every 60 seconds
    app.job_queue.run_repeating(background_scan, interval=CHECK_INTERVAL, first=10)

    logger.info("Chima Dtrader AI bot starting...")
    app.run_polling()


if __name__ == "__main__":
    main()
