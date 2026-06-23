"""
London Session Trading Bot - Monitors 5 pairs for strategy signals
Sends Telegram alerts when conditions are met:
1. 4H candle high/low swiped (1-2pm GMT)
2. 5M structure shift (break of recent high/low)
3. RSI crosses 50 midline

Runs a tiny Flask web server alongside the bot for Render deployment.
"""
import os
import json
import logging
import threading
from flask import Flask
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.constants import ParseMode

from analysis import analyze_pair, PAIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHECK_INTERVAL_SECONDS = 30  # check every 30 seconds
STATE_FILE = "state.json"
PORT = int(os.environ.get("PORT", 10000))

# --- Tiny web server for Render ---
flask_app = Flask(__name__)

@flask_app.route("/")
def health():
    return "London Session Trading Bot is running."

def run_web_server():
    flask_app.run(host="0.0.0.0", port=PORT)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"chat_ids": [], "alerts_sent": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def format_signal_message(symbol, result):
    direction = result["direction"]
    price = result["price"]
    level_high = result["level_high"]
    level_low = result["level_low"]
    emoji = "🟢" if direction == "BUY" else "🔴"
    
    # Show which level was swiped
    swipe_info = ""
    if direction == "BUY":
        swipe_info = "4H LOW swiped → Reversal BUY from support"
    else:
        swipe_info = "4H HIGH swiped → Reversal SELL from resistance"
    
    return (
        f"{emoji} *{symbol} — {direction} SIGNAL* {emoji}\n\n"
        f"💰 Price: {price:.4f}\n"
        f"📐 4H High: {level_high:.4f}\n"
        f"📐 4H Low: {level_low:.4f}\n\n"
        f"📌 *Strategy:* {swipe_info}\n\n"
        f"✅ *Conditions Met:*\n"
        f"  ✓ 4H level swiped\n"
        f"  ✓ 5M structure shift ({direction})\n"
        f"  ✓ RSI crossed 50 ({direction})\n\n"
        f"_Reasoning:_\n" + "\n".join(f"• {r}" for r in result["reasons"]) +
        f"\n\n⚠️ Not financial advice. Confirm before entering."
    )

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    chat_id = update.effective_chat.id
    if chat_id not in state["chat_ids"]:
        state["chat_ids"].append(chat_id)
        save_state(state)
    
    pairs_list = "\n  • ".join(PAIRS.keys())
    await update.message.reply_text(
        f"🚀 *London Session Trading Bot Active*\n\n"
        f"📊 *Monitoring:*\n  • {pairs_list}\n\n"
        f"⏰ *London Session:* 07:00 - 15:00 GMT\n"
        f"📌 *Level Formation:* 13:00 - 14:00 GMT\n\n"
        f"*Strategy Rules:*\n"
        f"  1. 4H level must be swiped\n"
        f"  2. 5M structure shift (break recent high/low)\n"
        f"  3. RSI crosses 50 midline\n\n"
        f"📌 *Trade Logic:*\n"
        f"  • LOW swiped → BUY (reversal from support)\n"
        f"  • HIGH swiped → SELL (reversal from resistance)\n\n"
        f"You'll get alerts when ALL 3 conditions align.\n\n"
        f"Send /signal for an on-demand check, or /stop to turn off alerts.",
        parse_mode=ParseMode.MARKDOWN
    )

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    chat_id = update.effective_chat.id
    if chat_id in state["chat_ids"]:
        state["chat_ids"].remove(chat_id)
        save_state(state)
    await update.message.reply_text("Automatic alerts turned off. Send /start to re-enable.")

async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Analyzing all pairs...")
    
    results = []
    for symbol in PAIRS.keys():
        try:
            result = analyze_pair(symbol)
            if result["direction"] != "NEUTRAL":
                results.append((symbol, result))
        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}")
    
    if not results:
        await update.message.reply_text(
            "📊 *No Signals Detected*\n\n"
            "All pairs are currently neutral.\n"
            "Waiting for conditions to align:\n"
            "  • 4H level swiped\n"
            "  • 5M structure shift\n"
            "  • RSI 50 cross",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    for symbol, result in results:
        msg = format_signal_message(symbol, result)
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def check_for_new_signal(context: ContextTypes.DEFAULT_TYPE):
    """Runs on schedule. Pushes message when a new signal appears."""
    state = load_state()
    if not state["chat_ids"]:
        return
    
    # Initialize alerts_sent if not exists
    if "alerts_sent" not in state:
        state["alerts_sent"] = {}
    
    for symbol in PAIRS.keys():
        try:
            result = analyze_pair(symbol)
            direction = result["direction"]
            
            if direction in ("BUY", "SELL"):
                # Check if alert already sent for this signal
                alert_key = f"{symbol}_{result['signal_id']}"
                if alert_key not in state["alerts_sent"]:
                    msg = format_signal_message(symbol, result)
                    for chat_id in state["chat_ids"]:
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=msg,
                                parse_mode=ParseMode.MARKDOWN
                            )
                            logger.info(f"Alert sent: {symbol} - {direction}")
                        except Exception as e:
                            logger.error(f"Failed to send to {chat_id}: {e}")
                    
                    # Mark alert as sent
                    state["alerts_sent"][alert_key] = True
                    save_state(state)
                    
        except Exception as e:
            logger.error(f"Background check failed for {symbol}: {e}")

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable not set")
    
    # Start web server
    threading.Thread(target=run_web_server, daemon=True).start()
    
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("signal", signal))
    
    # Schedule background checks
    app.job_queue.run_repeating(
        check_for_new_signal, interval=CHECK_INTERVAL_SECONDS, first=10
    )
    
    logger.info("🚀 London Session Bot starting...")
    logger.info(f"📊 Monitoring: {', '.join(PAIRS.keys())}")
    logger.info(f"⏰ Check interval: {CHECK_INTERVAL_SECONDS} seconds")
    app.run_polling()

if __name__ == "__main__":
    main()
