"""
Telegram bot: watches XAUUSD 15M chart in the background and
PUSHES you a message automatically when a new Buy/Sell signal appears.
Also supports /signal for an on-demand check.
"""
import os
import json
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.constants import ParseMode

from analysis import analyze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHECK_INTERVAL_SECONDS = 15 * 60  # check every 15 minutes, matches candle close
STATE_FILE = "state.json"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"chat_ids": [], "last_direction": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def format_signal_message(result):
    direction = result["direction"]
    price = result["price"]
    emoji = "🟢" if direction == "BUY" else "🔴"
    return (
        f"{emoji} *XAUUSD 15M — {direction} SIGNAL*\n\n"
        f"Entry: {price:.2f}\n"
        f"SL: {result['sl']:.2f}\n"
        f"TP: {result['tp']:.2f}\n"
        f"R:R: 1:2 | ATR: {result['atr']:.2f}\n\n"
        f"_Reasoning:_\n" + "\n".join(f"• {r}" for r in result["reasons"]) +
        f"\n\n⚠️ Not financial advice. Confirm before entering."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    chat_id = update.effective_chat.id
    if chat_id not in state["chat_ids"]:
        state["chat_ids"].append(chat_id)
        save_state(state)
    await update.message.reply_text(
        "Gold (XAUUSD) Signal Bot active.\n\n"
        "You'll automatically get a message here whenever a new Buy/Sell "
        "signal forms (checked every 15 minutes).\n\n"
        "You can also send /signal anytime for an on-demand check, or "
        "/stop to turn off automatic alerts."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    state = load_state()
    chat_id = update.effective_chat.id
    if chat_id in state["chat_ids"]:
        state["chat_ids"].remove(chat_id)
        save_state(state)
    await update.message.reply_text("Automatic alerts turned off. Send /start to re-enable.")


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Analyzing XAUUSD 15M...")
    try:
        result = analyze()
    except Exception as e:
        logger.exception("Analysis failed")
        await update.message.reply_text(f"Error fetching/analyzing data: {e}")
        return

    if result["direction"] == "NEUTRAL":
        msg = (
            f"📊 *XAUUSD 15M — NO CLEAR SIGNAL*\n\n"
            f"Price: {result['price']:.2f}\n"
            f"Indicators are mixed — sit out this one.\n\n"
            f"_Reasoning:_\n" + "\n".join(f"• {r}" for r in result["reasons"])
        )
    else:
        msg = format_signal_message(result)

    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def check_for_new_signal(context: ContextTypes.DEFAULT_TYPE):
    """Runs on a schedule. Pushes a message only when direction CHANGES
    to BUY or SELL (i.e. a fresh signal), not on every check."""
    state = load_state()
    if not state["chat_ids"]:
        return  # nobody subscribed yet

    try:
        result = analyze()
    except Exception as e:
        logger.error(f"Background check failed: {e}")
        return

    direction = result["direction"]
    last_direction = state.get("last_direction")

    # Only alert on a genuine new actionable signal (not NEUTRAL,
    # and not the same direction repeated from last check)
    if direction in ("BUY", "SELL") and direction != last_direction:
        msg = format_signal_message(result)
        for chat_id in state["chat_ids"]:
            try:
                await context.bot.send_message(
                    chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN
                )
            except Exception as e:
                logger.error(f"Failed to send to {chat_id}: {e}")

    state["last_direction"] = direction
    save_state(state)


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable not set")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("signal", signal))

    # Schedule background checks
    app.job_queue.run_repeating(
        check_for_new_signal, interval=CHECK_INTERVAL_SECONDS, first=10
    )

    logger.info("Bot starting with background watcher...")
    app.run_polling()


if __name__ == "__main__":
    main()
