import logging
import asyncio
from datetime import datetime, time
import pytz
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import twelvedata as td

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TWELVEDATA_KEY  = "YOUR_TWELVEDATA_API_KEY"

SYMBOLS = ["EUR/USD", "GBP/USD", "US500", "US30", "NAS100"]

WAT = pytz.timezone("Africa/Lagos")      # WAT = UTC+1
GMT = pytz.timezone("UTC")

# Strategy time windows (WAT)
RANGE_START = time(1, 0)   # 1:00 AM WAT  → mark 4H candle HIGH/LOW
RANGE_END   = time(2, 0)   # 2:00 AM WAT

LONDON_START = time(8, 0)  # 8:00 AM WAT  → sweep must happen here
LONDON_END   = time(12, 0) # 12:00 PM WAT → no signals after this

RSI_PERIOD   = 14
RSI_MIDLINE  = 50

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── STATE ────────────────────────────────────────────────────────────────────
asia_ranges: dict[str, dict] = {}   # { symbol: { high, low, date } }
alert_subscribers: set[int] = set()
monitoring_active: bool = False

# ─── TWELVEDATA HELPERS ───────────────────────────────────────────────────────

def fetch_candles(symbol: str, interval: str, count: int = 100) -> pd.DataFrame | None:
    """Fetch OHLCV candles from Twelve Data."""
    try:
        ts = td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=count,
            apikey=TWELVEDATA_KEY,
            timezone="UTC"
        )
        df = ts.as_pandas()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        return df
    except Exception as e:
        logger.error(f"fetch_candles error [{symbol}]: {e}")
        return None


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_structure_break(df: pd.DataFrame) -> str | None:
    """
    Detect 1M structure break (CHoCH) or Inverse FVG on the last few candles.
    Returns 'bullish', 'bearish', or None.
    """
    if len(df) < 5:
        return None

    highs = df["high"].values
    lows  = df["low"].values
    closes = df["close"].values

    # ── Structure Break (CHoCH): break of recent swing high/low ──
    recent_high = max(highs[-6:-1])
    recent_low  = min(lows[-6:-1])
    last_close  = closes[-1]

    if last_close > recent_high:
        return "bullish"   # broke above → BUY structure shift
    if last_close < recent_low:
        return "bearish"   # broke below → SELL structure shift

    # ── Inverse FVG: gap between candle[-3] and candle[-1] ──
    # Bullish IFVG: candle[-3].low > candle[-1].high (gap above filled from below)
    # Bearish IFVG: candle[-3].high < candle[-1].low (gap below filled from above)
    c3_low  = lows[-3]
    c1_high = highs[-1]
    c3_high = highs[-3]
    c1_low  = lows[-1]

    if last_close > c3_low and c3_low > c1_high:
        return "bullish"
    if last_close < c3_high and c3_high < c1_low:
        return "bearish"

    return None


# ─── CORE STRATEGY LOGIC ─────────────────────────────────────────────────────

def mark_asia_range(symbol: str) -> dict | None:
    """
    Grab the 4H candle that falls within 1AM–2AM WAT and store its HIGH/LOW.
    """
    df_4h = fetch_candles(symbol, "4h", count=10)
    if df_4h is None:
        return None

    today_wat = datetime.now(WAT).date()

    for ts, row in df_4h.iterrows():
        candle_wat = ts.astimezone(WAT)
        candle_time = candle_wat.time()
        candle_date = candle_wat.date()

        if candle_date == today_wat and RANGE_START <= candle_time < RANGE_END:
            level = {
                "high": float(row["high"]),
                "low":  float(row["low"]),
                "date": today_wat,
            }
            asia_ranges[symbol] = level
            logger.info(f"[{symbol}] Asia range marked → H:{level['high']} L:{level['low']}")
            return level

    return None


def check_signal(symbol: str) -> dict | None:
    """
    Full strategy check for one symbol:
    1. Asia range must be marked (1AM–2AM WAT 4H candle)
    2. London session sweep (8AM–12PM WAT)
    3. 1M structure break (CHoCH) or Inverse FVG
    4. RSI crosses 50 on 1M
    Returns signal dict or None.
    """
    now_wat = datetime.now(WAT)
    now_time = now_wat.time()

    # ── Guard: must be within London window ──
    if not (LONDON_START <= now_time < LONDON_END):
        return None

    # ── Get or refresh Asia range ──
    level = asia_ranges.get(symbol)
    if level is None or level["date"] != now_wat.date():
        level = mark_asia_range(symbol)
    if level is None:
        return None

    asia_high = level["high"]
    asia_low  = level["low"]

    # ── Fetch 1M candles ──
    df_1m = fetch_candles(symbol, "1min", count=60)
    if df_1m is None or len(df_1m) < 20:
        return None

    current_price = float(df_1m["close"].iloc[-1])

    # ── Check sweep ──
    high_swept = current_price > asia_high
    low_swept  = current_price < asia_low

    if not high_swept and not low_swept:
        return None   # no sweep yet

    sweep_direction = "HIGH" if high_swept else "LOW"
    signal_direction = "SELL" if high_swept else "BUY"

    # ── Structure break or Inverse FVG on 1M ──
    structure = detect_structure_break(df_1m)

    # Validate structure aligns with signal direction
    expected_structure = "bearish" if signal_direction == "SELL" else "bullish"
    if structure != expected_structure:
        return None

    # ── RSI cross 50 on 1M ──
    rsi = compute_rsi(df_1m["close"])
    rsi_now  = rsi.iloc[-1]
    rsi_prev = rsi.iloc[-2]

    if signal_direction == "BUY"  and not (rsi_prev < RSI_MIDLINE <= rsi_now):
        return None
    if signal_direction == "SELL" and not (rsi_prev > RSI_MIDLINE >= rsi_now):
        return None

    return {
        "symbol":    symbol,
        "direction": signal_direction,
        "sweep":     sweep_direction,
        "structure": structure,
        "rsi":       round(rsi_now, 2),
        "price":     current_price,
        "asia_high": asia_high,
        "asia_low":  asia_low,
        "time":      now_wat.strftime("%H:%M WAT"),
    }


def format_signal(s: dict) -> str:
    emoji = "🔴 SELL" if s["direction"] == "SELL" else "🟢 BUY"
    sweep_emoji = "⬆️" if s["sweep"] == "HIGH" else "⬇️"

    return (
        f"🚨 *SIGNAL ALERT*\n\n"
        f"📌 *Pair:* `{s['symbol']}`\n"
        f"📍 *Direction:* {emoji}\n\n"
        f"✅ *Conditions Met:*\n"
        f"  {sweep_emoji} Asia {s['sweep']} swept @ `{s['price']}`\n"
        f"  🔀 1M Structure: `{s['structure'].upper()}`\n"
        f"  📊 RSI(14): `{s['rsi']}` crossed 50\n\n"
        f"🏔 *Asia Range:* H:`{s['asia_high']}` | L:`{s['asia_low']}`\n"
        f"🕐 *Time:* {s['time']}"
    )


# ─── BACKGROUND MONITOR ───────────────────────────────────────────────────────

async def monitor_loop(app: Application):
    global monitoring_active
    monitoring_active = True
    fired_today: set[str] = set()
    last_reset_date = datetime.now(WAT).date()

    logger.info("Monitor loop started.")

    while monitoring_active:
        now_wat = datetime.now(WAT)

        # Reset fired signals daily
        if now_wat.date() != last_reset_date:
            fired_today.clear()
            asia_ranges.clear()
            last_reset_date = now_wat.date()
            logger.info("Daily reset done.")

        # Auto-mark Asia range at 1AM WAT
        now_time = now_wat.time()
        if RANGE_START <= now_time < RANGE_END:
            for symbol in SYMBOLS:
                if symbol not in asia_ranges:
                    mark_asia_range(symbol)

        # Check signals during London window
        if LONDON_START <= now_time < LONDON_END and alert_subscribers:
            for symbol in SYMBOLS:
                key = f"{symbol}_{now_wat.date()}"
                if key in fired_today:
                    continue  # one signal per symbol per day

                signal = check_signal(symbol)
                if signal:
                    msg = format_signal(signal)
                    for chat_id in alert_subscribers:
                        try:
                            await app.bot.send_message(
                                chat_id=chat_id,
                                text=msg,
                                parse_mode="Markdown"
                            )
                        except Exception as e:
                            logger.error(f"Send error to {chat_id}: {e}")
                    fired_today.add(key)
                    logger.info(f"Signal fired: {symbol} {signal['direction']}")

        await asyncio.sleep(60)  # check every 60 seconds


# ─── TELEGRAM COMMANDS ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    alert_subscribers.add(chat_id)

    now_wat = datetime.now(WAT)
    now_time = now_wat.time()

    if LONDON_START <= now_time < LONDON_END:
        window = "🟢 London window ACTIVE — monitoring live"
    elif now_time < RANGE_START or now_time >= LONDON_END:
        window = "⏳ Waiting for next session"
    else:
        window = "🔵 Asia range marking window (1AM–2AM WAT)"

    pairs = "\n".join([f"  • {s}" for s in SYMBOLS])

    msg = (
        f"🚀 *Chima Dtrader AI — London Sweep Bot*\n\n"
        f"📊 *Monitoring:*\n{pairs}\n\n"
        f"📋 *Strategy Rules:*\n"
        f"  1️⃣ 4H candle 1AM–2AM WAT → marks HIGH & LOW\n"
        f"  2️⃣ London sweep of that HIGH/LOW (8AM–12PM WAT)\n"
        f"  3️⃣ 1M structure break (CHoCH) or Inverse FVG\n"
        f"  4️⃣ RSI(14) crosses 50 midline on 1M\n\n"
        f"🎯 *Trade Logic:*\n"
        f"  • HIGH swept → 🔴 SELL\n"
        f"  • LOW swept → 🟢 BUY\n\n"
        f"📡 *Status:* {window}\n\n"
        f"Send /signal for on-demand check\n"
        f"Send /range to see today's Asia range\n"
        f"Send /stop to disable alerts"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    alert_subscribers.discard(chat_id)
    await update.message.reply_text("🔕 Alerts disabled. Send /start to re-enable.")


async def cmd_range(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not asia_ranges:
        await update.message.reply_text(
            "⏳ No Asia range marked yet today.\n"
            "Range is captured from the 4H candle between 1AM–2AM WAT."
        )
        return

    lines = ["📐 *Today's Asia Ranges:*\n"]
    for sym, lvl in asia_ranges.items():
        lines.append(f"*{sym}*\n  🔼 High: `{lvl['high']}`\n  🔽 Low: `{lvl['low']}`\n")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning all pairs...")

    results = []
    for symbol in SYMBOLS:
        signal = check_signal(symbol)
        if signal:
            results.append(format_signal(signal))

    if results:
        for msg in results:
            await update.message.reply_text(msg, parse_mode="Markdown")
    else:
        now_wat = datetime.now(WAT)
        now_time = now_wat.time()

        if not (LONDON_START <= now_time < LONDON_END):
            reason = (
                "⏰ Outside London window (8AM–12PM WAT).\n"
                "Signals only fire during London session."
            )
        elif not asia_ranges:
            reason = (
                "📐 No Asia range marked yet.\n"
                "Range is set from 4H candle at 1AM–2AM WAT."
            )
        else:
            reason = "❌ No valid signals right now.\nConditions not fully aligned yet."

        await update.message.reply_text(reason)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def post_init(app: Application):
    asyncio.create_task(monitor_loop(app))


def main():
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("stop",   cmd_stop))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("range",  cmd_range))

    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
