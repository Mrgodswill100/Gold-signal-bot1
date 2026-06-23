"""
London Sweep Strategy - Upgraded

Strategy Rules:
1. Mark 4H candle HIGH/LOW formed between 1AM–2AM WAT (Asia midnight range)
2. Price sweeps that HIGH or LOW during London session (8AM–12PM WAT)
3. 1M structure break (CHoCH) OR Inverse FVG after sweep
4. RSI(14) crosses 50 midline on 1M — same direction as sweep

TRADE LOGIC:
- LOW swiped  → 1M BULLISH CHoCH/IFVG + RSI crosses UP 50  → BUY
- HIGH swiped → 1M BEARISH CHoCH/IFVG + RSI crosses DOWN 50 → SELL
"""

import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, time
import pytz

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")

WAT = pytz.timezone("Africa/Lagos")   # UTC+1

# Pairs to monitor
PAIRS = {
    "EURUSD":  "EUR/USD",
    "GBPUSD":  "GBP/USD",
    "US500":   "SPX",
    "US30":    "DJI",
    "NAS100":  "NDX",
}

# ── Time windows (WAT) ───────────────────────────────────────────────────────
RANGE_START  = time(1,  0)   # 1:00 AM WAT → start of Asia range candle
RANGE_END    = time(2,  0)   # 2:00 AM WAT → end of Asia range candle
LONDON_START = time(8,  0)   # 8:00 AM WAT → London open
LONDON_END   = time(12, 0)   # 12:00 PM WAT → hard stop, no signals after

RSI_PERIOD = 14

# ── Daily cache ──────────────────────────────────────────────────────────────
_asia_cache: dict = {}     # { display_name: { high, low, date } }


# ─── DATA ────────────────────────────────────────────────────────────────────

def fetch_candles(symbol: str, interval: str, outputsize: int = 150) -> pd.DataFrame:
    """Fetch OHLCV candles from TwelveData REST API."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     TWELVEDATA_API_KEY,
        "order":      "ASC",
    }
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()

    if "values" not in data:
        raise ValueError(f"TwelveData error [{interval}]: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ─── INDICATORS ──────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── STRATEGY FUNCTIONS ───────────────────────────────────────────────────────

def get_asia_range(display_name: str) -> dict | None:
    """
    Fetch 4H candles and return the HIGH/LOW of the candle
    whose open falls within 1AM–2AM WAT.
    Caches result for the day.
    """
    global _asia_cache
    today = datetime.now(WAT).date()

    cached = _asia_cache.get(display_name)
    if cached and cached["date"] == today:
        return cached

    symbol = PAIRS[display_name]
    try:
        df = fetch_candles(symbol, "4h", outputsize=20)
    except Exception as e:
        print(f"[{display_name}] Asia range fetch error: {e}")
        return None

    for _, row in df.iterrows():
        # Convert candle time to WAT
        candle_time_utc = row["time"].to_pydatetime().replace(tzinfo=pytz.utc)
        candle_wat      = candle_time_utc.astimezone(WAT)

        if (candle_wat.date() == today and
                RANGE_START <= candle_wat.time() < RANGE_END):

            level = {
                "high": float(row["high"]),
                "low":  float(row["low"]),
                "date": today,
            }
            _asia_cache[display_name] = level
            print(f"[{display_name}] Asia range → H:{level['high']} L:{level['low']}")
            return level

    print(f"[{display_name}] No 4H candle found in 1AM–2AM WAT window yet.")
    return None


def check_sweep(price: float, asia_high: float, asia_low: float) -> str | None:
    """
    Returns 'HIGH_SWIPED', 'LOW_SWIPED', or None.
    Small tolerance (0.05%) to catch wicks that barely tag the level.
    """
    tolerance = 0.0005
    if price >= asia_high * (1 - tolerance):
        return "HIGH_SWIPED"
    if price <= asia_low * (1 + tolerance):
        return "LOW_SWIPED"
    return None


def detect_structure_and_ifvg(df: pd.DataFrame) -> str | None:
    """
    Detect on 1M data:
      A) CHoCH — break of recent swing high/low (last 10 candles)
      B) Inverse FVG — unfilled gap between candle[-3] and candle[-1]

    Returns 'BULLISH', 'BEARISH', or None.
    """
    if len(df) < 15:
        return None

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    # ── A) CHoCH ─────────────────────────────────────────────────────────────
    recent_high = max(highs[-11:-1])   # swing high of last 10 candles
    recent_low  = min(lows[-11:-1])    # swing low  of last 10 candles
    prev_close  = closes[-2]
    last_close  = closes[-1]

    if last_close > recent_high and prev_close <= recent_high:
        return "BULLISH"
    if last_close < recent_low  and prev_close >= recent_low:
        return "BEARISH"

    # ── B) Inverse FVG ───────────────────────────────────────────────────────
    # 3-candle gap pattern: candle[-3], candle[-2] (body), candle[-1]
    # Bullish IFVG: gap between candle[-3].low and candle[-1].high
    #              price (close[-1]) fills into that gap from below
    # Bearish IFVG: gap between candle[-3].high and candle[-1].low
    #              price (close[-1]) fills into that gap from above

    c3_high = highs[-3]
    c3_low  = lows[-3]
    c1_high = highs[-1]
    c1_low  = lows[-1]

    # Bullish IFVG: c3_low > c1_high (gap above price, now being filled upward)
    if c3_low > c1_high and last_close >= c1_high:
        return "BULLISH"

    # Bearish IFVG: c3_high < c1_low (gap below price, now being filled downward)
    if c3_high < c1_low and last_close <= c1_low:
        return "BEARISH"

    return None


def check_rsi_cross(df: pd.DataFrame) -> str | None:
    """
    Returns 'UP' if RSI just crossed above 50,
            'DOWN' if RSI just crossed below 50,
            None otherwise.
    """
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)

    if len(df) < RSI_PERIOD + 2:
        return None

    rsi_now  = df["rsi"].iloc[-1]
    rsi_prev = df["rsi"].iloc[-2]

    if rsi_prev <= 50 < rsi_now:
        return "UP"
    if rsi_prev >= 50 > rsi_now:
        return "DOWN"
    return None


# ─── MAIN ANALYSIS FUNCTION ───────────────────────────────────────────────────

def analyze_pair(display_name: str) -> dict:
    """
    Full strategy analysis for one pair.
    Returns dict with direction, price, levels, reasons.
    """
    symbol  = PAIRS[display_name]
    reasons = []

    def neutral(msg=None):
        if msg:
            reasons.append(msg)
        return {
            "direction":  "NEUTRAL",
            "price":      0,
            "asia_high":  asia_high if "asia_high" in dir() else 0,
            "asia_low":   asia_low  if "asia_low"  in dir() else 0,
            "signal_id":  "",
            "reasons":    reasons,
        }

    # ── 1. Time gate — London window only ────────────────────────────────────
    now_wat  = datetime.now(WAT)
    now_time = now_wat.time()

    if not (LONDON_START <= now_time < LONDON_END):
        return neutral(f"⏰ Outside London window (8AM–12PM WAT). Now: {now_time.strftime('%H:%M')}")

    # ── 2. Get Asia range (1AM–2AM WAT 4H candle) ────────────────────────────
    level = get_asia_range(display_name)
    if level is None:
        return neutral("⏳ Asia range (1AM–2AM WAT) not available yet.")

    asia_high = level["high"]
    asia_low  = level["low"]
    reasons.append(f"📐 Asia range → H:{asia_high} | L:{asia_low}")

    # ── 3. Fetch 1M candles ──────────────────────────────────────────────────
    try:
        df_1m = fetch_candles(symbol, "1min", outputsize=100)
    except Exception as e:
        return neutral(f"❌ 1M fetch error: {e}")

    if df_1m.empty or len(df_1m) < 20:
        return neutral("❌ Not enough 1M data.")

    price = float(df_1m["close"].iloc[-1])

    # ── 4. Check sweep ────────────────────────────────────────────────────────
    sweep = check_sweep(price, asia_high, asia_low)
    if sweep is None:
        return neutral(f"⏳ No sweep yet. Price: {price:.4f} | H:{asia_high} L:{asia_low}")

    reasons.append(f"✅ {sweep} — price: {price:.4f}")

    # ── 5. Structure break or Inverse FVG on 1M ──────────────────────────────
    structure = detect_structure_and_ifvg(df_1m)
    if structure is None:
        return neutral("⏳ No 1M CHoCH or Inverse FVG yet.")

    reasons.append(f"✅ 1M structure: {structure}")

    # ── 6. RSI cross 50 on 1M ────────────────────────────────────────────────
    rsi_cross = check_rsi_cross(df_1m)
    if rsi_cross is None:
        return neutral(f"⏳ RSI has not crossed 50 yet.")

    reasons.append(f"✅ RSI crossed {rsi_cross} across 50")

    # ── 7. Determine direction ────────────────────────────────────────────────
    direction = None

    if sweep == "LOW_SWIPED":
        if structure == "BULLISH" and rsi_cross == "UP":
            direction = "BUY"
            reasons.append("🟢 LOW swept + BULLISH CHoCH/IFVG + RSI UP → BUY")
        else:
            return neutral(f"⏳ LOW swept — waiting for BULLISH structure + RSI UP")

    elif sweep == "HIGH_SWIPED":
        if structure == "BEARISH" and rsi_cross == "DOWN":
            direction = "SELL"
            reasons.append("🔴 HIGH swept + BEARISH CHoCH/IFVG + RSI DOWN → SELL")
        else:
            return neutral(f"⏳ HIGH swept — waiting for BEARISH structure + RSI DOWN")

    if direction is None:
        return neutral("❌ Conditions not fully aligned.")

    # ── 8. Build result ───────────────────────────────────────────────────────
    signal_id = f"{display_name}_{direction}_{now_wat.strftime('%Y%m%d_%H%M')}"

    return {
        "direction": direction,
        "price":     price,
        "asia_high": asia_high,
        "asia_low":  asia_low,
        "signal_id": signal_id,
        "reasons":   reasons,
        "time":      df_1m["time"].iloc[-1],
        "sweep":     sweep,
        "structure": structure,
        "rsi_cross": rsi_cross,
    }


def analyze_all() -> dict:
    """Run analysis across all pairs. Returns dict of results."""
    return {pair: analyze_pair(pair) for pair in PAIRS}


# ─── LEGACY COMPAT ───────────────────────────────────────────────────────────

def analyze():
    """Legacy function — keeps compatibility with existing bot.py callers."""
    return analyze_pair("EURUSD")
