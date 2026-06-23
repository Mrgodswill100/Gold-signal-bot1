"""
London Session Strategy - Final

Strategy Rules:
1. 4H candle range high/low must be swiped
2. 5M structure shift (break of RECENT high/low)
3. RSI crosses 50 in same direction

TRADING LOGIC:
- 4H LOW swiped  → Look for BULLISH shift + RSI up → BUY
- 4H HIGH swiped → Look for BEARISH shift + RSI down → SELL
"""
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, time
import pytz

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")

# Pairs to monitor
PAIRS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "US500": "SPX",
    "US30": "DJI",
    "NAS100": "NDX"
}

# Session times (GMT)
LONDON_START = time(7, 0)
LONDON_END = time(15, 0)
LEVEL_START = time(13, 0)
LEVEL_END = time(14, 0)

RSI_PERIOD = 14

# Track 4H levels
_levels_cache = {}
_levels_date = None


# ---------- Data ----------

def fetch_candles(symbol, interval, outputsize=150):
    """Fetch candle data from TwelveData"""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()
    
    if "values" not in data:
        raise ValueError(f"TwelveData error ({interval}): {data.get('message', data)}")
    
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "time"})
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ---------- Indicators ----------

def rsi(series, period=RSI_PERIOD):
    """Calculate RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------- Strategy Functions ----------

def get_4h_levels(symbol):
    """Get 4H candle high and low"""
    try:
        df = fetch_candles(symbol, "4h", 1)
        if df.empty:
            return None, None
        return df["high"].iloc[-1], df["low"].iloc[-1]
    except Exception as e:
        print(f"Error getting 4H levels for {symbol}: {e}")
        return None, None


def detect_structure_shift(df):
    """
    Detect 5M structure shift using RECENT high/low
    
    RECENT HIGH = Highest price in last 10 candles
    RECENT LOW = Lowest price in last 10 candles
    
    Returns: "BULLISH" (break above recent high) or "BEARISH" (break below recent low)
    """
    if len(df) < 20:
        return None
    
    # RECENT high and low (last 10 candles)
    recent_high = df["high"].iloc[-10:].max()
    recent_low = df["low"].iloc[-10:].min()
    
    current_close = df["close"].iloc[-1]
    previous_close = df["close"].iloc[-2]
    
    # Break above RECENT high = BULLISH
    if current_close > recent_high and previous_close <= recent_high:
        return "BULLISH"
    # Break below RECENT low = BEARISH
    elif current_close < recent_low and previous_close >= recent_low:
        return "BEARISH"
    
    return None


def check_level_swiped(price, level_high, level_low, tolerance=0.001):
    """Check if price has swiped 4H high or low"""
    if price >= level_high * (1 - tolerance):
        return "HIGH_SWIPED"
    elif price <= level_low * (1 + tolerance):
        return "LOW_SWIPED"
    return None


def analyze_pair(display_name):
    """
    Strategy Rules:
    - 4H LOW swiped  → Look for BULLISH shift + RSI up → BUY
    - 4H HIGH swiped → Look for BEARISH shift + RSI down → SELL
    """
    symbol = PAIRS[display_name]
    reasons = []
    
    # 1. Get 4H levels
    global _levels_cache, _levels_date
    current_date = datetime.now(pytz.timezone('GMT')).date()
    
    if _levels_date != current_date:
        _levels_cache = {}
        _levels_date = current_date
    
    if display_name not in _levels_cache:
        high, low = get_4h_levels(symbol)
        if high is not None and low is not None:
            _levels_cache[display_name] = (high, low)
        else:
            reasons.append("Failed to get 4H levels")
            return {"direction": "NEUTRAL", "price": 0, "level_high": 0, "level_low": 0, "signal_id": "", "reasons": reasons}
    
    level_high, level_low = _levels_cache[display_name]
    
    # 2. Get 5M data
    try:
        df_5m = fetch_candles(symbol, "5min", 100)
        if df_5m.empty:
            reasons.append("Failed to get 5M data")
            return {"direction": "NEUTRAL", "price": 0, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    except Exception as e:
        reasons.append(f"Error fetching 5M data: {e}")
        return {"direction": "NEUTRAL", "price": 0, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    last = df_5m.iloc[-1]
    price = last["close"]
    
    # 3. Check if 4H level was swiped
    swipe_status = check_level_swiped(price, level_high, level_low)
    if swipe_status is None:
        reasons.append(f"Price {price:.4f} - No 4H swipe yet")
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    reasons.append(f"✓ 4H {swipe_status}")
    
    # 4. Check RSI 50 cross
    df_5m["rsi"] = rsi(df_5m["close"], RSI_PERIOD)
    current_rsi = df_5m["rsi"].iloc[-1]
    previous_rsi = df_5m["rsi"].iloc[-2]
    
    rsi_direction = None
    if current_rsi > 50 and previous_rsi <= 50:
        rsi_direction = "UP"
        reasons.append(f"✓ RSI crossed UP above 50 ({current_rsi:.1f})")
    elif current_rsi < 50 and previous_rsi >= 50:
        rsi_direction = "DOWN"
        reasons.append(f"✓ RSI crossed DOWN below 50 ({current_rsi:.1f})")
    else:
        reasons.append(f"RSI {current_rsi:.1f} - No 50 cross")
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    # 5. Check structure shift (using RECENT high/low)
    structure_shift = detect_structure_shift(df_5m)
    if structure_shift is None:
        reasons.append("No structure shift")
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    reasons.append(f"✓ Structure shift: {structure_shift}")
    
    # 6. DETERMINE TRADE DIRECTION
    direction = None
    
    # LOW SWIPED → Need BULLISH shift + RSI UP → BUY
    if swipe_status == "LOW_SWIPED":
        if structure_shift == "BULLISH" and rsi_direction == "UP":
            direction = "BUY"
            reasons.append("✅ LOW SWIPED + BULLISH shift + RSI UP = BUY")
        else:
            reasons.append(f"⏳ LOW swiped - waiting for BULLISH shift + RSI UP")
    
    # HIGH SWIPED → Need BEARISH shift + RSI DOWN → SELL
    elif swipe_status == "HIGH_SWIPED":
        if structure_shift == "BEARISH" and rsi_direction == "DOWN":
            direction = "SELL"
            reasons.append("✅ HIGH SWIPED + BEARISH shift + RSI DOWN = SELL")
        else:
            reasons.append(f"⏳ HIGH swiped - waiting for BEARISH shift + RSI DOWN")
    
    if direction is None:
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    # 7. Generate signal ID
    signal_id = f"{display_name}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    return {
        "direction": direction,
        "price": price,
        "level_high": level_high,
        "level_low": level_low,
        "signal_id": signal_id,
        "reasons": reasons,
        "time": last["time"]
    }


# ---------- Legacy function ----------

def analyze():
    """Legacy function for compatibility"""
    if "XAUUSD" not in PAIRS:
        PAIRS["XAUUSD"] = "XAU/USD"
    return analyze_pair("XAUUSD")
