"""
London Session Strategy - Multi-pair analysis

Strategy Rules:
1. 4H candle high/low must be swiped (touched)
2. 5M structure shift (break of swing high/low)
3. RSI crosses 50 midline

Pairs: EURUSD, GBPUSD, US500, US30, NAS100
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
    "US500": "SPX",      # S&P 500
    "US30": "DJI",       # Dow Jones
    "NAS100": "NDX"      # Nasdaq
}

# Session times (GMT)
LONDON_START = time(7, 0)
LONDON_END = time(15, 0)
LEVEL_START = time(13, 0)
LEVEL_END = time(14, 0)

# RSI period
RSI_PERIOD = 14

# Track 4H levels per pair (reset daily)
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


def atr(df, period=14):
    """Average True Range"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------- Strategy Functions ----------

def get_4h_levels(symbol):
    """Get 4H candle high and low for a symbol"""
    try:
        df = fetch_candles(symbol, "4h", 1)
        if df.empty:
            return None, None
        return df["high"].iloc[-1], df["low"].iloc[-1]
    except Exception as e:
        print(f"Error getting 4H levels for {symbol}: {e}")
        return None, None


def is_london_session():
    """Check if currently in London trading session"""
    now = datetime.now(pytz.timezone('GMT')).time()
    return LONDON_START <= now <= LONDON_END


def is_level_formation_time():
    """Check if it's 1-2pm GMT for 4H level formation"""
    now = datetime.now(pytz.timezone('GMT')).time()
    return LEVEL_START <= now <= LEVEL_END


def detect_structure_shift(df):
    """Detect 5M structure shift (break of swing high/low)"""
    if len(df) < 20:
        return None
    
    # Recent swing high and low (last 10 candles)
    recent_high = df["high"].iloc[-10:].max()
    recent_low = df["low"].iloc[-10:].min()
    
    current_close = df["close"].iloc[-1]
    previous_close = df["close"].iloc[-2]
    
    # Break above swing high = BUY signal
    if current_close > recent_high and previous_close <= recent_high:
        return "BUY"
    # Break below swing low = SELL signal
    elif current_close < recent_low and previous_close >= recent_low:
        return "SELL"
    
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
    Analyze a single pair for London session strategy signals
    
    Returns:
        dict: {
            "direction": "BUY" | "SELL" | "NEUTRAL",
            "price": float,
            "level_high": float,
            "level_low": float,
            "signal_id": str,
            "reasons": list
        }
    """
    symbol = PAIRS[display_name]
    reasons = []
    
    # 1. Get 4H levels (only valid during/after 1-2pm GMT)
    global _levels_cache, _levels_date
    current_date = datetime.now(pytz.timezone('GMT')).date()
    
    # Reset cache at midnight
    if _levels_date != current_date:
        _levels_cache = {}
        _levels_date = current_date
    
    if display_name not in _levels_cache:
        high, low = get_4h_levels(symbol)
        if high is not None and low is not None:
            _levels_cache[display_name] = (high, low)
            reasons.append(f"4H Level set: High={high:.4f}, Low={low:.4f}")
        else:
            reasons.append("Failed to get 4H levels")
            return {"direction": "NEUTRAL", "price": 0, "level_high": 0, "level_low": 0, "signal_id": "", "reasons": reasons}
    
    level_high, level_low = _levels_cache[display_name]
    
    # 2. Get current price (from 5M data)
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
        reasons.append(f"Price ({price:.4f}) has not swiped 4H levels yet")
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    reasons.append(f"✓ 4H level swiped: {swipe_status}")
    
    # 4. Check structure shift
    structure_signal = detect_structure_shift(df_5m)
    if structure_signal is None:
        reasons.append("No 5M structure shift detected")
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    reasons.append(f"✓ 5M structure shift: {structure_signal}")
    
    # 5. Check RSI 50 cross
    df_5m["rsi"] = rsi(df_5m["close"], RSI_PERIOD)
    current_rsi = df_5m["rsi"].iloc[-1]
    previous_rsi = df_5m["rsi"].iloc[-2]
    
    rsi_signal = None
    if current_rsi > 50 and previous_rsi <= 50:
        rsi_signal = "BUY"
        reasons.append(f"✓ RSI crossed above 50 ({current_rsi:.1f})")
    elif current_rsi < 50 and previous_rsi >= 50:
        rsi_signal = "SELL"
        reasons.append(f"✓ RSI crossed below 50 ({current_rsi:.1f})")
    else:
        reasons.append(f"RSI at {current_rsi:.1f} - no 50 cross")
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    # 6. Confirm all conditions align
    direction = None
    if structure_signal == "BUY" and rsi_signal == "BUY" and swipe_status == "HIGH_SWIPED":
        direction = "BUY"
    elif structure_signal == "SELL" and rsi_signal == "SELL" and swipe_status == "LOW_SWIPED":
        direction = "SELL"
    else:
        reasons.append("Conditions don't align (structure, RSI, swipe mismatch)")
        return {"direction": "NEUTRAL", "price": price, "level_high": level_high, "level_low": level_low, "signal_id": "", "reasons": reasons}
    
    # 7. Generate signal ID for deduplication
    signal_id = f"{display_name}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    reasons.append(f"✅ ALL CONDITIONS MET! {direction} signal confirmed")
    
    return {
        "direction": direction,
        "price": price,
        "level_high": level_high,
        "level_low": level_low,
        "signal_id": signal_id,
        "reasons": reasons,
        "time": last["time"]
    }


# ---------- Original analyze() for backward compatibility ----------

def analyze():
    """
    Legacy function - returns XAUUSD analysis
    Kept for compatibility with existing /signal command
    """
    # Check if XAUUSD is in our pairs
    if "XAUUSD" not in PAIRS:
        PAIRS["XAUUSD"] = "XAU/USD"
    
    return analyze_pair("XAUUSD")
