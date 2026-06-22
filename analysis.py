"""
Gold (XAUUSD) Smart Money Concepts analysis — multi-timeframe.

Logic:
  4H  -> macro trend bias (EMA structure + higher high/higher low pattern)
  1H  -> confirms bias is intact, identifies the active supply/demand zone
  15M -> waits for price to actually tap that zone and react, confirms
         with RSI/MACD before firing a signal

A signal only fires when ALL THREE align. This is intentionally strict —
fewer signals, but each one means 4H trend + 1H structure + 15M reaction
all agree, matching real SMC confluence trading instead of a simple
indicator vote.
"""
import requests
import pandas as pd
import numpy as np
import os

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
SYMBOL = "XAU/USD"
RR_RATIO = 2  # 1:2 risk-reward

ZONE_LOOKBACK = 30        # candles to scan for swing-based zones
ZONE_TOUCH_BUFFER = 0.0015  # 0.15% price buffer to count as "tapping" a zone


# ---------- Data ----------

def fetch_candles(interval, outputsize=150):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
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

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line  # histogram


def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------- Structure / bias ----------

def find_swings(df, window=3):
    """Mark swing highs/lows using a simple fractal: a candle whose high/low
    is the most extreme within `window` candles on each side."""
    highs, lows = df["high"], df["low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    for i in range(window, len(df) - window):
        if highs[i] == highs[i - window:i + window + 1].max():
            swing_high[i] = True
        if lows[i] == lows[i - window:i + window + 1].min():
            swing_low[i] = True
    return swing_high, swing_low


def determine_bias(df):
    """Determine trend bias from EMA stack + swing structure
    (higher highs/higher lows = bullish, lower highs/lower lows = bearish)."""
    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    swing_high, swing_low = find_swings(df, window=3)

    recent_highs = df.loc[swing_high, "high"].tail(2).tolist()
    recent_lows = df.loc[swing_low, "low"].tail(2).tolist()

    structure = "unclear"
    if len(recent_highs) == 2 and len(recent_lows) == 2:
        if recent_highs[-1] > recent_highs[-2] and recent_lows[-1] > recent_lows[-2]:
            structure = "bullish"
        elif recent_highs[-1] < recent_highs[-2] and recent_lows[-1] < recent_lows[-2]:
            structure = "bearish"

    last = df.iloc[-1]
    ema_trend = "bullish" if last["ema20"] > last["ema50"] else "bearish"

    if structure == ema_trend:
        bias = structure
    elif structure == "unclear":
        bias = ema_trend
    else:
        bias = "mixed"  # EMA and structure disagree -> no clean bias

    return bias, df


# ---------- Supply / demand zones ----------

def find_zones(df, window=3):
    """Identify the most recent unmitigated demand zone (last bullish swing
    low with a strong move away) and supply zone (last bearish swing high
    with a strong move away)."""
    swing_high, swing_low = find_swings(df, window=window)

    demand_zone = None
    for i in df.index[swing_low][::-1]:
        if i + 1 < len(df) and df["close"][i + 1] > df["high"][i]:
            demand_zone = (df["low"][i], df["high"][i])
            break

    supply_zone = None
    for i in df.index[swing_high][::-1]:
        if i + 1 < len(df) and df["close"][i + 1] < df["low"][i]:
            supply_zone = (df["low"][i], df["high"][i])
            break

    return demand_zone, supply_zone


def price_in_zone(price, zone, buffer_pct=ZONE_TOUCH_BUFFER):
    if zone is None:
        return False
    low, high = zone
    buffer = price * buffer_pct
    return (low - buffer) <= price <= (high + buffer)


# ---------- Main analysis ----------

def analyze():
    df_4h = fetch_candles("4h", 100)
    df_1h = fetch_candles("1h", 150)
    df_15m = fetch_candles("15min", 150)

    bias_4h, _ = determine_bias(df_4h)
    bias_1h, df_1h = determine_bias(df_1h)

    demand_zone, supply_zone = find_zones(df_1h)

    df_15m = df_15m.copy()
    df_15m["rsi"] = rsi(df_15m["close"], 14)
    df_15m["macd_hist"] = macd(df_15m["close"])
    df_15m["atr"] = atr(df_15m, 14)

    last = df_15m.iloc[-1]
    price = last["close"]

    reasons = [
        f"4H bias: {bias_4h}",
        f"1H bias: {bias_1h}",
    ]

    direction = "NEUTRAL"

    # Require 4H and 1H to agree before considering any trade
    htf_bias = None
    if bias_4h == bias_1h and bias_4h in ("bullish", "bearish"):
        htf_bias = bias_4h
        reasons.append("4H and 1H bias aligned")
    else:
        reasons.append("4H/1H bias not aligned — no trade")

    if htf_bias == "bullish":
        reasons.append(
            f"Demand zone: {demand_zone[0]:.2f}-{demand_zone[1]:.2f}" if demand_zone
            else "No clear demand zone found"
        )
        if price_in_zone(price, demand_zone):
            reasons.append(f"Price ({price:.2f}) is reacting at demand zone")
            if last["rsi"] < 50 and last["macd_hist"] > df_15m["macd_hist"].iloc[-2]:
                reasons.append(f"RSI {last['rsi']:.1f} recovering, MACD histogram rising")
                direction = "BUY"
            else:
                reasons.append("Waiting — 15M momentum not confirming yet")
        else:
            reasons.append("Price has not reached the demand zone yet")

    elif htf_bias == "bearish":
        reasons.append(
            f"Supply zone: {supply_zone[0]:.2f}-{supply_zone[1]:.2f}" if supply_zone
            else "No clear supply zone found"
        )
        if price_in_zone(price, supply_zone):
            reasons.append(f"Price ({price:.2f}) is reacting at supply zone")
            if last["rsi"] > 50 and last["macd_hist"] < df_15m["macd_hist"].iloc[-2]:
                reasons.append(f"RSI {last['rsi']:.1f} fading, MACD histogram falling")
                direction = "SELL"
            else:
                reasons.append("Waiting — 15M momentum not confirming yet")
        else:
            reasons.append("Price has not reached the supply zone yet")

    atr_val = last["atr"]
    sl_distance = atr_val * 1.0
    tp_distance = sl_distance * RR_RATIO

    if direction == "BUY":
        sl = price - sl_distance
        tp = price + tp_distance
    elif direction == "SELL":
        sl = price + sl_distance
        tp = price - tp_distance
    else:
        sl = tp = None

    return {
        "direction": direction,
        "price": price,
        "sl": sl,
        "tp": tp,
        "atr": atr_val,
        "reasons": reasons,
        "time": last["time"],
    }
    
