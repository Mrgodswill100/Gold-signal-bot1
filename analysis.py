"""
Gold (XAUUSD) signal analysis using EMA, RSI, MACD, and ATR.
Pulls 15-minute candle data from TwelveData free API.
"""
import requests
import pandas as pd
import numpy as np
import os

TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
SYMBOL = "XAU/USD"
INTERVAL = "15min"
RR_RATIO = 2  # 1:2 risk-reward


def fetch_candles(outputsize=100):
    """Fetch recent 15M candles for XAU/USD from TwelveData."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    resp = requests.get(url, params=params, timeout=15)
    data = resp.json()

    if "values" not in data:
        raise ValueError(f"TwelveData error: {data.get('message', data)}")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "time", "open": "open", "high": "high",
        "low": "low", "close": "close"
    })
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


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
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def analyze():
    """Run full analysis and return a signal dict."""
    df = fetch_candles(150)

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi"] = rsi(df["close"], 14)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    df["atr"] = atr(df, 14)

    last = df.iloc[-1]
    price = last["close"]

    votes = []
    reasons = []

    # Trend: EMA20 vs EMA50
    if last["ema20"] > last["ema50"]:
        votes.append(1)
        reasons.append("EMA20 > EMA50 (uptrend)")
    else:
        votes.append(-1)
        reasons.append("EMA20 < EMA50 (downtrend)")

    # Momentum: RSI
    if last["rsi"] > 55:
        votes.append(1)
        reasons.append(f"RSI {last['rsi']:.1f} (bullish momentum)")
    elif last["rsi"] < 45:
        votes.append(-1)
        reasons.append(f"RSI {last['rsi']:.1f} (bearish momentum)")
    else:
        votes.append(0)
        reasons.append(f"RSI {last['rsi']:.1f} (neutral)")

    # Confirmation: MACD histogram
    if last["macd_hist"] > 0:
        votes.append(1)
        reasons.append("MACD histogram positive")
    else:
        votes.append(-1)
        reasons.append("MACD histogram negative")

    score = sum(votes)

    if score >= 2:
        direction = "BUY"
    elif score <= -2:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

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
