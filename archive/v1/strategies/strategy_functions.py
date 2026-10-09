import pandas as pd
import numpy as np

def generate_macd_rsi_signals(df):
    df = df.copy()
    df["Signal"] = 0
    df.loc[(df["MACD"] > df["MACD_Signal"]) & (df["RSI"] < 30), "Signal"] = 1
    df.loc[(df["MACD"] < df["MACD_Signal"]) & (df["RSI"] > 70), "Signal"] = -1
    return df

def run_backtest(df, signal_col="Signal"):
    df = df.copy()
    df["Position"] = df[signal_col].replace(0, np.nan).ffill().fillna(0)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Strategy_Return"] = df["Position"].shift(1) * df["Daily_Return"]
    df["Portfolio_Value"] = (1 + df["Strategy_Return"]).cumprod()
    return df
