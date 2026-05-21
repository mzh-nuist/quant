# -*- coding: utf-8 -*-
"""
因子计算模块
-----------
基于 OHLCV 历史数据计算多因子原始值。

因子定义：
  - momentum_20d : 过去 20 个交易日的累计收益率
  - volatility_60d : 过去 60 个交易日的日收益率标准差（乘以 -1，偏好低波动）
  - volume_ratio : 近 5 日均量 / 近 20 日均量 - 1（乘以 -1，回避异常放量）
  - rsi_14d : 14 日 RSI 偏离 50 的程度（乘以 -1，偏好非超买）
  - size_log : ln(日均成交额)（乘以 -1，小盘偏好）
"""

import numpy as np
import pandas as pd


def momentum_factor(close: pd.Series, period: int = 20) -> float:
    """
    动量因子：过去 N 日的价格收益率。
    正值表示上涨趋势，预期正相关。
    """
    if len(close) < period + 1:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-(period + 1)] - 1.0)


def volatility_factor(close: pd.Series, period: int = 60) -> float:
    """
    低波动因子：过去 N 日年化波动率的相反数。
    低波动股票通常风险调整后收益更高。
    """
    if len(close) < max(period, 5):
        return np.nan
    rets = close.pct_change().dropna().iloc[-period:]
    if len(rets) < 5:
        return np.nan
    daily_vol = float(rets.std())
    return -daily_vol  # 取反：低波动 → 高分


def volume_factor(volume: pd.Series, short: int = 5, long_period: int = 20) -> float:
    """
    量能因子：短期均量相对长期均量的偏离（取反）。
    异常放量可能意味着筹码松动或游资炒作。
    """
    if len(volume) < long_period:
        return np.nan
    avg_short = float(volume.iloc[-short:].mean())
    avg_long = float(volume.iloc[-long_period:].mean())
    if avg_long <= 0:
        return np.nan
    ratio = avg_short / avg_long - 1.0
    return -ratio  # 取反：缩量 → 高分


def rsi_factor(close: pd.Series, period: int = 14) -> float:
    """
    反转因子（基于 RSI）：RSI 偏离 50 的程度（取反）。
    RSI 过高意味着短期超买，未来可能回调。
    """
    if len(close) < period + 1:
        return np.nan
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = float(gain.iloc[-period:].mean())
    avg_loss = float(loss.iloc[-period:].mean())
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    return -(rsi - 50.0) / 50.0  # 归一化到约 [-1, 1]，取反


def size_factor(close: pd.Series, volume: pd.Series, period: int = 20) -> float:
    """
    规模因子：log(日均成交额) 的相反数。
    小盘股在 A 股市场存在长期溢价效应。
    """
    if len(close) < period or len(volume) < period:
        return np.nan
    avg_amount = float((close.iloc[-period:] * volume.iloc[-period:]).mean())
    if avg_amount <= 0:
        return np.nan
    return -np.log(avg_amount + 1.0)


# ---------------------------------------------------------------------------
#  批量计算
# ---------------------------------------------------------------------------
FACTOR_DEFS = {
    "momentum":    ("动量(20日)",     momentum_factor,    True),
    "volatility":  ("低波动(60日)",    volatility_factor,  True),
    "volume_ratio":("低换手(5/20)",    volume_factor,      True),
    "rsi":         ("RSI反转(14日)",   rsi_factor,         True),
    "size":        ("小盘规模",        size_factor,        True),
}


def compute_all_factors(
    close: pd.Series, volume: pd.Series
) -> dict | None:
    """
    计算单只股票的全部因子值。

    参数
    ----
    close : pd.Series
        收盘价序列（按日期升序）。
    volume : pd.Series
        成交量序列（按日期升序）。

    返回
    ----
    dict | None
        {factor_name: value, ...}，数据不足时返回 None。
    """
    try:
        return {
            "momentum": momentum_factor(close),
            "volatility": volatility_factor(close),
            "volume_ratio": volume_factor(volume),
            "rsi": rsi_factor(close),
            "size": size_factor(close, volume),
        }
    except Exception:
        return None
