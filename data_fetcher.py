# -*- coding: utf-8 -*-
"""
数据获取模块
-----------
支持两种数据来源：
  1. akshare 在线获取 A 股历史日线行情（优先）
  2. 本地模拟数据（离线回退，确保在网络不可用时也能运行）

数据自动缓存到本地 CSV，避免重复请求网络。
"""

import os
import numpy as np
import pandas as pd
import akshare as ak


def _generate_sample_data(
    start_date: str = "20200101",
    end_date: str = "20251231",
    symbol: str = "600519",
) -> pd.DataFrame:
    """
    生成模拟股票日线数据（具有趋势+震荡特征的随机游走价格序列）。

    不同股票代码会产生不同的价格走势、波动率和价格中枢，
    确保每个标的都有独立且可复现的模拟行情。

    当 akshare 网络请求失败时用作回退数据源，保证策略在任何
    环境下都能正常运行和验证。
    """
    # 根据股票代码生成确定性随机种子：同一代码永远生成同一走势
    base_seed = hash(symbol) % 2**31
    rng = np.random.default_rng(base_seed)

    # 生成交易日序列（剔除周末）
    date_range = pd.bdate_range(start=start_date, end=end_date)
    n = len(date_range)

    # 每只股票有不同的特征参数（由 seed 唯一确定）
    # 价格中枢：20 ~ 200 之间
    base_price = 20 + rng.uniform(0, 180)
    # 趋势振幅：价格的 15% ~ 60%
    trend_amp = base_price * rng.uniform(0.15, 0.60)
    # 噪声幅度
    noise_scale = base_price * rng.uniform(0.02, 0.12)
    # 正弦周期偏移量（让不同股票的涨跌节奏错开）
    phase_shift = rng.uniform(0, 2 * np.pi)

    # ---------- 价格生成：多段趋势 + 随机游走 ----------
    t = np.linspace(0, 4 * np.pi, n) + phase_shift
    trend = base_price + trend_amp * np.sin(t)

    # 叠加累积噪声模拟随机游走
    noise = rng.normal(0, noise_scale, n).cumsum()
    close = trend + noise

    # 确保价格不为负
    close = close - close.min() + base_price * 0.2

    # 基于收盘价生成 OHLC
    daily_range = close * rng.uniform(0.015, 0.05, n)  # 日内振幅 1.5%~5%
    open_price = close - rng.uniform(-0.5, 0.5, n) * daily_range
    high = np.maximum(open_price, close) + rng.uniform(0, 0.5, n) * daily_range
    low = np.minimum(open_price, close) - rng.uniform(0, 0.5, n) * daily_range

    # 成交量：对数正态分布，与日内振幅正相关
    base_volume = rng.lognormal(mean=14, sigma=0.6, size=n)
    volume_multiplier = 1 + daily_range / close * 10
    volume = (base_volume * volume_multiplier).astype(int)

    # 组装 DataFrame
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=date_range)

    df.index.name = "date"
    print(f"[模拟数据] 已生成 {len(df)} 根 K 线（离线模式，symbol={symbol}）")
    return df


def fetch_stock_daily(
    symbol: str = "600519",
    start_date: str = "20200101",
    end_date: str = "20251231",
    adjust: str = "qfq",
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    获取 A 股历史日线数据：优先 akshare 在线获取，失败则用模拟数据回退。

    参数
    ----
    symbol : str
        股票代码（不含市场前缀），如 "600519" 代表贵州茅台。
    start_date : str
        起始日期，格式 YYYYMMDD。
    end_date : str
        结束日期，格式 YYYYMMDD。
    adjust : str
        复权类型："qfq"前复权 / "hfq"后复权 / ""不复权。
    cache_dir : str
        本地缓存目录，默认为 data/。

    返回
    ----
    pd.DataFrame
        包含 date(索引), open, high, low, close, volume 列的 DataFrame。
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 缓存文件路径（真实数据与模拟数据使用不同前缀以避免混淆）
    cache_file = os.path.join(
        cache_dir, f"{symbol}_{start_date}_{end_date}_{adjust}.csv"
    )

    # 如果缓存存在，直接读取
    if os.path.exists(cache_file):
        print(f"[缓存命中] 从本地加载: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        keep_cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in keep_cols if c in df.columns]]
        return df

    # ---- 尝试从 akshare 在线获取真实数据 ----
    print(f"[网络请求] 正在获取 {symbol} 的历史行情...")
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception as e:
        print(f"[网络失败] akshare 请求失败: {e}")
        print("[回退方案] 使用本地模拟数据运行回测...")
        return _generate_sample_data(start_date, end_date, symbol)

    if df.empty:
        print("[警告] akshare 返回空数据，使用模拟数据代替。")
        return _generate_sample_data(start_date, end_date, symbol)

    # ---- 清洗真实数据 ----
    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df.rename(columns=column_map, inplace=True)

    keep_cols = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # 保存到本地 CSV 缓存
    df.to_csv(cache_file, encoding="utf-8-sig")
    print(f"[缓存写入] 数据已保存到: {cache_file}")

    return df


if __name__ == "__main__":
    df = fetch_stock_daily("600519", "20230101", "20251231", adjust="qfq")
    print(f"\n数据维度: {df.shape}")
    print(f"日期范围: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(df.head())
