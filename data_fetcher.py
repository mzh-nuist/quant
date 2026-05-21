# -*- coding: utf-8 -*-
"""
数据获取模块
-----------
数据来源：
  1. Yahoo Finance（yfinance 库，需梯子/代理）
  2. 本地 CSV 缓存（所有数据共用）
  3. 模拟数据（离线回退）

数据自动缓存到本地 CSV，避免重复请求网络。
"""

import os
import numpy as np
import pandas as pd


# ============================================================================
#  A 股代码 → Yahoo Finance 符号
# ============================================================================
def _to_yahoo_symbol(symbol: str) -> str:
    """
    600519 → 600519.SS（上证）
    000001 → 000001.SZ（深证）
    """
    code = symbol.strip()
    if code.startswith(("6", "68")):
        return f"{code}.SS"
    else:
        return f"{code}.SZ"


# ============================================================================
#  Yahoo Finance 数据获取
# ============================================================================
def _fetch_from_yahoo(
    symbol: str,
    start_date: str,
    end_date: str,
    proxy: str | None,
) -> pd.DataFrame:
    """通过 yfinance 获取 A 股历史日线数据。"""
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    import yfinance as yf

    ysym = _to_yahoo_symbol(symbol)
    start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    ticker = yf.Ticker(ysym)
    df = ticker.history(start=start_fmt, end=end_fmt)

    if df.empty:
        raise ValueError(f"Yahoo Finance 返回空数据: {ysym}")

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    keep_cols = ["open", "high", "low", "close", "volume"]
    df = df[[c for c in keep_cols if c in df.columns]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.sort_index(inplace=True)
    df["volume"] = df["volume"].astype(int)

    print(f"  [Yahoo] {ysym}，{len(df)} 条日线")
    return df


# ============================================================================
#  模拟数据生成（离线回退）
# ============================================================================
def _generate_sample_data(
    start_date: str = "20200101",
    end_date: str = "20251231",
    symbol: str = "600519",
) -> pd.DataFrame:
    """
    生成模拟股票日线数据（随机游走 + 正弦趋势）。

    不同股票代码产生不同的确定性走势，确保回测逻辑可验证。
    仅在无缓存且无网络时使用，回测结果不具备参考价值。
    """
    base_seed = hash(symbol) % 2**31
    rng = np.random.default_rng(base_seed)

    date_range = pd.bdate_range(start=start_date, end=end_date)
    n = len(date_range)

    base_price = 20 + rng.uniform(0, 180)
    trend_amp = base_price * rng.uniform(0.15, 0.60)
    noise_scale = base_price * rng.uniform(0.02, 0.12)
    phase_shift = rng.uniform(0, 2 * np.pi)

    t = np.linspace(0, 4 * np.pi, n) + phase_shift
    trend = base_price + trend_amp * np.sin(t)
    noise = rng.normal(0, noise_scale, n).cumsum()
    close = trend + noise
    close = close - close.min() + base_price * 0.2

    daily_range = close * rng.uniform(0.015, 0.05, n)
    open_price = close - rng.uniform(-0.5, 0.5, n) * daily_range
    high = np.maximum(open_price, close) + rng.uniform(0, 0.5, n) * daily_range
    low = np.minimum(open_price, close) - rng.uniform(0, 0.5, n) * daily_range

    base_volume = rng.lognormal(mean=14, sigma=0.6, size=n)
    volume_multiplier = 1 + daily_range / close * 10
    volume = (base_volume * volume_multiplier).astype(int)

    df = pd.DataFrame({
        "open": open_price, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=date_range)
    df.index.name = "date"
    print(f"  [模拟数据] {symbol}，{len(df)} 条日线（离线模式）")
    return df


# ============================================================================
#  统一数据获取入口
# ============================================================================
def fetch_stock_daily(
    symbol: str = "600519",
    start_date: str = "20200101",
    end_date: str = "20251231",
    cache_dir: str = "data",
    force_refresh: bool = False,
    proxy: str | None = None,
) -> pd.DataFrame:
    """
    获取 A 股历史日线数据（缓存优先，Yahoo Finance 在线，模拟回退）。

    参数
    ----
    symbol : str
        A 股代码，如 "600519"。
    start_date / end_date : str
        起止日期 YYYYMMDD。
    cache_dir : str
        本地缓存目录。
    force_refresh : bool
        True = 跳过缓存，强制从 Yahoo Finance 重新获取。
    proxy : str | None
        HTTP 代理地址，如 "http://127.0.0.1:10808"。

    返回
    ----
    pd.DataFrame
        包含 date(索引), open, high, low, close, volume 列的 DataFrame。
    """
    # 使用项目根目录下的绝对路径，避免 CWD 变化导致找不到缓存
    _project_root = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(_project_root, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(
        cache_dir, f"{symbol}_{start_date}_{end_date}.csv"
    )

    # ---- 缓存命中 ----
    if not force_refresh and os.path.exists(cache_file):
        print(f"[本地缓存] {symbol}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        keep_cols = ["open", "high", "low", "close", "volume"]
        return df[[c for c in keep_cols if c in df.columns]]

    # ---- 离线模式 ----
    if not force_refresh:
        print(f"[离线模式] {symbol} 无缓存，使用模拟数据。")
        print(f"[提示] 可用 --refresh 从 Yahoo Finance 获取真实数据。")
        return _generate_sample_data(start_date, end_date, symbol)

    # ---- 在线获取 ----
    print(f"[网络请求] {symbol} ...")
    try:
        df = _fetch_from_yahoo(symbol, start_date, end_date, proxy)
    except Exception as e:
        print(f"[网络失败] {symbol}: {e}")
        print("[回退] 使用模拟数据。")
        return _generate_sample_data(start_date, end_date, symbol)

    # ---- 写入缓存 ----
    df.to_csv(cache_file, encoding="utf-8-sig")
    print(f"[缓存写入] {cache_file}")

    return df


if __name__ == "__main__":
    df = fetch_stock_daily("600519", "20230101", "20250630")
    print(f"\n数据: {df.shape}, {df.index[0].date()} ~ {df.index[-1].date()}")
    print(df.head())
