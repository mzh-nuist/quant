# -*- coding: utf-8 -*-
"""
选股模块
-------
截面标准化与综合打分，选出每期最优股票组合。
"""

import numpy as np
from scipy import stats


# 默认因子权重（总和为 1.0）
DEFAULT_WEIGHTS = {
    "momentum":     0.25,   # 动量
    "volatility":   0.20,   # 低波动
    "volume_ratio": 0.15,   # 低异常换手
    "rsi":          0.15,   # RSI 反转
    "size":         0.25,   # 小盘
}


def zscore_normalize(
    factor_dict: dict[str, dict[str, float]],
    factor_name: str,
) -> dict[str, float]:
    """
    对某一因子全截面做 Z-Score 标准化。

    参数
    ----
    factor_dict : dict
        {symbol: {factor_name: value, ...}, ...}
    factor_name : str
        要标准化的因子名称。

    返回
    ----
    dict[str, float]
        {symbol: zscore, ...}，无法标准化的返回 0.0。
    """
    values = []
    symbols = list(factor_dict.keys())
    for sym in symbols:
        fd = factor_dict.get(sym)
        if fd and factor_name in fd:
            values.append(fd[factor_name])
        else:
            values.append(np.nan)

    arr = np.array(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 3:
        # 有效样本太少，全部给 0
        return {sym: 0.0 for sym in symbols}

    z = np.full(len(arr), np.nan)
    valid = arr[mask]
    z[mask] = stats.zscore(valid)

    # 将 NaN 填充为均值 0
    result = {}
    for i, sym in enumerate(symbols):
        result[sym] = float(z[i]) if not np.isnan(z[i]) else 0.0
    return result


def composite_score(
    all_factors: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    计算每只股票的综合因子得分。

    参数
    ----
    all_factors : dict
        {symbol: {factor_name: value, ...}, ...}
    weights : dict | None
        各因子权重，默认使用 DEFAULT_WEIGHTS。

    返回
    ----
    dict[str, float]
        {symbol: composite_score, ...}，按得分降序排列即为选股优先级。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # 先对每个因子截面标准化
    normalized = {}
    for factor_name in weights:
        normalized[factor_name] = zscore_normalize(all_factors, factor_name)

    # 加权合成
    scores = {}
    for symbol in all_factors:
        score = 0.0
        for factor_name, w in weights.items():
            if factor_name in normalized:
                score += w * normalized[factor_name].get(symbol, 0.0)
        scores[symbol] = score

    return scores


def select_top(
    scores: dict[str, float],
    top_n: int = 20,
) -> list[str]:
    """
    按综合得分选出 Top N 股票。

    参数
    ----
    scores : dict
        {symbol: score, ...}
    top_n : int
        选取数量。

    返回
    ----
    list[str]
        得分最高的股票代码列表。
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [sym for sym, s in ranked[:top_n] if not np.isnan(s)]


def select_top_weighted(
    scores: dict[str, float],
    top_n: int = 20,
) -> dict[str, float]:
    """
    按综合得分选出 Top N 股票，并以得分为基础分配权重（等权）。

    参数
    ----
    scores : dict
        {symbol: score, ...}
    top_n : int
        选取数量。

    返回
    ----
    dict[str, float]
        {symbol: weight, ...}，权重总和为 1.0。
    """
    selected = select_top(scores, top_n)
    if not selected:
        return {}
    w = 1.0 / len(selected)
    return {sym: w for sym in selected}
