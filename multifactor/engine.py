# -*- coding: utf-8 -*-
"""
多因子回测引擎
-------------
向量化回测：每月根据因子得分选股，等权持仓，计算组合绩效。

提供：
  - 结构化回测结果（Sharpe / Sortino / 最大回撤 / 胜率等）
  - 基准对比（等权全市场组合）
  - 格式化输出
"""

from dataclasses import dataclass
import sys
import os as _os
import math
import time

import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path 中
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_fetcher import fetch_stock_daily
from multifactor.factors import compute_all_factors
from multifactor.selector import composite_score, select_top, DEFAULT_WEIGHTS


# ============================================================================
#  结果数据结构
# ============================================================================
@dataclass
class MultifactorResult:
    """多因子选股回测结果。"""
    # 资金 & 收益
    initial_cash: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0          # 总收益率 %
    annual_return: float = 0.0          # 年化收益率 %
    annual_volatility: float = 0.0      # 年化波动率 %
    benchmark_return: float = 0.0       # 等权基准收益率 %

    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 回撤
    max_drawdown: float = 0.0
    max_drawdown_days: int = 0

    # 月度统计
    monthly_win_rate: float = 0.0       # 月度胜率 %
    best_month: float = 0.0             # 最佳月度收益 %
    worst_month: float = 0.0            # 最差月度收益 %

    # 元信息
    universe_size: int = 0
    top_n: int = 20
    rebalance_freq: str = "monthly"
    start_date: str = ""
    end_date: str = ""
    selected_count: int = 0             # 历史选股总次数


# ============================================================================
#  回测引擎
# ============================================================================
def run_backtest(
    universe: list[str] | None = None,
    start_date: str = "20200101",
    end_date: str = "20251231",
    top_n: int = 20,
    cash: float = 1_000_000.0,
    commission: float = 0.001,
    weights: dict[str, float] | None = None,
    min_data_days: int = 120,
    force_refresh: bool = False,
    proxy: str | None = None,
    verbose: bool = True,
) -> MultifactorResult:
    """
    执行多因子选股回测。

    参数
    ----
    proxy : str | None
        HTTP 代理地址，如 "http://127.0.0.1:10808"。
        是否打印进度。

    返回
    ----
    MultifactorResult
    """
    if universe is None:
        universe = _default_universe()
    if weights is None:
        weights = DEFAULT_WEIGHTS

    result = MultifactorResult(
        initial_cash=cash,
        top_n=top_n,
        start_date=start_date,
        end_date=end_date,
        universe_size=len(universe),
    )

    # ---- Step 1: 获取全部股票数据 ----
    if verbose:
        print(f"\n{'='*55}")
        print(f"  多因子选股回测")
        print(f"{'='*55}")
        print(f"  股票池: {len(universe)} 只")
        print(f"  回测区间: {start_date} ~ {end_date}")
        print(f"  每期持股: {top_n} 只")
        print(f"  调仓频率: 月度")
        print(f"{'='*55}\n")
        print("正在获取行情数据...")

    close_dict = {}
    volume_dict = {}
    valid_symbols = []

    for i, sym in enumerate(universe):
        # 股票间延时，避免触发东方财富反爬
        if i > 0:
            time.sleep(1.0)
        try:
            df = fetch_stock_daily(sym, start_date, end_date,
                                   force_refresh=force_refresh, proxy=proxy)
            if len(df) >= min_data_days:
                close_dict[sym] = df["close"]
                volume_dict[sym] = df["volume"]
                valid_symbols.append(sym)
        except Exception:
            pass

    if verbose:
        print(f"有效股票: {len(valid_symbols)} / {len(universe)}")

    if len(valid_symbols) < top_n:
        raise ValueError(
            f"有效股票数量({len(valid_symbols)})少于持股数量({top_n})，"
            f"请扩大股票池或缩短回测区间。"
        )

    # ---- Step 2: 构建对齐的价格/收益率矩阵 ----
    prices_df = pd.DataFrame(close_dict).sort_index()
    volumes_df = pd.DataFrame(volume_dict).sort_index()
    # 前向填充停牌日
    prices_df = prices_df.ffill()
    volumes_df = volumes_df.ffill()
    # 日收益率
    returns_df = prices_df.pct_change().dropna()

    # 交易日期
    all_dates = returns_df.index
    if len(all_dates) < 60:
        raise ValueError("有效交易日不足 60 天，无法回测。")

    # ---- Step 3: 确定调仓日期（每月第一个交易日） ----
    month_groups = returns_df.index.to_period("M")
    rebalance_set = set()
    for month_val in month_groups.unique():
        month_dates = all_dates[month_groups == month_val]
        if len(month_dates) > 0:
            rebalance_set.add(month_dates[0])

    # ---- Step 4: 向量化回测（基于收益率，避免 share/cash 跟踪问题） ----
    # 每月初选股，当月等权持有，计算组合日收益率
    # 调整日期格式以支持筛选期间的收益率
    start_dt = pd.Timestamp(all_dates[0])
    end_dt = pd.Timestamp(all_dates[-1])

    # 调仓日期（每月第一个交易日）
    rebalance_dates = sorted(rebalance_set)
    if all_dates[0] not in rebalance_set:
        rebalance_dates.insert(0, all_dates[0])

    portfolio_value = cash
    daily_values = [(all_dates[0] - pd.Timedelta(days=1), cash)]

    # 基准：等权持有全部股票
    benchmark_ret = returns_df.mean(axis=1)

    rebalance_count = 0

    # 对每个调仓期间的收益率向量化计算
    for ri in range(len(rebalance_dates)):
        rb_date = rebalance_dates[ri]
        # 下一调仓日（不含）
        if ri + 1 < len(rebalance_dates):
            next_rb_date = rebalance_dates[ri + 1]
        else:
            next_rb_date = all_dates[-1] + pd.Timedelta(days=1)

        # 找到 rb_date 在 all_dates 中的索引
        try:
            rb_idx = all_dates.get_loc(rb_date)
        except KeyError:
            continue
        # 确保 rb_idx 是整数
        if isinstance(rb_idx, slice):
            rb_idx = rb_idx.start
        if isinstance(rb_idx, np.ndarray):
            rb_idx = int(rb_idx[0])

        # ---- 用 rb_date 之前的数据计算因子（避免未来信息） ----
        cutoff_idx = max(rb_idx - 1, 0)
        cutoff_date = all_dates[cutoff_idx]
        prices_slice = prices_df.loc[:cutoff_date]
        volumes_slice = volumes_df.loc[:cutoff_date]

        all_factors = {}
        for sym in valid_symbols:
            if sym not in prices_slice.columns or sym not in volumes_slice.columns:
                continue
            c = prices_slice[sym].dropna()
            v = volumes_slice[sym].dropna()
            if len(c) < 20:
                continue
            fac = compute_all_factors(c, v)
            if fac is not None:
                all_factors[sym] = fac

        selected = []
        if len(all_factors) >= top_n:
            scores = composite_score(all_factors, weights)
            selected = select_top(scores, top_n)
            rebalance_count += 1

        # ---- 获取本期收益率 ----
        mask = (returns_df.index >= rb_date) & (returns_df.index < next_rb_date)
        period_dates = returns_df.index[mask]

        if len(period_dates) == 0:
            continue

        if selected:
            # 等权组合日收益率 = 选中股票的平均日收益率
            avail = [s for s in selected if s in returns_df.columns]
            if avail:
                period_rets = returns_df.loc[period_dates, avail].mean(axis=1)
            else:
                period_rets = pd.Series(0.0, index=period_dates)
        else:
            # 无持仓期间，收益率为 0
            period_rets = pd.Series(0.0, index=period_dates)

        # 扣除换手佣金（简化：每月 turnover ≈ 20% 单向，即 0.2 * commission）
        turnover_cost = 0.2 * commission if selected else 0.0

        # 逐日复利
        for d, ret in period_rets.items():
            portfolio_value *= (1.0 + float(ret) - turnover_cost / len(period_dates) if len(period_dates) > 0 else 0.0)
            daily_values.append((d, portfolio_value))

    # 转为 Series
    if len(daily_values) <= 1:
        raise ValueError("回测期间无有效交易日。")

    dv_dates = [d for d, _ in daily_values]
    dv_vals = [v for _, v in daily_values]
    dv_series = pd.Series(dv_vals, index=dv_dates)

    # ---- Step 5: 计算绩效指标 ----
    # 组合日收益率
    port_returns = dv_series.pct_change().dropna()

    result.final_value = float(dv_series.iloc[-1])
    result.total_return = float(dv_series.iloc[-1] / cash - 1.0) * 100.0
    result.selected_count = rebalance_count

    # 年化指标
    days = max((dv_series.index[-1] - dv_series.index[0]).days, 1)
    years = days / 365.25

    if years > 0 and result.total_return > -100:
        result.annual_return = ((dv_series.iloc[-1] / cash) ** (1.0 / years) - 1.0) * 100.0

    if len(port_returns) > 1:
        result.annual_volatility = float(port_returns.std() * np.sqrt(252)) * 100.0

    # Sharpe
    rf_daily = 0.02 / 252
    if len(port_returns) > 1 and port_returns.std() > 0:
        excess = port_returns - rf_daily
        result.sharpe_ratio = float(excess.mean() / port_returns.std() * np.sqrt(252))

    # Sortino
    if len(port_returns) > 1:
        downside = port_returns[port_returns < 0]
        if len(downside) > 1:
            downside_std = float(downside.std() * np.sqrt(252))
            if downside_std > 0:
                result.sortino_ratio = (result.annual_return / 100 - 0.02) / downside_std

    # 最大回撤
    cummax = dv_series.cummax()
    drawdown = (dv_series - cummax) / cummax * 100
    result.max_drawdown = float(drawdown.min())

    # 回撤持续天数
    dd_series = drawdown < 0
    max_dd_days = 0
    current_days = 0
    for is_dd in dd_series:
        if is_dd:
            current_days += 1
            max_dd_days = max(max_dd_days, current_days)
        else:
            current_days = 0
    result.max_drawdown_days = max_dd_days

    # Calmar
    if abs(result.max_drawdown) > 0.01:
        result.calmar_ratio = result.annual_return / abs(result.max_drawdown)

    # 基准：等权全市场组合
    bench_cum = (1 + benchmark_ret).cumprod()
    result.benchmark_return = float(bench_cum.iloc[-1] - 1) * 100.0

    # 月度统计
    monthly_rets = port_returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    ).dropna()
    if len(monthly_rets) > 0:
        result.monthly_win_rate = float((monthly_rets > 0).mean() * 100)
        result.best_month = float(monthly_rets.max() * 100)
        result.worst_month = float(monthly_rets.min() * 100)

    if verbose:
        print(f"回测完成 | 调仓 {rebalance_count} 次 | 最终资金 {result.final_value:,.0f}\n")

    return result


# ============================================================================
#  格式化输出
# ============================================================================
def print_result(result: MultifactorResult):
    """在控制台打印完整回测结果。"""
    print("\n" + "=" * 60)
    print("           多因子选股回测结果")
    print("=" * 60)
    print(f"  回测区间     : {result.start_date} ~ {result.end_date}")
    print(f"  股票池规模   : {result.universe_size} 只")
    print(f"  每期持股     : {result.top_n} 只（{result.rebalance_freq} 调仓）")
    print(f"  历史调仓次数 : {result.selected_count}")
    print("-" * 60)
    print(f"  初始资金     : {result.initial_cash:>12,.0f}")
    print(f"  最终资金     : {result.final_value:>12,.0f}")
    print(f"  总收益率     : {result.total_return:>11.2f}%")
    print(f"  等权基准     : {result.benchmark_return:>11.2f}%")
    print(f"  年化收益率   : {result.annual_return:>11.2f}%")
    print(f"  年化波动率   : {result.annual_volatility:>11.2f}%")
    print("-" * 60)
    print(f"  夏普比率     : {result.sharpe_ratio:>12.2f}")
    print(f"  索提诺比率   : {result.sortino_ratio:>12.2f}")
    print(f"  卡尔马比率   : {result.calmar_ratio:>12.2f}")
    print(f"  最大回撤     : {result.max_drawdown:>11.2f}%")
    print(f"  最长回撤天数 : {result.max_drawdown_days:>11}")
    print("-" * 60)
    print(f"  月度胜率     : {result.monthly_win_rate:>11.2f}%")
    print(f"  最佳月度     : {result.best_month:>11.2f}%")
    print(f"  最差月度     : {result.worst_month:>11.2f}%")
    print("=" * 60)


# ============================================================================
#  默认股票池（沪深 300 核心成分股）
# ============================================================================
def _default_universe() -> list[str]:
    """返回内置默认股票池。"""
    return [
        "600519", "000001", "000002", "000858", "002594", "300750",
        "601318", "600036", "000333", "600276", "601888", "002415",
        "600900", "600030", "000651", "601398", "601857", "600028",
        "002475", "300059", "688981", "600809", "000568", "002714",
        "601012", "300274", "600585", "000725", "002230", "300124",
        "601166", "600000", "601899", "600050", "601728", "600941",
        "002352", "000338", "600031", "002304", "000625", "601238",
        "600104", "002466", "002460", "300014", "600438", "600690",
        "000100", "002371", "603259", "300015", "300122", "000661",
        "600196", "600887", "002142", "600048", "601668", "600309",
        "002049", "603986", "688111", "002236", "000063", "600570",
        "300033", "002410", "600588", "000977", "601088", "600019",
    ]
