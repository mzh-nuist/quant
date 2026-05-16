# -*- coding: utf-8 -*-
"""
回测引擎模块
-----------
封装 backtrader 的 Cerebro 引擎，提供：
  - 结构化回测结果（含 Sharpe、Sortino、Buy & Hold 基准对比）
  - 参数网格搜索优化
  - 供 CLI 和 GUI 共用
"""

from dataclasses import dataclass, field
from typing import Tuple, Union, List
import sys
import io
import math

import numpy as np
import backtrader as bt

from data_fetcher import fetch_stock_daily
from sma_crossover import SMACrossover


@dataclass
class BacktestResult:
    """回测结果数据结构。"""
    # 资金 & 收益
    initial_cash: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0          # 总收益率 %
    buy_hold_return: float = 0.0       # 买入持有收益率 %
    annual_return: float = 0.0         # 年化收益率 %
    annual_volatility: float = 0.0     # 年化波动率 %

    # 风险调整收益
    sharpe_ratio: float = 0.0          # 夏普比率（年化）
    sortino_ratio: float = 0.0         # 索提诺比率（年化）
    calmar_ratio: float = 0.0          # 卡尔马比率 = 年化收益 / |最大回撤|

    # 回撤
    max_drawdown: float = 0.0          # 最大回撤 %
    max_drawdown_days: int = 0         # 最长回撤持续天数

    # 交易统计
    trade_total: int = 0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0              # 胜率 %
    avg_win: float = 0.0               # 平均每笔盈利
    avg_loss: float = 0.0              # 平均每笔亏损
    profit_factor: float = 0.0         # 盈亏比 = 总盈利/总亏损

    # 元信息
    symbol: str = ""
    short_period: int = 10
    long_period: int = 30
    start_date: str = ""
    end_date: str = ""
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.0
    use_trailing: bool = True


@dataclass
class OptResult:
    """参数优化单条结果。"""
    short_period: int
    long_period: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    final_value: float


# ============================================================================
#  自定义 Sortino 分析器
# ============================================================================
class SortinoRatio(bt.Analyzer):
    """
    计算年化索提诺比率。
    Sortino = (年化收益率 - 无风险利率) / 下行标准差
    只对负收益（下行）计算标准差，比夏普更合理。
    """

    params = (("riskfreerate", 0.02), ("periods", 252))

    def __init__(self):
        self.returns = []

    def notify_fund(self, cash, value, fundvalue, shares):
        pass  # 非逐笔

    def next(self):
        """每个 bar 记录资金变化。"""
        self.returns.append(self.strategy.broker.getvalue())

    def stop(self):
        if len(self.returns) < 2:
            self.ratio = 0.0
            return

        # 日收益率序列
        vals = np.array(self.returns)
        daily = np.diff(vals) / vals[:-1]

        # 年化收益
        total = (vals[-1] / vals[0] - 1.0)
        periods = len(daily)
        if periods > 0 and total > -1:
            annual_return = (1 + total) ** (self.p.periods / periods) - 1.0
        else:
            annual_return = 0.0

        # 下行标准差（只取负收益）
        negative = daily[daily < 0]
        if len(negative) > 1:
            downside_std = np.std(negative, ddof=1) * math.sqrt(self.p.periods)
        else:
            downside_std = 0.0

        if downside_std > 0:
            self.ratio = (annual_return - self.p.riskfreerate) / downside_std
        else:
            self.ratio = 0.0

    def get_analysis(self):
        return {"sortino_ratio": self.ratio}


# ============================================================================
#  回测运行
# ============================================================================
def run_backtest(
    symbol: str = "600519",
    start_date: str = "20200101",
    end_date: str = "20251231",
    short_period: int = 10,
    long_period: int = 30,
    cash: float = 100_000.0,
    commission: float = 0.001,
    use_stop_loss: bool = False,
    stop_loss_pct: float = 0.05,
    use_trailing: bool = True,
    verbose: bool = True,
    return_cerebro: bool = False,
) -> Union[BacktestResult, Tuple[BacktestResult, bt.Cerebro]]:
    """
    执行 SMA 双均线交叉回测并返回结构化结果。

    返回
    ----
    BacktestResult
        包含 Sharpe、Sortino、Buy & Hold 基准对比等完整指标。
    """
    result = BacktestResult(
        symbol=symbol,
        short_period=short_period,
        long_period=long_period,
        start_date=start_date,
        end_date=end_date,
        use_stop_loss=use_stop_loss,
        stop_loss_pct=stop_loss_pct,
        use_trailing=use_trailing,
    )

    # ---- 获取数据 ----
    df = fetch_stock_daily(symbol, start_date, end_date, adjust="qfq")
    data = bt.feeds.PandasData(dataname=df)

    # ---- Buy & Hold 基准 ----
    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    result.buy_hold_return = (last_close / first_close - 1.0) * 100.0

    # 计算回测年数
    days = max((df.index[-1] - df.index[0]).days, 1)
    years = days / 365.25

    # ---- 初始化 Cerebro ----
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(
        SMACrossover,
        short_period=short_period,
        long_period=long_period,
        use_stop_loss=use_stop_loss,
        stop_loss_pct=stop_loss_pct,
        use_trailing=use_trailing,
    )
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    # ---- 分析器 ----
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="sharpe",
                        riskfreerate=0.02, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annreturn")
    cerebro.addanalyzer(SortinoRatio, _name="sortino", riskfreerate=0.02, periods=252)

    # ---- 运行 ----
    if not verbose:
        sys.stdout = io.StringIO()

    strat_results = cerebro.run()

    if not verbose:
        sys.stdout = sys.__stdout__

    strat = strat_results[0]

    # ---- 提取结果 ----
    result.initial_cash = cash
    result.final_value = cerebro.broker.getvalue()
    result.total_return = (result.final_value / cash - 1.0) * 100.0

    # 年化收益 & 波动率
    if years > 0 and result.total_return > -100:
        result.annual_return = (
            (result.final_value / cash) ** (1.0 / years) - 1.0
        ) * 100.0

    # Sharpe
    sh = strat.analyzers.sharpe.get_analysis()
    result.sharpe_ratio = sh.get("sharperatio", 0.0) or 0.0

    # Sortino
    so = strat.analyzers.sortino.get_analysis()
    result.sortino_ratio = so.get("sortino_ratio", 0.0) or 0.0

    # Annual volatility (from annual return analyzer)
    ar = strat.analyzers.annreturn.get_analysis()
    # ar is dict like {year: return%}. Compute std of annual returns.
    annual_rets = [v for v in ar.values() if isinstance(v, (int, float))]
    if len(annual_rets) > 1:
        result.annual_volatility = np.std(annual_rets, ddof=1)
    elif len(annual_rets) == 1:
        result.annual_volatility = 0.0

    # 最大回撤
    dd = strat.analyzers.drawdown.get_analysis()
    result.max_drawdown = dd.get("max", {}).get("drawdown", 0.0) or 0.0
    result.max_drawdown_days = dd.get("max", {}).get("len", 0) or 0

    # Calmar
    if result.max_drawdown > 0.01:
        result.calmar_ratio = result.annual_return / abs(result.max_drawdown)

    # 交易统计
    ta = strat.analyzers.trades.get_analysis()
    result.trade_total = ta.get("total", {}).get("total", 0)
    result.win_count = ta.get("won", {}).get("total", 0) or 0
    result.loss_count = ta.get("lost", {}).get("total", 0) or 0
    result.trade_count = result.win_count + result.loss_count

    if result.trade_count > 0:
        result.win_rate = (result.win_count / result.trade_count) * 100.0

    # 平均盈利/亏损 & 盈亏比
    avg_win_val = ta.get("won", {}).get("pnl", {}).get("average", 0.0) or 0.0
    avg_loss_val = ta.get("lost", {}).get("pnl", {}).get("average", 0.0) or 0.0
    result.avg_win = avg_win_val
    result.avg_loss = avg_loss_val

    total_won = ta.get("won", {}).get("pnl", {}).get("total", 0.0) or 0.0
    total_lost = abs(ta.get("lost", {}).get("pnl", {}).get("total", 0.0) or 0.0)
    if total_lost > 0:
        result.profit_factor = total_won / total_lost

    if return_cerebro:
        return result, cerebro
    return result


# ============================================================================
#  参数网格搜索
# ============================================================================
def run_optimization(
    symbol: str = "600519",
    start_date: str = "20200101",
    end_date: str = "20251231",
    cash: float = 100_000.0,
    commission: float = 0.001,
    use_stop_loss: bool = False,
    stop_loss_pct: float = 0.05,
    use_trailing: bool = True,
    short_range: tuple = (5, 55, 5),     # (start, stop, step)
    long_range: tuple = (20, 120, 10),
    top_n: int = 20,
) -> List[OptResult]:
    """
    网格搜索最优 SMA 参数组合，返回按夏普比率排序的 top N 结果。

    参数
    ----
    short_range : tuple
        (min, max_exclusive, step) 短线周期搜索范围。
    long_range : tuple
        (min, max_exclusive, step) 长线周期搜索范围。
    top_n : int
        返回前 N 个最优结果。

    返回
    ----
    list[OptResult]
        按总收益率降序排列的最优参数列表。
    """
    print(f"\n{'='*55}")
    print(f"  Grid Search Optimization")
    print(f"{'='*55}")
    print(f"  Symbol: {symbol}  |  Stop-loss: {use_stop_loss}")
    print(f"  Short SMA range: {short_range}")
    print(f"  Long SMA range:  {long_range}")
    print(f"  Total combos: "
          f"{((short_range[1]-short_range[0])//short_range[2]) * ((long_range[1]-long_range[0])//long_range[2])}")
    print(f"{'='*55}\n")

    # 获取数据（只拉一次）
    df = fetch_stock_daily(symbol, start_date, end_date, adjust="qfq")

    results = []
    short_start, short_end, short_step = short_range
    long_start, long_end, long_step = long_range

    for sp in range(short_start, short_end, short_step):
        for lp in range(long_start, long_end, long_step):
            if sp >= lp:
                continue

            cerebro = bt.Cerebro()
            cerebro.adddata(bt.feeds.PandasData(dataname=df))
            cerebro.addstrategy(
                SMACrossover,
                short_period=sp,
                long_period=lp,
                use_stop_loss=use_stop_loss,
                stop_loss_pct=stop_loss_pct,
                use_trailing=use_trailing,
            )
            cerebro.broker.setcash(cash)
            cerebro.broker.setcommission(commission=commission)
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="sharpe",
                               riskfreerate=0.02, timeframe=bt.TimeFrame.Days)

            # 静默运行
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            strat_results = cerebro.run()
            sys.stdout = old_stdout

            strat = strat_results[0]
            final_value = cerebro.broker.getvalue()
            total_return = (final_value / cash - 1.0) * 100.0

            dd = strat.analyzers.drawdown.get_analysis()
            max_dd = dd.get("max", {}).get("drawdown", 0.0) or 0.0

            sh = strat.analyzers.sharpe.get_analysis()
            sharpe = sh.get("sharperatio", 0.0) or 0.0

            ta = strat.analyzers.trades.get_analysis()
            wins = ta.get("won", {}).get("total", 0) or 0
            losses = ta.get("lost", {}).get("total", 0) or 0
            trade_count = wins + losses
            win_rate = (wins / trade_count * 100.0) if trade_count > 0 else 0.0

            results.append(OptResult(
                short_period=sp,
                long_period=lp,
                total_return=total_return,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                trade_count=trade_count,
                final_value=final_value,
            ))

    # 按总收益率降序排列
    results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
    return results[:top_n]


def print_optimization_results(results: List[OptResult]):
    """格式化打印网格搜索结果表格。"""
    print(f"\n{'Top':<5} {'Short':<7} {'Long':<7} {'Return%':<10} "
          f"{'Sharpe':<8} {'MaxDD%':<9} {'WinRate%':<10} {'Trades':<7} {'Final$':<12}")
    print("-" * 78)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r.short_period:<7} {r.long_period:<7} "
              f"{r.total_return:>8.2f}  {r.sharpe_ratio:>6.2f}  "
              f"{r.max_drawdown:>7.2f}  {r.win_rate:>8.2f}  "
              f"{r.trade_count:>5}  {r.final_value:>10,.2f}")


def print_result(result: BacktestResult):
    """在控制台打印完整回测结果。"""
    def _color(val, fmt_str, good_is_positive=True):
        s = fmt_str.format(val)
        return s

    print("\n" + "=" * 60)
    print("              Backtest Result Summary")
    print("=" * 60)
    print(f"  Symbol        : {result.symbol}")
    print(f"  SMA Periods   : {result.short_period} / {result.long_period}")
    print(f"  Date Range    : {result.start_date} ~ {result.end_date}")
    sl = f"Trail {result.stop_loss_pct*100:.0f}%" if result.use_stop_loss else "None"
    if result.use_stop_loss and not result.use_trailing:
        sl = f"Fixed {result.stop_loss_pct*100:.0f}%"
    print(f"  Stop Loss     : {sl}")
    print("-" * 60)
    print(f"  Initial Cash  : {result.initial_cash:>12,.2f}")
    print(f"  Final Value   : {result.final_value:>12,.2f}")
    print(f"  Total Return  : {result.total_return:>11.2f}%")
    print(f"  Buy & Hold    : {result.buy_hold_return:>11.2f}%")
    print(f"  Annual Return : {result.annual_return:>11.2f}%")
    print(f"  Ann.Volatility: {result.annual_volatility:>11.2f}%")
    print("-" * 60)
    print(f"  Sharpe Ratio  : {result.sharpe_ratio:>12.2f}")
    print(f"  Sortino Ratio : {result.sortino_ratio:>12.2f}")
    print(f"  Calmar Ratio  : {result.calmar_ratio:>12.2f}")
    print(f"  Max Drawdown  : {result.max_drawdown:>11.2f}%")
    print(f"  Max DD Days   : {result.max_drawdown_days:>11}")
    print("-" * 60)
    print(f"  Total Trades  : {result.trade_count:>11}")
    print(f"  Win / Loss    : {result.win_count} / {result.loss_count}")
    print(f"  Win Rate      : {result.win_rate:>11.2f}%")
    print(f"  Avg Win/Loss  : {result.avg_win:>8.2f} / {result.avg_loss:.2f}")
    print(f"  Profit Factor : {result.profit_factor:>12.2f}")
    print("=" * 60)
