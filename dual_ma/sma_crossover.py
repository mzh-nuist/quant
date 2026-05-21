# -*- coding: utf-8 -*-
"""
SMA 双均线交叉策略（含止损）
-----------------------------
当短期均线上穿长期均线时（金叉），买入；
当短期均线下穿长期均线时（死叉），卖出。
支持固定百分比止损和移动止损（trailing stop），保护资金曲线。
"""

import backtrader as bt


class SMACrossover(bt.Strategy):
    """
    双均线交叉策略，可选止损。

    参数
    ----
    short_period : int
        短期均线周期，默认 10。
    long_period : int
        长期均线周期，默认 30。
    use_stop_loss : bool
        是否启用止损，默认 False。
    stop_loss_pct : float
        止损百分比（0.05 = 5%），默认 0.05。
    use_trailing : bool
        True = 移动止损（追踪最高价回撤），False = 固定止损（从入场价计算）。
    """

    params = (
        ("short_period", 10),
        ("long_period", 30),
        ("use_stop_loss", False),
        ("stop_loss_pct", 0.05),
        ("use_trailing", True),
    )

    def __init__(self):
        # ---- 均线 ----
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.short_period
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.long_period
        )
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

        # ---- 订单与统计 ----
        self.order = None
        self.trade_count = 0

        # ---- 止损跟踪 ----
        self.entry_price = 0.0            # 入场价
        self.highest_since_entry = 0.0    # 持仓期间最高价（移动止损用）

    def log(self, txt: str, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"[{dt.isoformat()}] {txt}")

    # ------------------------------------------------------------------
    #  订单通知
    # ------------------------------------------------------------------
    def notify_order(self, order: bt.Order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"买入执行 价格={order.executed.price:.2f} 数量={order.executed.size}")
                # 记录入场价，初始化移动止损最高价
                self.entry_price = order.executed.price
                self.highest_since_entry = order.executed.price
            else:
                self.log(f"卖出执行 价格={order.executed.price:.2f} 数量={order.executed.size}")
                # 清仓后重置止损状态
                self.entry_price = 0.0
                self.highest_since_entry = 0.0
            self.trade_count += 1
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单异常! 状态={order.getstatusname()}")

        self.order = None

    # ------------------------------------------------------------------
    #  交易通知
    # ------------------------------------------------------------------
    def notify_trade(self, trade: bt.Trade):
        if trade.isclosed:
            reason = "信号"
            if trade.historyon and trade.history:
                pass  # 可在此扩展退出原因记录
            self.log(
                f"交易完成 | "
                f"毛利={trade.pnl:.2f} | "
                f"净利={trade.pnlcomm:.2f} | "
                f"持仓天数={trade.barlen}"
            )

    # ------------------------------------------------------------------
    #  核心决策
    # ------------------------------------------------------------------
    def next(self):
        # 有未成交订单时不下新单
        if self.order:
            # 仍有订单但已持仓：更新移动止损最高价
            if self.position.size > 0:
                self.highest_since_entry = max(
                    self.highest_since_entry, self.data.high[0]
                )
            return

        # ---- 持仓状态：检查止损 ----
        if self.position.size > 0:
            # 更新移动止损参考价
            self.highest_since_entry = max(
                self.highest_since_entry, self.data.high[0]
            )

            if self.params.use_stop_loss:
                if self._stop_triggered():
                    self.order = self.close()
                    if self.params.use_trailing:
                        self.log(
                            f"移动止损触发! 最高={self.highest_since_entry:.2f} "
                            f"当前={self.data.close[0]:.2f} "
                            f"回撤>{self.params.stop_loss_pct*100:.1f}%"
                        )
                    else:
                        self.log(
                            f"固定止损触发! 入场={self.entry_price:.2f} "
                            f"当前={self.data.close[0]:.2f} "
                            f"跌幅>{self.params.stop_loss_pct*100:.1f}%"
                        )
                    return  # 止损优先，不再检查交叉信号

            # ---- 死叉卖出 ----
            if self.crossover < 0:
                self.order = self.close()
                self.log(
                    f"死叉信号! SMA{self.params.short_period} 下穿 "
                    f"SMA{self.params.long_period}，卖出"
                )

        # ---- 空仓状态：金叉买入 ----
        elif self.crossover > 0:
            available_cash = self.broker.getcash() * 0.92
            price = self.data.close[0]
            size = int(available_cash / price)
            if size > 0:
                self.order = self.buy(size=size)
                self.log(
                    f"金叉信号! SMA{self.params.short_period} 上穿 "
                    f"SMA{self.params.long_period}，买入 {size} 股 @ {price:.2f}"
                )

    def _stop_triggered(self) -> bool:
        """判断当前持仓是否触发止损条件。"""
        close = self.data.close[0]

        if self.params.use_trailing:
            # 移动止损：从持仓期间最高点回撤超过 stop_loss_pct
            if self.highest_since_entry > 0:
                drawdown = (self.highest_since_entry - close) / self.highest_since_entry
                return drawdown >= self.params.stop_loss_pct
        else:
            # 固定止损：从入场价下跌超过 stop_loss_pct
            if self.entry_price > 0:
                loss = (self.entry_price - close) / self.entry_price
                return loss >= self.params.stop_loss_pct

        return False

    # ------------------------------------------------------------------
    #  回测结束汇总
    # ------------------------------------------------------------------
    def stop(self):
        final_value = self.broker.getvalue()
        initial_value = self.broker.startingcash
        total_return = (final_value / initial_value - 1) * 100

        sl_status = f"止损={self.params.stop_loss_pct*100:.0f}%" if self.params.use_stop_loss else "无止损"
        trail = "(移动)" if self.params.use_trailing and self.params.use_stop_loss else ""

        print("\n" + "=" * 55)
        print("                策略回测结果汇总")
        print("=" * 55)
        print(f"  策略名称     : SMA 双均线交叉 "
              f"({self.params.short_period}/{self.params.long_period})")
        print(f"  风控         : {sl_status}{trail}")
        print(f"  初始资金     : {initial_value:,.2f} 元")
        print(f"  最终资金     : {final_value:,.2f} 元")
        print(f"  总收益率     : {total_return:.2f} %")
        print(f"  成交笔数     : {self.trade_count}")
        print("=" * 55)
