# -*- coding: utf-8 -*-
"""
SMA 双均线交叉回测 —— 命令行入口
---------------------------------
功能：
    1. 单次回测（含 Sharpe / Sortino / Buy & Hold 基准对比）
    2. 参数网格搜索优化 (--optimize)
    3. 可选固定/移动止损

用法：
    python main.py                              # 默认回测
    python main.py --stop-loss                   # 启用移动止损
    python main.py --stop-loss --fixed           # 启用固定止损
    python main.py --optimize                    # 参数网格搜索
    python main.py --optimize --stop-loss        # 带止损的参数搜索
"""

import argparse
import sys
import os

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from dual_ma.engine import (
    run_backtest, run_optimization,
    print_result, print_optimization_results,
)


def configure_matplotlib():
    """Set up matplotlib style for clean charts."""
    try:
        mplstyle.use("seaborn-v0_8-darkgrid")
    except Exception:
        try:
            mplstyle.use("seaborn-v0_8")
        except Exception:
            pass
    plt.rcParams.update({
        "figure.figsize": (16, 10),
        "figure.dpi": 120,
        "lines.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.edgecolor": "#333333",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.color": "#AAAAAA",
        "legend.fontsize": 10,
    })


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="SMA Crossover Strategy Backtest & Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                           # Default backtest
  python main.py --stop-loss                                # + trailing stop 5%
  python main.py --stop-loss --fixed --stop-loss-pct 0.08   # Fixed stop 8%
  python main.py --optimize                                 # Grid search
  python main.py -c 000001 -s 10 -l 50 --stop-loss          # Custom
        """,
    )
    parser.add_argument("-c", "--code", default="600519",
                        help="Stock ticker (default: 600519)")
    parser.add_argument("-s", "--short", type=int, default=10,
                        help="Short SMA period (default: 10)")
    parser.add_argument("-l", "--long", type=int, default=30,
                        help="Long SMA period (default: 30)")
    parser.add_argument("--start", default="20200101",
                        help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20251231",
                        help="End date YYYYMMDD")
    parser.add_argument("--cash", type=float, default=100000,
                        help="Initial capital (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="Commission rate (default: 0.001)")

    # 止损选项
    parser.add_argument("--stop-loss", action="store_true",
                        help="Enable stop-loss")
    parser.add_argument("--stop-loss-pct", type=float, default=0.05,
                        help="Stop-loss percentage (default: 0.05 = 5%%)")
    parser.add_argument("--fixed", action="store_true",
                        help="Use fixed stop-loss (default: trailing)")

    # 图表
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable chart output")
    parser.add_argument("-o", "--output", default=None,
                        help="Chart output path")

    # 数据刷新
    parser.add_argument("--refresh", action="store_true",
                        help="强制从网络刷新数据（跳过本地缓存）")

    # 优化
    parser.add_argument("--optimize", action="store_true",
                        help="Run grid search optimization")
    return parser.parse_args()


def save_chart(cerebro, output_path: str = None, code: str = "600519",
               short: int = 10, long: int = 30,
               start: str = "20200101", end: str = "20251231"):
    """Generate and save backtest chart."""
    print("\nGenerating chart, please wait...")
    figure = cerebro.plot(
        style="candlestick",
        barup="red", bardown="green",
        volup="red", voldown="green",
        grid=True, plotdist=0.06, dpi=120,
    )

    if not output_path:
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join(
            "results",
            f"{code}_SMA{short}-{long}_{start}-{end}.png"
        )

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    for fig_group in figure:
        for i, fig in enumerate(fig_group):
            if hasattr(fig, "savefig"):
                base, ext = os.path.splitext(output_path)
                fname = f"{base}_{i}{ext}" if len(fig_group) > 1 else output_path
                fig.savefig(fname, bbox_inches="tight", facecolor="white")
                print(f"  Chart saved: {os.path.abspath(fname)}")


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    configure_matplotlib()
    args = parse_args()

    # ================================================================
    #  优化模式：网格搜索
    # ================================================================
    if args.optimize:
        results = run_optimization(
            symbol=args.code,
            start_date=args.start,
            end_date=args.end,
            cash=args.cash,
            commission=args.commission,
            use_stop_loss=args.stop_loss,
            stop_loss_pct=args.stop_loss_pct,
            use_trailing=not args.fixed,
            force_refresh=args.refresh,
        )
        print_optimization_results(results)
        return

    # ================================================================
    #  单次回测模式
    # ================================================================
    result, cerebro = run_backtest(
        symbol=args.code,
        start_date=args.start,
        end_date=args.end,
        short_period=args.short,
        long_period=args.long,
        cash=args.cash,
        commission=args.commission,
        use_stop_loss=args.stop_loss,
        stop_loss_pct=args.stop_loss_pct,
        use_trailing=not args.fixed,
        verbose=True,
        return_cerebro=True,
        force_refresh=args.refresh,
    )

    print_result(result)

    if not args.no_plot:
        save_chart(cerebro, args.output, args.code,
                   args.short, args.long, args.start, args.end)

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
