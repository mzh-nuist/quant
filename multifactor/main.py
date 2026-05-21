# -*- coding: utf-8 -*-
"""
多因子选股回测 —— 命令行入口
-----------------------------
月度调仓，基于动量/低波/量能/RSI/规模五因子综合打分选股。

用法：
    python -m multifactor.main                          # 默认回测（50 只股票池，选 Top 20）
    python -m multifactor.main -n 10                    # 每期选 10 只
    python -m multifactor.main --start 20220101         # 指定回测区间
    python -m multifactor.main --top 15 --cash 500000   # 自定义参数
"""

import argparse
import sys
import os

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from multifactor.engine import run_backtest, print_result, _default_universe
from multifactor.selector import DEFAULT_WEIGHTS


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Multi-Factor Stock Selection Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
因子体系：
  动量(25%) + 低波动(20%) + 低异常换手(15%) + RSI反转(15%) + 小盘规模(25%)

示例：
  python -m multifactor.main
  python -m multifactor.main -n 10 --start 20210101
  python -m multifactor.main --top 15 --cash 2000000
        """,
    )
    parser.add_argument("-n", "--top", type=int, default=20,
                        help="每期持股数量（默认: 20）")
    parser.add_argument("--start", default="20200101",
                        help="起始日期 YYYYMMDD（默认: 20200101）")
    parser.add_argument("--end", default="20251231",
                        help="结束日期 YYYYMMDD（默认: 20251231）")
    parser.add_argument("--cash", type=float, default=1_000_000,
                        help="初始资金（默认: 1000000）")
    parser.add_argument("--commission", type=float, default=0.001,
                        help="佣金率（默认: 0.001）")
    parser.add_argument("--universe", default=None,
                        help="自定义股票池文件路径（每行一个代码），默认使用内置池")
    parser.add_argument("--refresh", action="store_true",
                        help="强制从网络刷新数据（跳过本地缓存）")
    parser.add_argument("--proxy", default="http://127.0.0.1:10808",
                        help="HTTP 代理地址（默认: http://127.0.0.1:10808）")
    return parser.parse_args()


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    # 股票池
    if args.universe and os.path.exists(args.universe):
        with open(args.universe, "r", encoding="utf-8") as f:
            universe = [line.strip() for line in f if line.strip()]
        print(f"从文件加载股票池: {len(universe)} 只")
    else:
        universe = _default_universe()

    # 执行回测
    result = run_backtest(
        universe=universe,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top,
        cash=args.cash,
        commission=args.commission,
        weights=DEFAULT_WEIGHTS,
        force_refresh=args.refresh,
        proxy=args.proxy,
        verbose=True,
    )

    # 打印结果
    print_result(result)
    print("\n回测完成。")


if __name__ == "__main__":
    main()
