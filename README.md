# Quant

A 股量化策略回测框架，基于 [backtrader](https://www.backtrader.com/)，数据源 [Yahoo Finance](https://pypi.org/project/yfinance/) + [akshare](https://github.com/akfamily/akshare) + 本地 CSV 缓存。

🔬 **因子研究项目：** [quant-intern](https://github.com/mzh-nuist/quant-intern) — A 股多市值层因子系统性研究（微盘→超大市值），七层交叉验证框架，聚宽实盘策略

---

## 策略模块

| 模块 | 类型 | 说明 | 文档 |
|------|------|------|------|
| [dual_ma/](dual_ma/) | SMA 双均线交叉 | 金叉买入、死叉卖出，支持止损，CLI + GUI | [README](dual_ma/README.md) |
| [multifactor/](multifactor/) | 多因子选股 | 月度调仓，五因子打分，等权组合 | [README](multifactor/README.md) |
| [etf_rotation/](etf_rotation/) | ETF 动量轮动 | 双周调仓，趋势过滤+波动率校准+截面动量+相关性去重 | 见下方 |

---

## 项目结构

```
qu/
├── data_fetcher.py            # 数据获取模块（Yahoo Finance + CSV 缓存，共享）
├── data/                      # 行情数据缓存目录
├── results/                   # 回测图表输出目录
│
├── dual_ma/                   # 双均线交叉策略
│   ├── sma_crossover.py       #   策略核心：金叉/死叉信号、止损逻辑
│   ├── engine.py              #   回测引擎：Sharpe/Sortino、网格搜索
│   ├── main.py                #   命令行入口
│   ├── gui.py                 #   Tkinter 图形界面
│   └── README.md              #   策略完整文档
│
├── multifactor/               # 多因子选股策略
│   ├── factors.py             #   因子计算（动量/低波/量能/RSI/规模）
│   ├── selector.py            #   截面标准化与选股
│   ├── engine.py              #   向量化回测引擎
│   ├── main.py                #   命令行入口
│   └── gui.py                 #   Tkinter 图形界面
│
├── etf_rotation/              # ETF 动量轮动策略
│   ├── strategy.py            #   策略核心：趋势过滤+截面动量+波动率校准+相关性去重
│   ├── backtest.py            #   事件驱动回测引擎（手续费+滑点+参数扫描）
│   ├── data_fetcher.py        #   ETF 数据获取（akshare，21只跨资产ETF）
│   ├── main.py                #   命令行入口，支持参数扫描
│   └── data/                  #   ETF 行情缓存
│
└── README.md                  # 本文件（项目概览）
```

---

## 环境准备

```bash
conda create -n quant python=3.12 -y
conda activate quant
pip install backtrader pandas matplotlib yfinance numpy scipy akshare
```

---

## 快速开始

### 双均线交叉策略

```bash
python dual_ma/main.py                              # 默认回测（茅台 600519）
python dual_ma/main.py -c 000001 -s 5 -l 20         # 自定义参数
python dual_ma/main.py --stop-loss                   # 启用移动止损
python dual_ma/main.py --optimize                    # 参数网格搜索
python dual_ma/main.py --refresh                     # 强制刷新数据
python dual_ma/gui.py                                # 图形界面
```

详细文档：[dual_ma/README.md](dual_ma/README.md)

### 多因子选股策略

```bash
python -m multifactor.main                           # 默认回测（72 只，选 Top 20）
python -m multifactor.main -n 10 --start 20210101    # 自定义参数
python -m multifactor.gui                            # 图形界面
```

详细文档：[multifactor/README.md](multifactor/README.md)

### ETF 动量轮动策略

```bash
python etf_rotation/main.py                          # 默认回测（21只ETF，5持仓）
python etf_rotation/main.py --sweep                  # 参数网格扫描
```

---

## 免责声明

本项目仅供学习和研究使用，**不构成任何投资建议**。量化回测基于历史数据，过去的表现不代表未来的收益。实盘交易有风险，请谨慎决策，盈亏自负。
