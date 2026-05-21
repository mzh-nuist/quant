# SMA 双均线交叉量化回测策略

基于 [backtrader](https://www.backtrader.com/) 的趋势跟踪策略，使用 [Yahoo Finance](https://pypi.org/project/yfinance/) 获取 A 股真实行情数据（需代理），GUI 股票搜索支持 akshare 实时查询。金叉买入、死叉卖出，支持固定/移动止损，提供 CLI 和 GUI 两种操作方式。

---

## 目录

- [策略原理](#策略原理)
- [数学公式](#数学公式)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [GUI 界面](#gui-界面)
- [止损机制](#止损机制)
- [回测结果解读](#回测结果解读)
- [参数优化](#参数优化)
- [代码说明](#代码说明)
- [策略局限性](#策略局限性)
- [常见问题](#常见问题)
- [免责声明](#免责声明)

---

## 策略原理

### 什么是双均线交叉？

双均线交叉（SMA Crossover）是最经典的趋势跟踪策略之一，核心逻辑非常简单：

| 信号 | 条件 | 含义 |
|------|------|------|
| **金叉** (Golden Cross) | 短期均线 **上穿** 长期均线 | 短期趋势转强，看涨 → **买入** |
| **死叉** (Death Cross) | 短期均线 **下穿** 长期均线 | 短期趋势转弱，看跌 → **卖出** |

```
        金叉 ↑                         死叉 ↓
          \  短期均线                     \  长期均线
     ──────\──── 长期均线          ────────\──── 短期均线
            \                              \
     ────────\─────────            ──────────\───────
              \                              \
```

- **短期均线**对价格变化更敏感，反应更快
- **长期均线**更平滑，代表中长期趋势
- 两线交叉的瞬间，意味着短期方向的改变正在确认

### 为什么均线交叉有效？

1. **趋势跟踪**：均线天然具有平滑价格波动的特性，能过滤市场噪音。
2. **惯性原理**：市场趋势一旦形成往往具有一定的持续性，均线能捕捉到趋势的主升/主跌段。
3. **心理锚定**：均线被大量交易者关注，本身具有自我实现的特性——当足够多的人在同一位置买卖，该位置就真的成了支撑/阻力。

---

## 数学公式

简单移动平均（SMA）的计算公式：

$$
\text{SMA}_{n}(t) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}
$$

其中 $P_t$ 为第 $t$ 日的收盘价，$n$ 为均线周期。

交叉信号由 CrossOver 指标判定：

$$
\text{CrossOver}(t) =
\begin{cases}
+1 & \text{if } \text{SMA}_{\text{short}}(t-1) \leq \text{SMA}_{\text{long}}(t-1) \text{ 且 } \text{SMA}_{\text{short}}(t) > \text{SMA}_{\text{long}}(t) \\
-1 & \text{if } \text{SMA}_{\text{short}}(t-1) \geq \text{SMA}_{\text{long}}(t-1) \text{ 且 } \text{SMA}_{\text{short}}(t) < \text{SMA}_{\text{long}}(t) \\
0  & \text{otherwise}
\end{cases}
$$

---

## 项目结构

```
dual_ma/
├── sma_crossover.py          # 策略核心类（继承 bt.Strategy）
├── engine.py                 # 回测引擎（Sharpe/Sortino 指标、网格搜索优化）
├── main.py                   # 命令行入口（CLI）
├── gui.py                    # Tkinter 图形界面（GUI）
└── README.md                 # 本文件
```

依赖的共享模块（位于上级目录）：

```
../
├── data_fetcher.py           # 数据获取（Yahoo Finance 在线 + CSV 本地缓存 + 模拟回退）
├── data/                     # 行情数据缓存目录
└── results/                  # 回测图表输出目录
```

| 文件 | 职责 |
|------|------|
| `sma_crossover.py` | 继承 `bt.Strategy`，定义均线计算、信号生成、订单管理、止损逻辑 |
| `engine.py` | 封装 Cerebro 引擎，提供结构化结果（`BacktestResult`）、参数优化（`run_optimization`） |
| `main.py` | argparse 命令行参数解析，串联数据获取 → 回测 → 图表输出全流程 |
| `gui.py` | Tkinter 图形界面，股票模糊搜索、参数配置、结果面板、网格搜索弹窗 |

---

## 环境准备

### 1. 创建并激活 Conda 环境

```bash
conda create -n quant python=3.12 -y
conda activate quant
```

### 2. 安装依赖

```bash
pip install backtrader pandas matplotlib yfinance numpy
```

### 3. 验证安装

```bash
python -c "import backtrader; import yfinance; import pandas; import matplotlib; print('所有依赖就绪')"
```

---

## 快速开始

### 默认回测（贵州茅台 600519，SMA 10/30）

```bash
conda activate quant
python dual_ma/main.py
```

程序会自动：
1. 从本地 CSV 缓存加载贵州茅台 2020~2025 年日线数据（首次需 --refresh 联网获取）
2. 使用 10 日均线和 30 日均线运行交叉策略回测
3. 控制台输出每笔交易日志和最终收益率汇总
4. 弹窗显示 K 线图 + 成交量 + 两条均线叠加 + 买卖标记

### 常用命令

```bash
# 平安银行 000001，5/20 均线交叉
python dual_ma/main.py -c 000001 -s 5 -l 20

# 比亚迪 002594，20/60 均线，50 万初始资金
python dual_ma/main.py -c 002594 -s 20 -l 60 --cash 500000

# 指定回测日期范围
python dual_ma/main.py -c 600519 --start 20180101 --end 20231231

# 启用移动止损（默认 5%）
python dual_ma/main.py --stop-loss

# 固定止损 8%
python dual_ma/main.py --stop-loss --fixed --stop-loss-pct 0.08

# 仅输出文字结果，不显示图表
python dual_ma/main.py -c 600519 --no-plot

# 图表保存到指定路径
python dual_ma/main.py -o my_chart.png

# 强制从网络刷新数据（跳过缓存）
python dual_ma/main.py --refresh

# 启动图形界面
python dual_ma/gui.py
```

---

## 命令行参数

| 参数 | 简写 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--code` | `-c` | str | `600519` | 股票代码（6 位数字，上证 6 开头，深证 0/3 开头） |
| `--short` | `-s` | int | `10` | 短期均线周期 |
| `--long` | `-l` | int | `30` | 长期均线周期 |
| `--start` | | str | `20200101` | 回测起始日期 YYYYMMDD |
| `--end` | | str | `20251231` | 回测结束日期 YYYYMMDD |
| `--cash` | | float | `100000` | 初始资金（元） |
| `--commission` | | float | `0.001` | 交易佣金费率（0.001 = 千分之一） |
| `--stop-loss` | | flag | False | 启用止损保护 |
| `--stop-loss-pct` | | float | `0.05` | 止损百分比（0.05 = 5%） |
| `--fixed` | | flag | False | 使用固定止损模式（默认移动止损） |
| `--refresh` | | flag | False | 强制从 Yahoo Finance 刷新数据（跳过本地缓存） |
| `--optimize` | | flag | False | 运行参数网格搜索优化 |
| `--no-plot` | | flag | False | 禁用图表显示 |
| `-o` / `--output` | | str | 自动生成 | 图表输出路径 |

---

## GUI 界面

```bash
python dual_ma/gui.py
```

### 界面布局

| 区域 | 功能 |
|------|------|
| **Stock Lookup** | 公司名称模糊搜索 → 自动填充股票代码；内置 80+ 热门 A 股离线回退 |
| **Parameters** | 设置起止日期、短期/长期均线周期、初始资金 |
| **Stop-Loss** | 开关止损、止损百分比、移动/固定止损模式切换 |
| **Chart Options** | 显示/保存 K 线图 |
| **Results Panel** | 三组共 21 项指标：资金与基准、风险调整收益、交易统计 |

### 结果指标

| 分组 | 指标 |
|------|------|
| **Capital & Benchmark** | 初始资金、最终资金、总收益率、买入持有基准、年化收益、年化波动 |
| **Risk-Adjusted Ratios** | Sharpe 比率、Sortino 比率、Calmar 比率、最大回撤、最长回撤天数 |
| **Trade Statistics** | 总交易笔数、胜/负次数、胜率、平均盈利/亏损、盈亏比 |

### 参数优化（Grid Search）

点击 **Optimize (Grid Search)** 按钮，系统自动搜索短线周期 5~50（步长 5）和长线周期 20~110（步长 10）的全部有效组合（短线 < 长线），按 Sharpe 比率降序排列。双击某行结果自动将最优参数填入主界面。

---

## 止损机制

策略内置两种止损模式，可在持仓期间保护资金曲线。

### 固定止损（Fixed Stop-Loss）

从入场价计算，当价格跌幅超过设定百分比时立即平仓。

```
止损条件: (entry_price - current_price) / entry_price >= stop_loss_pct
```

### 移动止损（Trailing Stop，默认）

追踪持仓期间的最高价，当价格从最高点回撤超过设定百分比时平仓。相比固定止损，移动止损能让利润充分奔跑，只在趋势反转时才退出。

```
止损条件: (highest_since_entry - current_price) / highest_since_entry >= stop_loss_pct
```

### 使用示例

```bash
# 移动止损 5%（默认模式）
python dual_ma/main.py --stop-loss

# 固定止损 8%
python dual_ma/main.py --stop-loss --fixed --stop-loss-pct 0.08

# 移动止损 3%（更保守）
python dual_ma/main.py --stop-loss --stop-loss-pct 0.03
```

---

## 回测结果解读

### 控制台输出

运行结束后，控制台会输出类似以下格式的汇总：

```
=======================================================
                策略回测结果汇总
=======================================================
  策略名称     : SMA 双均线交叉 (10/30)
  风控         : 无止损
  初始资金     : 100,000.00 元
  最终资金     : 152,367.42 元
  总收益率     : 52.37 %
  成交笔数     : 34
=======================================================
```

### 回测引擎输出（engine.py）

包含更完整的风险调整指标：

```
============================================================
              Backtest Result Summary
============================================================
  Symbol        : 600519
  SMA Periods   : 10 / 30
  Date Range    : 20200101 ~ 20251231
------------------------------------------------------------
  Total Return  :       52.37%      ← 策略总收益
  Buy & Hold    :       38.12%      ← 买入持有基准（用于对比）
  Annual Return :        8.85%      ← 年化收益率
  Ann.Volatility:       18.20%      ← 年化波动率
------------------------------------------------------------
  Sharpe Ratio  :         0.45      ← > 0 表示风险调整后为正收益
  Sortino Ratio :         0.62      ← 只看下行风险，比 Sharpe 更合理
  Calmar Ratio  :         0.28      ← 年化收益 / |最大回撤|
  Max Drawdown  :       31.50%      ← 历史最大回撤
------------------------------------------------------------
  Win Rate      :       42.86%      ← 趋势策略胜率通常不高，靠盈亏比取胜
  Profit Factor :         1.85      ← > 1 表示整体盈利
```

### 图表解读

![回测图表示例](../results/600519_SMA10-30_20200101-20251231.png)

图表包含：

1. **K 线图**：红色阳线（涨）、绿色阴线（跌）
2. **均线叠加**：短期均线（蓝色）、长期均线（橙色）叠加在 K 线上
3. **买卖标记**：绿色向上箭头 = 买入点，红色向下箭头 = 卖出点
4. **成交量柱**：辅助判断市场活跃度
5. **资金曲线**：展示账户净值变化过程

---

## 参数优化

### 网格搜索

```bash
python dual_ma/main.py --optimize
```

默认搜索范围：
- 短期均线：5 到 50（步长 5，共 9 个值）
- 长期均线：20 到 110（步长 10，共 9 个值）
- 有效组合（短期 < 长期）约 36 组

结果按 Sharpe 比率降序排列，同时展示收益率、最大回撤、胜率和最终资金。

### 推荐参数组合

| 市场风格 | 推荐短周期 | 推荐长周期 | 特点 |
|----------|-----------|-----------|------|
| 短线交易 | 5 | 20 | 对趋势变化敏感，信号多，但手续费影响大 |
| 中线波段 | 10 | 30 | 平衡型，过滤部分噪音但不过于滞后（**默认**） |
| 长线趋势 | 20 | 60 | 信号少但可靠性高，适合大资金低频交易 |
| 超长趋势 | 50 | 200 | 著名"黄金交叉/死亡交叉"组合，信号极少 |

> **提示**：最佳均线参数因股票和时间段而异。建议先用 `--optimize` 对目标标的进行网格搜索，再选择最优参数组合进行验证。切勿仅凭单次回测结果实盘交易。

---

## 代码说明

### `sma_crossover.py` — 策略核心

- 继承 `bt.Strategy`，通过 `params` 定义可配置参数（均线周期、止损开关、止损比例、止损模式）
- `__init__()` 中创建 SMA 短期、SMA 长期、CrossOver 三个指标
- `next()` 是核心决策方法，每个 bar 被调用一次：
  1. 若有未成交订单 → 跳过（只更新止损参考价）
  2. 若持仓 → 先检查止损，再检查死叉卖出信号
  3. 若空仓 → 检查金叉买入信号
- `_stop_triggered()` 根据止损模式判断是否触发止损
- `notify_order()` 记录每笔订单的执行状态和成交价格
- `notify_trade()` 记录每笔已平仓交易的毛利润和净利润
- `stop()` 输出回测结束汇总

### `engine.py` — 回测引擎

- `BacktestResult` / `OptResult` 数据类：结构化存储回测和优化结果
- `SortinoRatio` 自定义分析器：只对负收益计算标准差，比 Sharpe 更合理
- `run_backtest()`：串联数据获取 → Cerebro 初始化 → 策略注入 → 分析器挂载 → 结果提取
- `run_optimization()`：网格搜索最优均线参数，静默运行，按 Sharpe 排序
- `print_result()` / `print_optimization_results()`：格式化控制台输出

### `main.py` — 命令行入口

- `configure_matplotlib()` 配置 seaborn 风格和全局图表样式（16×10 尺寸、120 DPI）
- `parse_args()` 解析所有命令行参数
- `save_chart()` 使用 `cerebro.plot()` 绘制 K 线图并保存为 PNG
- `main()` 根据 `--optimize` 标志分别进入优化或单次回测模式

### `gui.py` — 图形界面

- `StockLookup`：akshare 实时查询股票代码，离线时回退到内置 80+ 热门股票列表
- `BacktestGUI`：完整 Tkinter 应用，左侧参数面板、右侧结果面板、底部状态栏
- 线程安全：回测在 daemon 线程执行，结果通过 `root.after(0, ...)` 更新 UI
- 网格搜索弹窗：`Toplevel` + `ttk.Treeview`，双击行自动填入最优参数

---

## 策略局限性

- **震荡市表现差**：横盘整理时均线频繁交叉，产生大量虚假信号导致连续亏损。
- **滞后性**：均线是滞后指标，金叉发生时趋势往往已走出一段。
- **单边趋势依赖**：策略在趋势明显时表现好，但市场 70% 的时间处于震荡。
- **个股风险**：单一股票回测存在幸存者偏差，建议结合多标的分散测试。

---

## 常见问题

### Q1: 运行报错 "未找到中文字体"

图表中文标题和标签显示为方框。解决方法：
- **Windows**：通常已自带 SimHei（黑体），正常不会出现此问题。
- **macOS**：系统自带 PingFang SC，一般正常。
- **Linux**：安装中文字体 `sudo apt install fonts-noto-cjk`。

### Q2: 数据获取失败 / 返回空数据

- 检查网络代理是否正常（Yahoo Finance 需科学上网）。
- 中国大陆用户需配置 HTTP 代理，确保 yfinance 能访问 Yahoo Finance。
- 首次运行建议使用 `--refresh` 强制从网络获取真实数据并写入本地缓存。
- 确认股票代码格式正确（上证 6 开头，深证 0/3 开头，创业板 3 开头）。
- 无法联网时自动使用本地 CSV 缓存；无缓存时生成模拟数据供逻辑验证。

### Q3: 回测收益率与实盘不符

回测与实际交易存在以下差异：
- **滑点**：回测按收盘价成交，实盘存在买卖价差和滑点。
- **流动性**：回测假设订单全部成交，实盘小盘股可能无法在预期价格成交。
- **市场冲击**：大资金交易会影响市场价格，回测未考虑。
- **幸存者偏差**：历史数据可能不包含已退市股票。

### Q4: 如何修改默认每笔买入比例？

策略默认使用可用资金的 92% 买入（`self.broker.getcash() * 0.92`）。修改 `sma_crossover.py` 中 `next()` 方法的比例系数即可。

### Q5: 如何添加更多技术指标辅助判断？

在 `sma_crossover.py` 的 `__init__` 中添加 backtrader 内置指标：

```python
# 示例：添加 RSI 和 MACD 辅助过滤
self.rsi = bt.indicators.RSI(self.data.close, period=14)
self.macd = bt.indicators.MACD(self.data.close)
```

然后在 `next()` 中增加对应的判断条件。

---

## 免责声明

本项目仅供学习和研究使用，**不构成任何投资建议**。量化回测基于历史数据，过去的表现不代表未来的收益。任何策略都存在失效风险，实盘交易请谨慎决策，盈亏自负。
