"""
龙头失效条件分析
用扩展后的面板数据 + Q1/Q4市场指标缓存，分组统计龙头失效率
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings, os
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE55 = Path('research_cache/55stock')
DATA = Path('research_cache')

# ═══════════════════════════════════════════
# Step 1: 加载面板数据 + 市场指标
# ═══════════════════════════════════════════
print('=' * 60)
print('Step 1: 加载数据')

df = pd.read_csv(CACHE55 / 'leader_panel.csv', parse_dates=['date'])
print(f'面板: {len(df)}条, {df["code"].nunique()}只, {df["date"].min().date()}~{df["date"].max().date()}')

# Load market indicators
csi1k = pd.read_csv(DATA / 'CSI1000_price.csv', index_col=0, parse_dates=True)
csi300 = pd.read_csv(DATA / 'CSI300_price.csv', index_col=0, parse_dates=True)
print(f'CSI1000: {len(csi1k)}r, CSI300: {len(csi300)}r')

# ═══════════════════════════════════════════
# Step 2: 计算市场状态指标（月度频率，对齐面板）
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 2: 构建月度市场状态指标')

# Align to monthly level
monthly_dates = sorted(df['date'].unique())
print(f'面板月份: {len(monthly_dates)} 个月')

# 2a. 大小盘风格: CSI1000/CSI300 月度比值及其变化方向
small_large_ratio = csi1k['close'] / csi300['close']
small_large_monthly = small_large_ratio.resample('MS').last().dropna()
small_large_chg = small_large_monthly.diff()  # 月度变化

# 2b. 风格方向: 上升=小盘偏强, 下降=大盘偏强
style_direction = pd.Series(index=small_large_chg.index, dtype=str)
style_direction[small_large_chg > 0] = '小盘偏强'
style_direction[small_large_chg < 0] = '大盘偏强'
style_direction[small_large_chg.abs() < 0.01] = '中性'

# 2c. 小盘/大盘比值的历史分位
for dt in small_large_monthly.index:
    lookback = small_large_monthly.loc[:dt].iloc[-24:]  # 2-year rolling
    if len(lookback) >= 12:
        pct = (lookback < small_large_monthly.loc[dt]).mean()
        style_direction.loc[dt] += f' | 分位{100*pct:.0f}%'

# 2d. 市场波动率（CSI300 60日年化波动率）
csi300_ret = csi300['close'].pct_change().dropna()
csi300_vol = csi300_ret.rolling(60).std() * np.sqrt(252)
csi300_vol_monthly = csi300_vol.resample('MS').last().dropna()

# 2e. CSI300 月度收益率
csi300_monthly_ret = csi300['close'].resample('MS').last().pct_change().dropna()

# 3f. 市场状态标记
market_state = pd.DataFrame({
    'small_large_ratio': small_large_monthly,
    'style_dir': style_direction,
    'csi300_vol': csi300_vol_monthly,
    'csi300_ret': csi300_monthly_ret,
    'small_large_chg': small_large_chg
}).dropna()

print(f'市场状态: {len(market_state)} 个月')

# ═══════════════════════════════════════════
# Step 3: 合并面板与市场状态 → 分组统计
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 3: 合并面板 + 市场状态 → 分组统计')

# Match date to month start for joining
df['month_start'] = df['date'].dt.to_period('M').dt.start_time
market_state['month_start'] = market_state.index

df_full = df.merge(market_state, on='month_start', how='left')
df_full = df_full.dropna(subset=['style_dir'])
print(f'合并后: {len(df_full)} 条')

# ── 3a. 按大小盘风格分组 ──
print('\n── 龙头失效率 vs 大小盘风格 ──')

# Simplified: 小盘偏强 vs 大盘偏强 (binary)
df_full['style_simple'] = '中性'
df_full.loc[df_full['small_large_chg'] > 0.005, 'style_simple'] = '小盘偏强'
df_full.loc[df_full['small_large_chg'] < -0.005, 'style_simple'] = '大盘偏强'

style_group = df_full.groupby('style_simple').agg(
    月份数=('date', 'nunique'),
    龙头数=('code', 'nunique'),
    失效率=('is_fail', 'mean'),
    月均超额=('excess_ret', 'mean')
).round(4)
style_group['失效率'] = style_group['失效率'] * 100
style_group['月均超额'] = style_group['月均超额'] * 100
print(style_group.to_string())

# ── 3b. 按市场波动率分位分组 ──
print('\n── 龙头失效率 vs 市场波动率 ──')

vol_median = df_full['csi300_vol'].median()
df_full['vol_regime'] = '低波'
df_full.loc[df_full['csi300_vol'] > vol_median, 'vol_regime'] = '高波'

vol_group = df_full.groupby('vol_regime').agg(
    失效率=('is_fail', 'mean'),
    月均超额=('excess_ret', 'mean')
).round(4)
vol_group['失效率'] = vol_group['失效率'] * 100
vol_group['月均超额'] = vol_group['月均超额'] * 100
print(vol_group.to_string())

# ── 3c. 按市场涨跌分组 ──
print('\n── 龙头失效率 vs 市场涨跌 ──')

df_full['market_dir'] = '上涨'
df_full.loc[df_full['csi300_ret'] < 0, 'market_dir'] = '下跌'

mkt_group = df_full.groupby('market_dir').agg(
    失效率=('is_fail', 'mean'),
    月均超额=('excess_ret', 'mean')
).round(4)
mkt_group['失效率'] = mkt_group['失效率'] * 100
mkt_group['月均超额'] = mkt_group['月均超额'] * 100
print(mkt_group.to_string())

# ── 3d. 按风格×波动率二维分组 ──
print('\n── 二维分组: 风格 × 波动率 ──')

df_full['style_vol'] = df_full['style_simple'] + ' × ' + df_full['vol_regime']
sv_group = df_full.groupby('style_vol').agg(
    n=('code', 'size'),
    失效率=('is_fail', 'mean'),
    月均超额=('excess_ret', 'mean')
).round(4)
sv_group['失效率'] = sv_group['失效率'] * 100
sv_group['月均超额'] = sv_group['月均超额'] * 100
sv_group = sv_group.sort_values('失效率')
print(sv_group.to_string())

# ── 3e. 按行业大类×风格分组 ──
print('\n── 行业大类 × 风格 交互 ──')

cat_style = df_full.groupby(['category','style_simple']).agg(
    失效率=('is_fail', 'mean'),
    月均超额=('excess_ret', 'mean')
).round(4)
cat_style['失效率'] = cat_style['失效率'] * 100
cat_style['月均超额'] = cat_style['月均超额'] * 100
print(cat_style.to_string())

# ═══════════════════════════════════════════
# Step 4: 时间序列 —— 龙头失效率的月度变化
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 4: 龙头失效率月度时间序列')

monthly_fail = df_full.groupby('month_start').agg(
    失效率=('is_fail', 'mean'),
    月均超额=('excess_ret', 'mean'),
    龙头数=('code', 'nunique'),
    csi300_ret=('csi300_ret', 'first'),
    small_large_chg=('small_large_chg', 'first')
).round(4)
monthly_fail['失效率'] = monthly_fail['失效率'] * 100
monthly_fail['月均超额'] = monthly_fail['月均超额'] * 100

# Print months with extreme failure rates
print('\n失效率最高月份 (top 10):')
top_fail = monthly_fail.nlargest(10, '失效率')
for dt, row in top_fail.iterrows():
    print(f'  {pd.Timestamp(dt).date()} | 失效率{row["失效率"]:.0f}% | CSI300:{row["csi300_ret"]*100:+.1f}% | 小/大比变化:{row["small_large_chg"]*100:+.1f}%')

print('\n失效率最低月份 (bottom 10):')
bot_fail = monthly_fail.nsmallest(10, '失效率')
for dt, row in bot_fail.iterrows():
    print(f'  {pd.Timestamp(dt).date()} | 失效率{row["失效率"]:.0f}% | CSI300:{row["csi300_ret"]*100:+.1f}% | 小/大比变化:{row["small_large_chg"]*100:+.1f}%')

# ═══════════════════════════════════════════
# Step 5: 总结 —— 什么环境龙头最危险
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 5: 龙头策略适用环境总结')

print(f'''
╔══════════════════════════════════════════════╗
║           龙头策略环境适配总结              ║
╠══════════════════════════════════════════════╣
║  样本: {df_full["code"].nunique()}只龙头, {len(df_full)}条月频记录           ║
║  时间: {df_full["date"].min().date()} ~ {df_full["date"].max().date()}            ║
║  整体失效率: {df_full["is_fail"].mean()*100:.1f}%                        ║
╚══════════════════════════════════════════════╝
''')

# Find worst and best regimes
print('最危险环境 (失效率最高):')
worst = sv_group.nlargest(3, '失效率')
for idx, row in worst.iterrows():
    print(f'  {idx}: 失效率{row["失效率"]:.0f}%, 月均超额{row["月均超额"]:+.2f}%')

print('\n最适宜环境 (失效率最低):')
best = sv_group.nsmallest(3, '失效率')
for idx, row in best.iterrows():
    print(f'  {idx}: 失效率{row["失效率"]:.0f}%, 月均超额{row["月均超额"]:+.2f}%')

# Save results
monthly_fail.to_csv(CACHE55 / 'monthly_failure_rates.csv')
print(f'\n结果已保存至 {CACHE55}/monthly_failure_rates.csv')
print('Done.')
