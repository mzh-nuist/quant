"""
模块2-4: 龙头趋势特征 + 失效前置条件 + 失效后路径
依赖: build_leader_trend_pool.py 产出的 leader_panel_monthly.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings, os, re
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')
CACHE = Path('research_cache/leader_trend')

print('=' * 60)
print('Step 0: 加载数据')

df_panel = pd.read_csv(CACHE / 'leader_panel_monthly.csv', parse_dates=['date'])
print(f'Leader panel: {len(df_panel)} rows, {df_panel["code"].nunique()} stocks')
print(f'Industries: {df_panel["sw"].nunique()}')
print(f'Date range: {df_panel["date"].min().date()} ~ {df_panel["date"].max().date()}')

# ── Module 2: 龙头趋势的共性特征 ──
print('\n' + '=' * 60)
print('MODULE 2: 龙头趋势期共性特征')

# Each stock can appear in multiple months - group consecutive months into "episodes"
# An episode = same stock, consecutive months where is_leader=True
# (panel already only contains leader months, so this is straightforward)

df = df_panel.sort_values(['code','date']).copy()
df['month_diff'] = df.groupby('code')['date'].diff().dt.days.fillna(999)

# A new episode starts when month_diff > 40 days (not consecutive months)
df['episode_id'] = (df['month_diff'] > 40).cumsum()

# Per episode stats
episodes = df.groupby(['episode_id','code','sw']).agg(
    start_date=('date','min'),
    end_date=('date','max'),
    months=('date','count'),
    avg_ret_12m=('ret_12m','mean'),
    max_ret_12m=('ret_12m','max'),
    min_rank=('rank_pct','min'),
).reset_index()

print(f'\n龙头趋势段: {len(episodes)} 段')
print(f'  来自: {episodes["code"].nunique()} 只股票, {episodes["sw"].nunique()} 个行业')

print('\n趋势段持续时间分布(月):')
print(f'  均值: {episodes["months"].mean():.1f}')
print(f'  中位数: {episodes["months"].median():.0f}')
print(f'  P25: {episodes["months"].quantile(0.25):.0f}')
print(f'  P75: {episodes["months"].quantile(0.75):.0f}')
print(f'  最长: {episodes["months"].max()}')

print('\n行业分布(top 10):')
sw_counts = episodes.groupby('sw').size().sort_values(ascending=False).head(10)
for sw, cnt in sw_counts.items():
    print(f'  {sw}: {cnt} 段')

print('\n年化特征(12m收益):')
ret = episodes['avg_ret_12m']
print(f'  均值: {ret.mean()*100:.0f}%')
print(f'  中位数: {ret.median()*100:.0f}%')
print(f'  P25: {ret.quantile(0.25)*100:.0f}%')
print(f'  P75: {ret.quantile(0.75)*100:.0f}%')

# ── Module 3: 失效前置条件 ──
print('\n' + '=' * 60)
print('MODULE 3: 龙头失效前置条件')

# For each episode, look at the last month and the months around it
# We need daily K-line data to compute:
# - Price failure: did close break below MA10/MA20?
# - Momentum decay: 60d ret vs 250d ret ratio
# - Volatility spike: 20d vol vs 60d vol ratio
# - MA structure: % of days with MA5>MA10>MA20>MA60

# Load market indicators
csi1k = pd.read_csv(DATA/'CSI1000_price.csv', index_col=0, parse_dates=True)
csi300 = pd.read_csv(DATA/'CSI300_price.csv', index_col=0, parse_dates=True)

# Market style: 12-month MA direction (Q1 1D methodology)
ratio = csi1k['close'] / csi300['close']
ratio_ma12 = ratio.rolling(252, min_periods=126).mean()
ratio_ma12_m = ratio_ma12.resample('ME').last().dropna()
ma_direction = ratio_ma12_m.diff()

# CSI300 volatility
csi300_vol = csi300['close'].pct_change().rolling(60).std() * np.sqrt(252)
vol_m = csi300_vol.resample('ME').last()

def load_daily(code):
    files = sorted(DATA.glob(f'stock_tx_{code}_*.csv'))
    if not files: return None, None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'close' in df.columns and len(df) > 10:
                dfs.append(df[['close']])
        except: continue
    if not dfs: return None, None
    daily = pd.concat(dfs).sort_index()
    daily = daily[~daily.index.duplicated(keep='last')]
    monthly = daily.resample('ME').last()
    return daily, monthly

# For each episode's last month, compute precursor signals
precursor_rows = []
for _, ep in episodes.iterrows():
    code = ep['code']
    end_dt = pd.Timestamp(ep['end_date'])

    daily, monthly = load_daily(code)
    if daily is None or len(daily) < 252: continue
    if end_dt not in monthly.index: continue

    # Window: last 60 trading days before episode end
    end_loc = daily.index.get_indexer([end_dt], method='ffill')[0]
    if end_loc < 120: continue
    window = daily.iloc[max(0,end_loc-120):end_loc+1]

    close = window['close']

    # 1. Price structure: close vs MA10/MA20 at end
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    below_ma10 = (close.iloc[-1] < ma10.iloc[-1])
    below_ma20 = (close.iloc[-1] < ma20.iloc[-1])

    # 2. Momentum decay: 60d ret / 250d ret ratio
    ret_60 = close.iloc[-1] / close.iloc[max(0,len(close)-61)] - 1
    ret_250 = close.iloc[-1] / close.iloc[0] - 1 if len(close) >= 250 else ret_60
    mom_ratio = ret_60 / ret_250 if abs(ret_250) > 0.01 else 1.0

    # 3. Volatility spike: 20d vol / 60d vol
    rets = close.pct_change().dropna()
    if len(rets) >= 60:
        vol20 = rets.iloc[-20:].std() * np.sqrt(252)
        vol60 = rets.iloc[-60:].std() * np.sqrt(252)
        vol_ratio = vol20 / vol60 if vol60 > 0 else 1.0
    else:
        vol_ratio = np.nan

    # 4. MA structure: % of days with MA5>MA10>MA20>MA60 in last 60 days
    ma5 = close.rolling(5).mean()
    ma60_roll = close.rolling(60).mean()
    aligned = pd.DataFrame({'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60_roll}).dropna()
    bullish_align = (aligned['ma5'] > aligned['ma10']) & \
                    (aligned['ma10'] > aligned['ma20']) & \
                    (aligned['ma20'] > aligned['ma60'])
    align_pct = bullish_align.mean() if len(bullish_align) > 0 else 0

    # Market context at end month
    style = '中性'
    if end_dt in ma_direction.index:
        d = ma_direction.loc[end_dt]
        if d > 0: style = '小盘偏强'
        elif d < 0: style = '大盘偏强'

    vol_reg = '中性'
    if end_dt in vol_m.index:
        vol_med = vol_m.dropna().median()
        v = vol_m.loc[end_dt]
        vol_reg = '高波' if v > vol_med else '低波'

    precursor_rows.append({
        'episode_id': ep['episode_id'],
        'code': code, 'sw': ep['sw'],
        'end_date': end_dt,
        'months': ep['months'],
        'below_ma10': below_ma10,
        'below_ma20': below_ma20,
        'mom_ratio': mom_ratio if np.isfinite(mom_ratio) else np.nan,
        'vol_ratio': vol_ratio if np.isfinite(vol_ratio) else np.nan,
        'align_pct': align_pct,
        'style': style,
        'vol_reg': vol_reg,
    })

df_pre = pd.DataFrame(precursor_rows)
print(f'\n有失效前置数据的龙头段: {len(df_pre)} 段')

# Analyze: at episode end, what conditions are most common?
print('\n失效时(趋势段最后一个月)的状态:')
print(f'  收盘<MA10: {df_pre["below_ma10"].mean()*100:.0f}%')
print(f'  收盘<MA20: {df_pre["below_ma20"].mean()*100:.0f}%')
print(f'  MA多头排列占比: 均值 {df_pre["align_pct"].mean()*100:.0f}%, 中位数 {df_pre["align_pct"].median()*100:.0f}%')
print(f'  动量比(60d/250d): 均值 {df_pre["mom_ratio"].mean():.2f}, 中位数 {df_pre["mom_ratio"].median():.2f}')
print(f'  波动率比(20d/60d): 均值 {df_pre["vol_ratio"].mean():.2f}, 中位数 {df_pre["vol_ratio"].median():.2f}')

# Group by style at failure time
print('\n失效时的市场风格:')
for s in ['小盘偏强','大盘偏强']:
    sub = df_pre[df_pre['style']==s]
    if len(sub) > 0:
        print(f'  {s}: n={len(sub)}, below_MA10={sub["below_ma10"].mean()*100:.0f}%, below_MA20={sub["below_ma20"].mean()*100:.0f}%')

# ── Module 4: 失效后路径 ──
print('\n' + '=' * 60)
print('MODULE 4: 龙头失效后价格路径')

# Track forward returns from each episode's end_date
post_returns = {1: [], 3: [], 6: [], 12: []}

for _, ep in episodes.iterrows():
    code = ep['code']
    end_dt = pd.Timestamp(ep['end_date'])

    daily, _ = load_daily(code)
    if daily is None: continue

    close = daily['close']
    if end_dt not in close.index: continue

    end_loc = close.index.get_loc(end_dt)
    end_price = close.iloc[end_loc]

    for horizon_months, store in post_returns.items():
        target_dt = end_dt + pd.DateOffset(months=horizon_months)
        if target_dt > close.index[-1]: continue
        # Find closest actual trading day
        future = close[close.index >= target_dt]
        if len(future) > 0:
            fwd_ret = future.iloc[0] / end_price - 1
            if np.isfinite(fwd_ret) and abs(fwd_ret) < 10:
                store.append(fwd_ret)

print('\n失效后N个月的累积收益分布:')
for horizon, rets in post_returns.items():
    if len(rets) < 5: continue
    arr = np.array(rets)
    print(f'  +{horizon}M: n={len(arr)}, mean={arr.mean()*100:+.1f}%, median={np.median(arr)*100:+.1f}%, '
          f'>0={((arr>0).mean()*100):.0f}%, >20%={((arr>0.2).mean()*100):.0f}%')

print('\nDone.')
