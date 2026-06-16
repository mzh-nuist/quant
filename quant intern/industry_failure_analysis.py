"""
行业内前兆信号 vs 失效后路径
检查：同一行业内，有前兆的龙头失效后是否跌得更惨？
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')
CACHE = Path('research_cache/leader_trend')

df_panel = pd.read_csv(CACHE / 'leader_panel_monthly.csv', parse_dates=['date'])
df = df_panel.sort_values(['code', 'date'])
df['gap'] = df.groupby('code')['date'].diff().dt.days.fillna(999)
df['episode_id'] = (df['gap'] > 40).cumsum()
episodes = df.groupby(['episode_id', 'code', 'sw']).agg(
    start=('date', 'min'), end=('date', 'max'),
    months=('date', 'count'), avg_ret=('ret_12m', 'mean')
).reset_index()


def load_daily(code):
    files = sorted(DATA.glob(f'stock_tx_{code}_*.csv'))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df_ = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'close' in df_.columns and len(df_) > 10:
                dfs.append(df_[['close']])
        except Exception:
            continue
    if not dfs:
        return None
    daily = pd.concat(dfs).sort_index()
    return daily[~daily.index.duplicated(keep='last')]


records = []
for _, ep in episodes.iterrows():
    code = ep['code']
    sw = ep['sw']
    end_dt = pd.Timestamp(ep['end'])

    daily = load_daily(code)
    if daily is None or len(daily) < 252:
        continue

    # Use get_indexer with ffill to find nearest trading day
    idx_arr = daily.index.get_indexer([end_dt], method='ffill')
    if idx_arr[0] < 0 or idx_arr[0] < 60:
        continue
    end_loc = idx_arr[0]

    w = daily.iloc[max(0, end_loc - 120):end_loc + 1]['close']

    ma10 = w.rolling(10).mean()
    ma20 = w.rolling(20).mean()
    below10 = float(w.iloc[-1] < ma10.iloc[-1])
    below20 = float(w.iloc[-1] < ma20.iloc[-1])

    ma5 = w.rolling(5).mean()
    ma60_r = w.rolling(60).mean()
    m = pd.DataFrame({'m5': ma5, 'm10': ma10, 'm20': ma20, 'm60': ma60_r}).dropna()
    align = float(((m['m5'] > m['m10']) & (m['m10'] > m['m20']) & (m['m20'] > m['m60'])).mean()) if len(m) > 0 else 0.0

    r60 = w.iloc[-1] / w.iloc[-61] - 1 if len(w) >= 61 else 0
    r250 = w.iloc[-1] / w.iloc[0] - 1 if len(w) >= 250 else (r60 if r60 != 0 else 0.01)
    mom = r60 / r250 if abs(r250) > 0.01 else 1.0

    dd = float(w.iloc[-1] / w.max() - 1)

    rets = w.pct_change().dropna()
    vol = np.nan
    if len(rets) >= 60:
        v20 = rets.iloc[-20:].std() * np.sqrt(252)
        v60 = rets.iloc[-60:].std() * np.sqrt(252)
        vol = float(v20 / v60) if v60 > 0 else 1.0

    # Forward returns — also use ffill
    close = daily['close']
    ep_price = close.iloc[end_loc]
    fwds = {}
    for h in [1, 3, 6, 12]:
        target = end_dt + pd.DateOffset(months=h)
        tidx = close.index.get_indexer([target], method='ffill')
        if tidx[0] >= 0 and tidx[0] < len(close):
            f = close.iloc[tidx[0]] / ep_price - 1
            if np.isfinite(f) and abs(f) < 10:
                fwds[h] = float(f)

    records.append({
        'code': code, 'sw': sw, 'months': ep['months'],
        'below_ma10': below10, 'below_ma20': below20,
        'align': align,
        'mom': float(mom) if np.isfinite(mom) else 1.0,
        'vol': vol if np.isfinite(vol) else np.nan,
        'dd': dd if np.isfinite(dd) else 0.0,
        **fwds
    })

df_r = pd.DataFrame(records)
print(f'Sample: {len(df_r)} episodes')

# Only industries with >=10 episodes
sw_counts = df_r['sw'].value_counts()
big_sws = sw_counts[sw_counts >= 10].index.tolist()
print(f'Industries with >=10 episodes: {len(big_sws)}')


# ===== 1. 同行业内：MA10跌破 vs 未跌破 =====
print(f'\n===== 1. 行业内 MA10 跌破 vs 未跌破，失效后中位数收益 =====')
print(f'{"Industry":<12s} {"MA10<":>6s} {"n":>3s} {"+1M":>8s} {"+3M":>8s} {"+6M":>8s} {"+12M":>8s}')
print('-' * 57)
for sw in big_sws:
    sub = df_r[df_r['sw'] == sw]
    for label, mask in [('below', sub['below_ma10'] > 0.5), ('above', sub['below_ma10'] < 0.5)]:
        s = sub[mask]
        if len(s) < 3:
            continue
        m1 = np.median(s[1].dropna()) * 100
        m3 = np.median(s[3].dropna()) * 100
        m6 = np.median(s[6].dropna()) * 100
        m12 = np.median(s[12].dropna()) * 100
        print(f'{sw:<12s} {label:>6s} {len(s):3d} {m1:+7.1f}% {m3:+7.1f}% {m6:+7.1f}% {m12:+7.1f}%')
    print()


# ===== 2. 双信号叠加 =====
print('===== 2. 多信号叠加：MA10+MA20 双跌破 vs 无信号 =====')
df_r['dual_break'] = ((df_r['below_ma10'] > 0.5) & (df_r['below_ma20'] > 0.5)).astype(int)
print(f'Dual MA break: {df_r["dual_break"].sum()}/{len(df_r)} ({df_r["dual_break"].mean()*100:.0f}%)\n')

for sw in big_sws:
    sub = df_r[df_r['sw'] == sw]
    for label, mask in [('dual_break', sub['dual_break'] == 1), ('no_signal', sub['dual_break'] == 0)]:
        s = sub[mask]
        if len(s) < 3:
            continue
        m1 = np.median(s[1].dropna()) * 100
        m3 = np.median(s[3].dropna()) * 100
        m12 = np.median(s[12].dropna()) * 100
        pos12 = (s[12].dropna() > 0).mean() * 100
        print(f'{sw:<12s} {label:>12s} n={len(s):2d}  +1M={m1:+6.0f}%  +3M={m3:+6.0f}%  +12M={m12:+6.0f}%  >0_12M={pos12:.0f}%')


# ===== 3. 关键行业：电子 (n最大) 细分 =====
print('\n===== 3. 电子行业：有前兆 = f(回撤深度, 多头排列%, 波动率) 的细分 =====')
sub_elec = df_r[df_r['sw'] == '电子'].copy()
print(f'电子 n={len(sub_elec)}')
# 回撤深度分组
sub_elec['dd_group'] = pd.cut(sub_elec['dd'], bins=[-1.0, -0.3, -0.15, -0.05, 0.0], labels=['DD>30%', 'DD15-30%', 'DD5-15%', 'DD<5%'])
print('\n回撤深度 x 失效后:')
for grp in ['DD>30%', 'DD15-30%', 'DD5-15%', 'DD<5%']:
    s = sub_elec[sub_elec['dd_group'] == grp]
    if len(s) < 2:
        continue
    m1 = np.median(s[1].dropna()) * 100
    m3 = np.median(s[3].dropna()) * 100
    m12 = np.median(s[12].dropna()) * 100
    pos12 = (s[12].dropna() > 0).mean() * 100
    print(f'  {grp}: n={len(s)}  +1M={m1:+.0f}%  +3M={m3:+.0f}%  +12M={m12:+.0f}%  >0_12M={pos12:.0f}%')

# MA多头排列
sub_elec['align_group'] = pd.cut(sub_elec['align'], bins=[0, 0.2, 0.4, 0.6, 1.0], labels=['0-20%', '20-40%', '40-60%', '60-100%'])
print('\nMA多头排列% x 失效后:')
for grp in ['0-20%', '20-40%', '40-60%', '60-100%']:
    s = sub_elec[sub_elec['align_group'] == grp]
    if len(s) < 2:
        continue
    m1 = np.median(s[1].dropna()) * 100
    m3 = np.median(s[3].dropna()) * 100
    m12 = np.median(s[12].dropna()) * 100
    pos12 = (s[12].dropna() > 0).mean() * 100
    print(f'  {grp}: n={len(s)}  +1M={m1:+.0f}%  +3M={m3:+.0f}%  +12M={m12:+.0f}%  >0_12M={pos12:.0f}%')


# ===== 4. 整体回归：前兆信号能解释失效后收益的多少？=====
print('\n===== 4. 整体：前兆信号 vs 失效后12M收益 (Spearman) =====')
from scipy.stats import spearmanr

signals = ['below_ma10', 'below_ma20', 'align', 'mom', 'vol', 'dd']
for sig in signals:
    valid = df_r[[sig, 12]].dropna()
    if len(valid) < 10:
        continue
    r, p = spearmanr(valid[sig], valid[12])
    sig_str = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
    print(f'  {sig:>12s} vs +12M: r={r:+.3f}, p={p:.3f} {sig_str}')

# 每个行业单独跑
print('\n分行业 Spearman (below_ma10 vs +12M):')
for sw in big_sws:
    sub = df_r[df_r['sw'] == sw][['below_ma10', 12]].dropna()
    if len(sub) < 5:
        continue
    r, p = spearmanr(sub['below_ma10'], sub[12])
    print(f'  {sw:<12s} n={len(sub):2d}: r={r:+.3f}, p={p:.3f}')

print('\nDone.')
