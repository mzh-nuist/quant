"""
模块1: 历史龙头池构建
从已有缓存识别历史上所有龙头趋势期
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings, os, glob as _glob
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')
CACHE = Path('research_cache/leader_trend')
CACHE.mkdir(parents=True, exist_ok=True)

print('Step 1: 加载股票池 + 行业映射')

# Load Shenwan industry from xlsx
xlsx_files = _glob.glob('成分详情*.xlsx')
df_xlsx = pd.concat([pd.read_excel(f) for f in xlsx_files], ignore_index=True)
df_xlsx['code'] = df_xlsx.iloc[:,1].astype(str).str.replace('.SZ','').str.replace('.SH','').str.zfill(6)
df_xlsx['sw'] = df_xlsx.iloc[:,14]

# Get unique codes from existing cache
import re
cached_codes = set()
for f in DATA.glob('stock_tx_*.csv'):
    m = re.match(r'stock_tx_(\d+)_', f.name)
    if m:
        cached_codes.add(m.group(1))

print(f'Cached stocks: {len(cached_codes)}')

# Map codes to industries
code_to_sw = {}
for _, row in df_xlsx.iterrows():
    if str(row['sw']) not in ['nan', '—', '']:
        code_to_sw[row['code']] = str(row['sw'])

coded_codes_with_sw = set(code_to_sw.keys())
usable = cached_codes & coded_codes_with_sw
print(f'Usable (has cache + SW industry): {len(usable)}')

# Save code-SW mapping
code_sw_df = pd.DataFrame([{'code': c, 'sw': code_to_sw[c]} for c in usable])
code_sw_df.to_csv(CACHE / 'code_sw_map.csv', index=False)

print('\nStep 2: 计算月度12月滚动收益（全量, 可能需要几分钟）')

# Build monthly return series for all usable stocks (same Q3/Q5/Q6 logic)
from tqdm import tqdm

monthly_rets = {}
for code in tqdm(sorted(usable), desc='Loading K-lines'):
    files = sorted(DATA.glob(f'stock_tx_{code}_*.csv'))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'close' in df.columns and len(df) > 10:
                dfs.append(df[['close']])
        except: continue
    if not dfs: continue
    df_all = pd.concat(dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    if len(df_all) < 252: continue  # Need at least 1 year

    monthly = df_all.resample('ME').last().pct_change().dropna()
    if len(monthly) > 12:
        # Compute rolling 12-month cumulative return
        roll_12m = (1 + monthly).rolling(12).apply(np.prod, raw=True) - 1
        monthly_rets[code] = roll_12m.dropna()

print(f'\nStocks with 12m rolling returns: {len(monthly_rets)}')

print('\nStep 3: 构建行业月度面板数据')

# For each month, for each industry, rank stocks by their 12m return
# And also compute MA12 and MA24 for each stock

# First, gather all unique dates across all stocks
all_dates = sorted(set(d for m in monthly_rets.values() for d in m.index))
print(f'Total months with data: {len(all_dates)}')
print(f'Date range: {all_dates[0].date()} ~ {all_dates[-1].date()}')

# Then, for each date, build industry-level ranking
# We'll also need MA12 and MA24 for each stock-date
panel_rows = []
DATE_CHUNK = all_dates  # Process all at once (manageable for ~1670 stocks × ~80 months)

print('Computing industry rankings per month...')
for dt in tqdm(DATE_CHUNK):
    month_data = []
    for code, rets_series in monthly_rets.items():
        if dt not in rets_series.index:
            continue
        sw = code_to_sw.get(code)
        if not sw: continue

        ret_12m = float(rets_series.loc[dt])
        if not np.isfinite(ret_12m): continue

        month_data.append({'code': code, 'sw': sw, 'ret_12m': ret_12m})

    if len(month_data) < 50:
        continue

    df_m = pd.DataFrame(month_data)

    # Rank within each industry
    df_m['rank_pct'] = df_m.groupby('sw')['ret_12m'].rank(pct=True)

    # Flag: 12m return > 100% AND industry rank top 10%
    df_m['is_leader'] = (df_m['ret_12m'] > 1.0) & (df_m['rank_pct'] > 0.9)

    # Filter to leaders
    leaders = df_m[df_m['is_leader']]
    if len(leaders) == 0: continue

    for _, row in leaders.iterrows():
        panel_rows.append({
            'date': dt,
            'code': row['code'],
            'sw': row['sw'],
            'ret_12m': row['ret_12m'],
            'rank_pct': row['rank_pct'],
        })

df_panel = pd.DataFrame(panel_rows)
df_panel.to_csv(CACHE / 'leader_panel_monthly.csv', index=False)
print(f'\nLeader panel: {len(df_panel)} rows')
print(f'Unique leaders: {df_panel["code"].nunique()} stocks')
print(f'Industries: {df_panel["sw"].nunique()}')

# Quick summary per year
df_panel['year'] = pd.to_datetime(df_panel['date']).dt.year
yearly = df_panel.groupby('year').agg(
    leaders=('code', 'nunique'),
    months=('date', 'nunique'),
).reset_index()
print('\nYearly leader count:')
print(yearly.to_string(index=False))

print('\nDone. Panel saved.')
