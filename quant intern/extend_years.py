"""
扩展龙头K线数据到 2022-2024
只拉取已有缓存中缺失的年份，不重复拉取
"""
import numpy as np
import pandas as pd
import akshare as ak
from pathlib import Path
from tqdm import tqdm
import warnings, time, re, os
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')
CACHE55 = Path('research_cache/55stock')

# ── Step 1: 获取龙头代码列表 ──
panel_path = CACHE55 / 'leader_panel.csv'
if panel_path.exists():
    df_panel = pd.read_csv(panel_path)
    leader_codes = sorted(df_panel['code'].astype(str).str.zfill(6).unique().tolist())
else:
    # Fallback: get from leader pool construction
    print('面板数据不存在，请先运行 build_leader_pool.py')
    exit(1)

print(f'龙头代码: {len(leader_codes)} 只')

# ── Step 2: 检查每只龙头的缓存覆盖 ──
target_years = ['2022', '2023', '2024']
missing = []

for code in leader_codes:
    # Find existing cache files for this code
    existing_files = list(DATA.glob(f'stock_tx_{code}_*.csv'))
    existing_years = set()
    for f in existing_files:
        m = re.match(rf'stock_tx_{code}_(\d{{4}})\d{{4}}_(\d{{4}})\d{{4}}\.csv', f.name)
        if m:
            existing_years.add(m.group(1))
            existing_years.add(m.group(2))

    for yr in target_years:
        if yr not in existing_years:
            missing.append((code, yr))

print(f'需拉取: {len(missing)} 条 (股票×年份)')
print(f'涉及股票: {len(set(c for c,_ in missing))} 只')

if len(missing) == 0:
    print('所有年份已覆盖，无需拉取。')
    exit(0)

# ── Step 3: 拉取缺失年份 ──
def fetch_year(code, year):
    """拉取单个股票单年日K线"""
    code = str(code).zfill(6)
    start_date = f'{year}0101'
    end_date = f'{year}1231'
    cache_file = DATA / f'stock_tx_{code}_{start_date}_{end_date}.csv'

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if len(df) > 100:
                return True
        except:
            pass

    if code.startswith(('6','68')):
        symbol = f'sh{code}'
    else:
        symbol = f'sz{code}'

    for attempt in range(4):
        try:
            df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 50:
                close_col = 'close' if 'close' in df.columns else 'Close'
                if close_col not in df.columns:
                    return False
                df = df.rename(columns={close_col: 'close'})
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                df = df[['close']].sort_index()
                df.to_csv(cache_file)
                return True
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
        except Exception as e:
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
            else:
                return False
    return False

success = 0
failed = []
for code, year in tqdm(missing, desc='拉取缺失年份'):
    ok = fetch_year(code, year)
    if ok:
        success += 1
    else:
        failed.append((code, year))
    time.sleep(0.3)  # Rate limit: ~3 requests/sec

print(f'\n成功: {success}/{len(missing)}')
if failed:
    print(f'失败 ({len(failed)}):')
    for c, y in failed[:20]:
        print(f'  {c} {y}')

# ── Step 4: 验证覆盖率 ──
print(f'\n=== 覆盖率验证 ===')
for yr in target_years:
    count = 0
    for code in leader_codes:
        existing = list(DATA.glob(f'stock_tx_{code}_*.csv'))
        years_covered = set()
        for f in existing:
            m = re.match(rf'stock_tx_{code}_(\d{{4}})\d{{4}}_(\d{{4}})\d{{4}}\.csv', f.name)
            if m:
                years_covered.add(m.group(1))
                years_covered.add(m.group(2))
        if yr in years_covered:
            count += 1
    print(f'  {yr}: {count}/{len(leader_codes)} ({100*count/len(leader_codes):.0f}%)')

print('\nDone.')
