"""
55stock_leader_premium 数据准备 (全缓存版)
- 行业基准 = CSI300/500内同行业股票等权平均月收益
- 所有数据从已有缓存读取，零新API调用
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings, re, os, glob as _glob
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE = Path('research_cache/55stock')
CACHE.mkdir(parents=True, exist_ok=True)
DATA = Path('research_cache')

# ═══════════════════════════════════════════
# Step 1: 加载并合并 CSI300 + CSI500
# ═══════════════════════════════════════════
print('=' * 60)
print('Step 1: 加载股票池')

xlsx_files = _glob.glob('成分详情*.xlsx')
csi300_path = [f for f in xlsx_files if '000300' in f][0]
csi500_path = [f for f in xlsx_files if '000905' in f][0]
csi300 = pd.read_excel(csi300_path)
csi500 = pd.read_excel(csi500_path)

cols_use = {1: 'code', 2: 'name', 11: 'total_mv', 14: 'industry'}
csi300 = csi300.iloc[:, list(cols_use.keys())].copy()
csi300.columns = list(cols_use.values())
csi500 = csi500.iloc[:, list(cols_use.keys())].copy()
csi500.columns = list(cols_use.values())

df_all = pd.concat([csi300, csi500], ignore_index=True)
df_all['code'] = df_all['code'].astype(str).str.replace('.SZ','').str.replace('.SH','').str.zfill(6)
df_all = df_all.drop_duplicates(subset=['code'])
df_all = df_all.dropna(subset=['industry', 'total_mv'])
df_all['total_mv'] = pd.to_numeric(df_all['total_mv'], errors='coerce')
df_all = df_all[df_all['total_mv'] > 0]
df_all = df_all[df_all['industry'] != '—']  # Remove stocks with placeholder industry

print(f'CSI300+500 合并去重: {len(df_all)} 只, {df_all["industry"].nunique()} 个申万一级行业')
print(f'市值: {df_all["total_mv"].min():.0f}-{df_all["total_mv"].max():.0f}亿, 中位数 {df_all["total_mv"].median():.0f}亿')

# ═══════════════════════════════════════════
# Step 2: 识别龙头(当前市值TOP3/行业)
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 2: 按行业市值排名，取 TOP3 为龙头')

leaders = []
for ind, grp in df_all.groupby('industry'):
    grp_sorted = grp.sort_values('total_mv', ascending=False)
    for rank, (_, row) in enumerate(grp_sorted.head(3).iterrows()):
        leaders.append({
            'industry': ind,
            'rank': rank + 1,
            'code': row['code'],
            'name': row['name'],
            'total_mv': row['total_mv']
        })

df_leader = pd.DataFrame(leaders)
print(f'龙头池: {len(df_leader)} 只 ({df_leader["industry"].nunique()} 行业 × top3)')
print(f'市值: {df_leader["total_mv"].min():.0f}-{df_leader["total_mv"].max():.0f}亿')

# ═══════════════════════════════════════════
# Step 3: 从缓存读取K线,计算月度收益
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 3: 从缓存读取K线 → 计算月度收益')

def load_daily_from_cache(code):
    """从已有缓存加载个股日K线，拼接多年数据"""
    cache_pattern = f'stock_tx_{code}_*.csv'
    files = sorted(DATA.glob(cache_pattern))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'close' in df.columns and len(df) > 10:
                dfs.append(df[['close']])
        except:
            continue

    if not dfs:
        return None

    df_all = pd.concat(dfs).sort_index()
    # Remove duplicates (overlapping dates between files)
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    return df_all

def daily_to_monthly_ret(df_close):
    """日线 → 月线收益率"""
    monthly = df_close.resample('ME').last().dropna()
    monthly_ret = monthly.pct_change().dropna()
    monthly_ret.columns = ['ret']
    return monthly_ret

# Build all-stock monthly returns (for industry benchmark calculation)
print('Building industry benchmarks from all CSI300/500 stocks...')
all_monthly_rets = {}
for _, row in tqdm(df_all.iterrows(), total=len(df_all), desc='全量月收益'):
    code = row['code']
    ind = row['industry']
    df_daily = load_daily_from_cache(code)
    if df_daily is None or len(df_daily) < 60:
        continue

    monthly_ret = daily_to_monthly_ret(df_daily)
    if len(monthly_ret) > 0:
        all_monthly_rets[code] = {
            'returns': monthly_ret,
            'industry': ind,
            'name': row['name']
        }

print(f'可用股票月收益: {len(all_monthly_rets)} 只')

# Compute industry equal-weight benchmarks
ind_benchmarks = {}
for ind in df_all['industry'].unique():
    members = [(c, d['returns']) for c, d in all_monthly_rets.items() if d['industry'] == ind]
    if len(members) < 3:
        continue

    # Combine all member returns
    all_dates = set()
    for _, rets in members:
        all_dates.update(rets.index)
    all_dates = sorted(all_dates)

    # Equal-weight average for each month
    monthly_vals = []
    for dt in all_dates:
        vals = []
        for _, rets in members:
            if dt in rets.index:
                vals.append(rets.loc[dt, 'ret'])
        if len(vals) >= max(3, len(members) * 0.3):  # At least 30% coverage
            monthly_vals.append({'date': dt, 'ind_ret': np.mean(vals)})

    if monthly_vals:
        ind_benchmarks[ind] = pd.DataFrame(monthly_vals).set_index('date')

print(f'行业基准: {len(ind_benchmarks)} 个行业')

# ═══════════════════════════════════════════
# Step 4: 构建面板数据 (股票×月份超额收益)
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 4: 构建面板数据')

panel_rows = []
for _, leader in tqdm(df_leader.iterrows(), total=len(df_leader), desc='面板构建'):
    code = leader['code']
    ind = leader['industry']

    if code not in all_monthly_rets:
        continue
    if ind not in ind_benchmarks:
        continue

    stock_rets = all_monthly_rets[code]['returns']
    ind_rets = ind_benchmarks[ind]

    common = stock_rets.index.intersection(ind_rets.index)
    for dt in common:
        sr = stock_rets.loc[dt, 'ret']
        ir = ind_rets.loc[dt, 'ind_ret']
        if np.isfinite(sr) and np.isfinite(ir):
            panel_rows.append({
                'date': dt,
                'code': code,
                'name': leader['name'],
                'industry': ind,
                'rank': leader['rank'],
                'total_mv': leader['total_mv'],
                'stock_ret': sr,
                'ind_ret': ir,
                'excess_ret': sr - ir
            })

df_panel = pd.DataFrame(panel_rows)
df_panel['year'] = df_panel['date'].dt.year
df_panel['month'] = df_panel['date'].dt.month
df_panel['is_fail'] = (df_panel['excess_ret'] < 0).astype(int)

# Add industry category
CATEGORY_MAP = {
    '食品饮料': '消费', '家用电器': '消费', '纺织服饰': '消费', '轻工制造': '消费',
    '商贸零售': '消费', '社会服务': '消费', '美容护理': '消费', '农林牧渔': '消费',
    '汽车': '消费',
    '医药生物': '医药',
    '电子': '科技', '计算机': '科技', '通信': '科技', '传媒': '科技',
    '电力设备': '高端制造', '机械设备': '高端制造', '国防军工': '高端制造',
    '建筑装饰': '周期', '建筑材料': '周期', '钢铁': '周期', '有色金属': '周期',
    '基础化工': '周期', '石油石化': '周期', '煤炭': '周期',
    '房地产': '金融地产', '银行': '金融地产', '非银金融': '金融地产',
    '公用事业': '公用事业', '交通运输': '公用事业', '环保': '公用事业',
    '综合': '其他'
}
df_panel['category'] = df_panel['industry'].map(CATEGORY_MAP).fillna('其他')

df_panel.to_csv(CACHE / 'leader_panel.csv', index=False)

print(f'面板数据: {len(df_panel)} 条')
print(f'覆盖: {df_panel["code"].nunique()} 只龙头, {df_panel["date"].min().date()}~{df_panel["date"].max().date()}')
print(f'龙头月均超额: {df_panel["excess_ret"].mean()*100:.2f}%')
print(f'龙头失效率: {df_panel["is_fail"].mean()*100:.1f}%')

# ═══════════════════════════════════════════
# Step 5: 年度摘要
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 5: 年度龙头溢价摘要')

yearly = df_panel.groupby('year').agg(
    龙头数=('code', 'nunique'),
    月均超额=('excess_ret', 'mean'),
    超额中位数=('excess_ret', 'median'),
    超额正比例=('excess_ret', lambda x: (x > 0).mean()),
    失效率=('is_fail', 'mean')
).reset_index()

for col in ['月均超额','超额中位数']:
    yearly[col] = yearly[col] * 100
for col in ['超额正比例','失效率']:
    yearly[col] = yearly[col] * 100

print(yearly.to_string(index=False))

# ═══════════════════════════════════════════
# Step 6: 按行业大类汇总
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 6: 行业大类龙头溢价')

# Map SW Level-1 industries to broad categories
CATEGORY_MAP = {
    '食品饮料': '消费', '家用电器': '消费', '纺织服饰': '消费', '轻工制造': '消费',
    '商贸零售': '消费', '社会服务': '消费', '美容护理': '消费', '农林牧渔': '消费',
    '汽车': '消费',
    '医药生物': '医药',
    '电子': '科技', '计算机': '科技', '通信': '科技', '传媒': '科技',
    '电力设备': '高端制造', '机械设备': '高端制造', '国防军工': '高端制造',
    '建筑装饰': '周期', '建筑材料': '周期', '钢铁': '周期', '有色金属': '周期',
    '基础化工': '周期', '石油石化': '周期', '煤炭': '周期',
    '房地产': '金融地产', '银行': '金融地产', '非银金融': '金融地产',
    '公用事业': '公用事业', '交通运输': '公用事业', '环保': '公用事业',
    '综合': '其他'
}

df_panel['category'] = df_panel['industry'].map(CATEGORY_MAP).fillna('其他')

cat_summary = df_panel.groupby('category').agg(
    行业数=('industry', 'nunique'),
    龙头数=('code', 'nunique'),
    月均超额=('excess_ret', 'mean'),
    失效率=('is_fail', 'mean')
).reset_index()
cat_summary['月均超额'] = cat_summary['月均超额'] * 100
cat_summary['失效率'] = cat_summary['失效率'] * 100

print(cat_summary.to_string(index=False))

# ═══════════════════════════════════════════
# Step 7: 按年×行业大类交叉
# ═══════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 7: 年度×行业大类 龙头失效率')

yearly_cat = df_panel.groupby(['year','category']).agg(
    月均超额=('excess_ret', 'mean'),
    失效率=('is_fail', 'mean'),
    n=('code', 'nunique')
).reset_index()
yearly_cat['月均超额'] = yearly_cat['月均超额'] * 100
yearly_cat['失效率'] = yearly_cat['失效率'] * 100

# Pivot for readability
pivot_fail = yearly_cat.pivot(index='category', columns='year', values='失效率')
print('\n失效率(%):')
print(pivot_fail.to_string())

pivot_excess = yearly_cat.pivot(index='category', columns='year', values='月均超额')
print('\n月均超额(%):')
print(pivot_excess.to_string())

print('\nDone. 全部数据已缓存。')
