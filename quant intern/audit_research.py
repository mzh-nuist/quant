"""
完整审计：55只概念龙头研究
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings, os, re, glob as _glob
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')
CACHE55 = Path('research_cache/55stock')

# ═══════════════════════════════
# AUDIT 1: 55只数量验证
# ═══════════════════════════════
print('=== AUDIT 1: 股票数量 ===')
df55 = pd.read_csv('55个股票.md', sep='\t', header=None)
df55.columns = ['code','name']
df55['code'] = df55['code'].astype(str).str.zfill(6)
print(f'Total rows: {len(df55)}')
print(f'Unique codes: {df55["code"].nunique()}')
dup = df55[df55.duplicated(subset='code', keep=False)]
if len(dup) > 0:
    print(f'DUPLICATES FOUND:')
    print(dup.to_string())
else:
    print('No duplicates - OK')

# ═══════════════════════════════
# AUDIT 2: 时间覆盖 —— 每月有多少只股票
# ═══════════════════════════════
print('\n=== AUDIT 2: 时间覆盖率 ===')

def load_daily(code):
    files = sorted(DATA.glob(f'stock_tx_{code}_*.csv'))
    if not files: return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'close' in df.columns and len(df) > 10:
                dfs.append(df[['close']])
        except: continue
    if not dfs: return None
    df_all = pd.concat(dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep='last')]
    return df_all

count_by_month = {}
first_by_stock = {}
for _, row in df55.iterrows():
    code = row['code']
    d = load_daily(code)
    if d is None: continue
    first_by_stock[code] = d.index[0]
    m = d.resample('ME').last().dropna()
    for dt in m.index:
        if dt not in count_by_month:
            count_by_month[dt] = 0
        count_by_month[dt] += 1

# Monthly coverage
print('Months with < 20 stocks:')
low_months = 0
for dt in sorted(count_by_month.keys()):
    cnt = count_by_month[dt]
    if cnt < 20:
        print(f'  {dt.date()}: {cnt} stocks')
        low_months += 1
if low_months == 0:
    print('  (none - OK)')

# Yearly coverage
years = {}
for dt, cnt in count_by_month.items():
    y = dt.year
    if y not in years: years[y] = []
    years[y].append(cnt)
print('\nYearly coverage:')
for y in sorted(years):
    vals = years[y]
    print(f'  {y}: avg {np.mean(vals):.0f}, min {min(vals)}, max {max(vals)} stocks/month')

# When did each stock first appear?
first_years = [d.year for d in first_by_stock.values()]
from collections import Counter
fc = Counter(first_years)
print(f'\nStocks first available in: {dict(sorted(fc.items()))}')
for yr in [2022, 2023, 2024, 2025, 2026]:
    count = sum(1 for c in first_by_stock if first_by_stock[c].year <= yr)
    print(f'  By {yr}: {count}/55 available')

# ═══════════════════════════════
# AUDIT 3: 组合超额验证
# ═══════════════════════════════
print('\n=== AUDIT 3: 超额收益验证 ===')

stock_monthly = {}
for _, row in df55.iterrows():
    code = row['code']
    d = load_daily(code)
    if d is None: continue
    m = d.resample('ME').last().pct_change().dropna()
    if len(m) > 3:
        stock_monthly[code] = m

all_dates = sorted(set(d for m in stock_monthly.values() for d in m.index))
portfolio_ret = []
for dt in all_dates:
    vals = [m.loc[dt].iloc[0] for m in stock_monthly.values()
            if dt in m.index and np.isfinite(m.loc[dt].iloc[0])]
    if len(vals) >= max(10, len(stock_monthly)*0.3):
        portfolio_ret.append({'date': dt, 'ret': np.mean(vals), 'n': len(vals)})

df_port = pd.DataFrame(portfolio_ret).set_index('date')

csi1k = pd.read_csv(DATA/'CSI1000_price.csv', index_col=0, parse_dates=True)
csi300 = pd.read_csv(DATA/'CSI300_price.csv', index_col=0, parse_dates=True)
csi1k_m = csi1k['close'].resample('ME').last().pct_change().dropna()
csi300_m = csi300['close'].resample('ME').last().pct_change().dropna()

common = df_port.index.intersection(csi1k_m.index).intersection(csi300_m.index)
excess = pd.Series(
    [df_port.loc[dt, 'ret'] - csi1k_m.loc[dt] for dt in common],
    index=common
)
csi300_excess = pd.Series(
    [df_port.loc[dt, 'ret'] - csi300_m.loc[dt] for dt in common],
    index=common
)

print(f'Portfolio months: {len(df_port)}')
print(f'Common months (vs CSI1000+300): {len(common)}')
print(f'Date range: {common[0].date()} ~ {common[-1].date()}')
print(f'Actual month count: {len(common)}')
print(f'(If script said 53 months - verify: {(common[-1].year - common[0].year)*12 + common[-1].month - common[0].month + 1} calendar months span)')

print(f'\nExcess vs CSI1000:')
print(f'  Mean: {excess.mean()*100:.2f}%/month')
print(f'  Median: {excess.median()*100:.2f}%/month')
print(f'  Win rate: {(excess > 0).mean()*100:.0f}%')
print(f'  Cumulative: {((1+excess).prod()-1)*100:.1f}%')
print(f'  Annualized: {((1+excess).prod()**(12/len(excess))-1)*100:.1f}%')

# Outlier analysis
sorted_ex = excess.sort_values()
top5 = sorted_ex.nlargest(5)
bot5 = sorted_ex.nsmallest(5)
print(f'\n  Top 5 months:')
for d, v in top5.items():
    print(f'    {d.date()}: {v*100:+.0f}% (portfolio={df_port.loc[d,"ret"]*100:+.0f}%, CSI1000={csi1k_m.loc[d]*100:+.0f}%)')
print(f'  Bottom 5 months:')
for d, v in bot5.items():
    print(f'    {d.date()}: {v*100:+.0f}% (portfolio={df_port.loc[d,"ret"]*100:+.0f}%, CSI1000={csi1k_m.loc[d]*100:+.0f}%)')

ex_no5 = excess.drop(top5.index).drop(bot5.index)
print(f'\n  Without top+bottom 5: mean={ex_no5.mean()*100:.2f}%/month, cumulative={((1+ex_no5).prod()-1)*100:.1f}%')

# ═══════════════════════════════
# AUDIT 4: 行业集中度偏差
# ═══════════════════════════════
print('\n=== AUDIT 4: 行业偏差 ===')

xlsx_files = _glob.glob('成分详情*.xlsx')
csi300_raw = pd.read_excel([f for f in xlsx_files if '000300' in f][0])
csi500_raw = pd.read_excel([f for f in xlsx_files if '000905' in f][0])
cols_use = {1: 'code', 2: 'name', 11: 'total_mv', 14: 'industry'}
csi300_r = csi300_raw.iloc[:, list(cols_use.keys())].copy()
csi300_r.columns = list(cols_use.values())
csi500_r = csi500_raw.iloc[:, list(cols_use.keys())].copy()
csi500_r.columns = list(cols_use.values())
df_all = pd.concat([csi300_r, csi500_r], ignore_index=True)
df_all['code'] = df_all['code'].astype(str).str.replace('.SZ','').str.replace('.SH','').str.zfill(6)
df_all = df_all.drop_duplicates(subset=['code'])

CAT_MAP = {
    '电子': '科技', '计算机': '科技', '通信': '科技', '传媒': '科技',
    '电力设备': '先进制造', '机械设备': '先进制造', '国防军工': '先进制造',
    '医药生物': '医药', '食品饮料': '消费', '家用电器': '消费', '汽车': '消费',
    '银行': '金融', '非银金融': '金融', '房地产': '金融',
    '基础化工': '周期', '有色金属': '周期', '钢铁': '周期', '石油石化': '周期', '煤炭': '周期',
    '建筑装饰': '周期', '建筑材料': '周期', '农林牧渔': '消费', '纺织服饰': '消费',
    '轻工制造': '消费', '商贸零售': '消费', '社会服务': '消费', '美容护理': '消费',
    '公用事业': '公用事业', '交通运输': '公用事业', '环保': '公用事业',
}
df_all['sector'] = df_all['industry'].map(CAT_MAP).fillna('其他')

df55_m = df55.merge(df_all[['code','industry']], on='code', how='left')
df55_m['sector'] = df55_m['industry'].map(CAT_MAP).fillna('未在CSI300/500')

print('\nCSI300+500 sector distribution (by count):')
print(df_all['sector'].value_counts(normalize=True).mul(100).round(1).to_string())

print('\n55-stock sector distribution:')
sector55 = df55_m['sector'].value_counts()
print(sector55.to_string())
tech_count = len(df55_m[df55_m['sector'].isin(['科技','先进制造'])])
print(f'\n  Tech+mfg ratio: {tech_count/len(df55_m)*100:.0f}%')

# ═══════════════════════════════
# AUDIT 5: 条件分析样本量
# ═══════════════════════════════
print('\n=== AUDIT 5: 条件分析样本量 ===')

sl_chg = (csi1k['close'] / csi300['close']).resample('ME').last().pct_change().dropna()
csi300_vol = csi300['close'].pct_change().rolling(60).std() * np.sqrt(252)
vol_m = csi300_vol.resample('ME').last()

df_cond = pd.DataFrame({'excess': excess}, index=common)
df_cond['style'] = '中性'
df_cond.loc[df_cond.index.isin(sl_chg[sl_chg > 0.005].index), 'style'] = '小盘偏强'
df_cond.loc[df_cond.index.isin(sl_chg[sl_chg < -0.005].index), 'style'] = '大盘偏强'
df_cond['vol'] = '低波'
vol_median = vol_m.reindex(common).median()
df_cond.loc[vol_m.reindex(common) > vol_median, 'vol'] = '高波'

print('Style x Vol sample counts:')
ct = pd.crosstab(df_cond['style'], df_cond['vol'])
print(ct.to_string())
print(f'\nTotal: {len(df_cond)} months')
neutral = (df_cond['style'] == '中性').sum()
print(f'Neutral (excluded from key conclusions): {neutral} months')

# Show mean excess for each cell
print('\nStyle x Vol mean excess (%):')
for s in ['小盘偏强','大盘偏强','中性']:
    for v in ['低波','高波']:
        sub = df_cond[(df_cond['style']==s) & (df_cond['vol']==v)]
        if len(sub) > 0:
            m = sub['excess'].mean() * 100
            w = (sub['excess'] > 0).mean() * 100
            print(f'  {s:6s} x {v:4s}: n={len(sub):2d}, excess={m:+.1f}%, win={w:.0f}%')

# ═══════════════════════════════
# AUDIT 6: 逻辑一致性
# ═══════════════════════════════
print('\n=== AUDIT 6: 逻辑检查 ===')

print()
print('1. 研究前提: "55只是概念辨识度龙头"')
print(f'   - 仅 {df55_m["sector"].value_counts().get("科技", 0)} 只在CSI300/500中被归类为科技')
print(f'   - {df55_m["sector"].value_counts().get("未在CSI300/500", 0)} 只不在CSI300/500成分中')
print('   - 前提本身成立: 这些不是传统市值龙头')

print()
print('2. 基准选择: CSI1000')
cs1k_mkt = (csi1k['close'].iloc[-1]/csi1k['close'].iloc[0]-1)*100
cs300_mkt = (csi300['close'].iloc[-1]/csi300['close'].iloc[0]-1)*100
print(f'   - CSI1000 全期收益: {cs1k_mkt:.0f}% (vs CSI300: {cs300_mkt:.0f}%)')
print(f'   - 55只组合 vs CSI1000 超额 {((1+excess).prod()-1)*100:.0f}%')
print(f'   - 55只组合 vs CSI300 超额 {((1+csi300_excess).prod()-1)*100:.0f}%')
print('   - 问题: 组合跑赢CSI1000远超CSI300，可能因为CSI1000本身就很弱')

print()
print('3. 后视偏差:')
print('   - 这55只是基于\"未来两年业绩能见度\"精选的 — 隐含了事后信息')
print('   - 2022-2024年这些股大涨 -> 超额可能部分来自selection bias')
print('   - 条件分析中\"小盘偏强+高波=最强\"可能因2023-2025恰好是小盘+高波牛市')

print()
print('4. 结论过度外推风险:')
print('   - \"概念溢价活跃，适宜策略运行\" 基于当前1个月的数据点')
print('   - 环境分类(n=2-14个月/格)的统计可靠性有限')
print('   - 没有做样本外验证')

print('\nDone.')
