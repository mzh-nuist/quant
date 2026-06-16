"""
55只等权组合超额收益分析
所有方法有前例可循，来源标注在注释中。
无自编指标。无预测量。
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings, os, re, glob as _glob
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')
CACHE55 = Path('research_cache/55stock')

# ═══════════════════════════════════
# Step 1: 加载数据 + 逐年覆盖率
# ═══════════════════════════════════
print('=' * 60)
print('Step 1: 加载55只 + 逐年覆盖率')

df55 = pd.read_csv('55个股票.md', sep='\t', header=None)
df55.columns = ['code','name']
df55['code'] = df55['code'].astype(str).str.zfill(6)
print(f'Total: {len(df55)} stocks, {df55["code"].nunique()} unique')

# 申万行业: 34只从xlsx col14获取（同Q5/Q6），21只从stock_profile_cninfo获取CSRC行业后手工映射为申万一级
# ⚠️ CSRC→SW 为手工映射（"铁路船舶航空航天和其他运输设备制造业"→"国防军工"等），可能有偏差
df_xlsx = pd.concat([pd.read_excel(f) for f in _glob.glob('成分详情*.xlsx')], ignore_index=True)
df_xlsx['code'] = df_xlsx.iloc[:,1].astype(str).str.replace('.SZ','').str.replace('.SH','').str.zfill(6)
df_xlsx['sw'] = df_xlsx.iloc[:,14]
CSRC_MAP = {
    '688521': '计算机', '601126': '电力设备', '603667': '机械设备', '002896': '机械设备',
    '688146': '国防军工', '688630': '机械设备', '001309': '电子', '300913': '电力设备',
    '603256': '建筑材料', '301217': '电子', '301511': '电子', '601208': '基础化工',
    '002364': '电力设备', '300870': '电子', '002536': '汽车', '601869': '电子',
    '603629': '钢铁', '603601': '建筑材料', '002484': '电子', '000636': '电子',
    '688313': '电子',
}
df55 = df55.merge(df_xlsx[['code','sw']].drop_duplicates(subset=['code']), on='code', how='left')
df55['sw'] = df55['sw'].fillna(df55['code'].map(CSRC_MAP)).fillna('其他')

# 从已有缓存加载日K（同Q5/Q6 extend_years.py的load_daily逻辑）
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

# 月收益序列（同Q3/Q5/Q6 build_leader_pool.py: resample('ME').last().pct_change()）
stock_monthly = {}
first_dates = {}
for _, row in df55.iterrows():
    code = row['code']
    d = load_daily(code)
    if d is None: continue
    first_dates[code] = d.index[0]
    m = d.resample('ME').last().pct_change().dropna()
    if len(m) > 3:
        stock_monthly[code] = m

print(f'Data available: {len(stock_monthly)}/55 stocks')

# 逐年覆盖率（同Q5 Cell 6数据质量表）
print('\n逐年覆盖率:')
for yr in range(2020, 2027):
    count = sum(1 for c in first_dates if first_dates[c].year <= yr)
    ipos = sum(1 for c in first_dates if first_dates[c].year == yr)
    print(f'  {yr}: {count}/55 have data {"(+" + str(ipos) + " newly available)" if ipos > 0 else ""}')

# 等权组合（同Q3 step4C行业等权逻辑）
all_dates = sorted(set(d for m in stock_monthly.values() for d in m.index))
portfolio_ret = []
for dt in all_dates:
    vals = [m.loc[dt].iloc[0] for m in stock_monthly.values()
            if dt in m.index and np.isfinite(m.loc[dt].iloc[0])]
    # 覆盖率阈值：≥30%股票 或 ≥10只（同Q3扩展C G4: len(sub)<50跳过的反向逻辑）
    if len(vals) >= max(10, len(stock_monthly)*0.3):
        portfolio_ret.append({'date': dt, 'ret': np.mean(vals), 'n': len(vals)})

df_port = pd.DataFrame(portfolio_ret).set_index('date')
print(f'\nPortfolio months: {len(df_port)}, {df_port.index[0].date()} ~ {df_port.index[-1].date()}')

# ═══════════════════════════════════
# Step 2: 双基准等权超额收益
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 2: 等权超额收益 (vs CSI1000 & vs CSI300)')

csi1k = pd.read_csv(DATA/'CSI1000_price.csv', index_col=0, parse_dates=True)
csi300 = pd.read_csv(DATA/'CSI300_price.csv', index_col=0, parse_dates=True)
csi1k_m = csi1k['close'].resample('ME').last().pct_change().dropna()
csi300_m = csi300['close'].resample('ME').last().pct_change().dropna()

common = df_port.index.intersection(csi1k_m.index).intersection(csi300_m.index)

excess_1k = pd.Series([df_port.loc[dt,'ret'] - csi1k_m.loc[dt] for dt in common], index=common)
excess_300 = pd.Series([df_port.loc[dt,'ret'] - csi300_m.loc[dt] for dt in common], index=common)

def describe_excess(ex, label):
    print(f'\n  55只等权组合 vs {label}:')
    print(f'    均值: {ex.mean()*100:+.2f}%/月')
    print(f'    中位数: {ex.median()*100:+.2f}%/月')
    print(f'    胜率: {(ex>0).mean()*100:.0f}%')
    print(f'    累计: {((1+ex).prod()-1)*100:.1f}%')
    print(f'    年化: {((1+ex).prod()**(12/len(ex))-1)*100:.1f}%')

    # 去极值: 标准IQR规则（1.5×IQR），不是随手定N
    Q1 = ex.quantile(0.25)
    Q3 = ex.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    ex_clean = ex[(ex >= lower) & (ex <= upper)]
    n_removed = len(ex) - len(ex_clean)
    print(f'    去极值(IQR×1.5, 移除{n_removed}个月):')
    print(f'      均值: {ex_clean.mean()*100:+.2f}%/月')
    print(f'      累计: {((1+ex_clean).prod()-1)*100:.1f}%')

    top5 = ex.nlargest(5)
    bot5 = ex.nsmallest(5)
    print(f'    最大5个月:')
    for d, v in top5.items():
        print(f'      {d.date()}: {v*100:+.0f}% (组合{df_port.loc[d,"ret"]*100:+.0f}%, {label}{csi1k_m.loc[d]*100:+.0f}%)')
    print(f'    最小5个月:')
    for d, v in bot5.items():
        print(f'      {d.date()}: {v*100:+.0f}% (组合{df_port.loc[d,"ret"]*100:+.0f}%, {label}{csi1k_m.loc[d]*100:+.0f}%)')

describe_excess(excess_1k, 'CSI1000')
describe_excess(excess_300, 'CSI300')

# ═══════════════════════════════════
# Step 3: 超额收益条件分组统计
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 3: 等权超额收益条件分组统计')
print('注意: 以下仅为描述统计，非预测量。不构成对未来的判断。')

# 风格方向: 用CSI1000/CSI300比值的12月MA方向（同Q1 1D方法论）
# Q1使用252日MA + argrelextrema(order=126)检测周期拐点
# 这里简化为月度频率: 12月MA上升=小盘偏强, 下降=大盘偏强
ratio = csi1k['close'] / csi300['close']
ratio_ma12 = ratio.rolling(252, min_periods=126).mean()
ratio_ma12_m = ratio_ma12.resample('ME').last().dropna()
ma_direction = ratio_ma12_m.diff()  # 正值=MA上升=小盘偏强

csi300_vol = csi300['close'].pct_change().rolling(60).std() * np.sqrt(252)
vol_m = csi300_vol.resample('ME').last()

df_cond = pd.DataFrame({
    'excess_1k': excess_1k.values,
    'excess_300': excess_300.values,
    'csi300_ret': csi300_m.reindex(common).values
}, index=common)

# 风格分类: 基于12月MA方向(Q1 1D同款)，不是随手阈值0.5%
df_cond['style'] = '中性'
df_cond.loc[df_cond.index.isin(ma_direction[ma_direction > 0].index), 'style'] = '小盘偏强'
df_cond.loc[df_cond.index.isin(ma_direction[ma_direction < 0].index), 'style'] = '大盘偏强'
df_cond['vol'] = '低波'
vol_median = vol_m.reindex(common).median()
df_cond.loc[vol_m.reindex(common) > vol_median, 'vol'] = '高波'
df_cond['mkt_dir'] = '上涨'
df_cond.loc[df_cond['csi300_ret'] < 0, 'mkt_dir'] = '下跌'

# Style x Vol
print('\n等权超额收益 = f(风格, 波动率):')
print(f'{"环境":<28s} {"n":>3s}  {"月均超额":>8s}  {"中位数超额":>8s}  {"胜率":>6s}')
print('-' * 60)
for s in ['小盘偏强','大盘偏强','中性']:
    for v in ['低波','高波']:
        sub = df_cond[(df_cond['style']==s) & (df_cond['vol']==v)]
        if len(sub) > 0:
            label = f'{s} x {v}'
            m = sub['excess_1k'].mean() * 100
            med = sub['excess_1k'].median() * 100
            w = (sub['excess_1k'] > 0).mean() * 100
            tag = ' <<<' if len(sub) < 6 else ''
            print(f'{label:<28s} {len(sub):3d}  {m:+7.1f}%  {med:+7.1f}%  {w:5.0f}%{tag}')

# Style x Market direction
print('\n等权超额收益 = f(风格, 市场方向):')
for s in ['小盘偏强','大盘偏强']:
    for d in ['上涨','下跌']:
        sub = df_cond[(df_cond['style']==s) & (df_cond['mkt_dir']==d)]
        if len(sub) > 0:
            m = sub['excess_1k'].mean() * 100
            w = (sub['excess_1k'] > 0).mean() * 100
            print(f'  {s:6s} x {d:4s}: n={len(sub):2d}, excess={m:+.1f}%/m, win={w:.0f}%')

# ═══════════════════════════════════
# Step 4: 申万行业层面 等权超额收益
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 4: 申万行业等权超额收益 (vs CSI1000)')
print('方法同Q5/Q6 Cell 19: 行业子组合等权平均，与指数基准做差')

sw_returns = {}
for _, row in df55.iterrows():
    code = row['code']
    sw = row['sw']
    if code not in stock_monthly: continue
    if sw not in sw_returns: sw_returns[sw] = []
    sw_returns[sw].append(stock_monthly[code])

sw_port = {}
for sw, rets_list in sw_returns.items():
    if len(rets_list) < 1: continue
    all_dates_t = sorted(set(d for m in rets_list for d in m.index))
    monthly = []
    for dt in all_dates_t:
        vals = [m.loc[dt].iloc[0] for m in rets_list if dt in m.index and np.isfinite(m.loc[dt].iloc[0])]
        if len(vals) >= max(1, len(rets_list)*0.5):
            monthly.append({'date': dt, 'ret': np.mean(vals)})
    if monthly:
        sw_port[sw] = pd.DataFrame(monthly).set_index('date')

sw_summary = []
for sw, df_t in sw_port.items():
    common_t = df_t.index.intersection(csi1k_m.index)
    if len(common_t) < 3: continue
    ex = df_t.loc[common_t, 'ret'] - csi1k_m.loc[common_t].values
    sw_summary.append({
        '申万行业': sw,
        '月数': len(common_t),
        '股数': len(sw_returns[sw]),
        '月均': round(ex.mean()*100, 2),
        '中位数': round(np.median(ex)*100, 2),
        '胜率': round((ex>0).mean()*100, 0),
        '累计': round(((1+pd.Series(ex.values, index=common_t)).cumprod()-1).iloc[-1]*100, 1)
    })

df_sw = pd.DataFrame(sw_summary).sort_values('月均', ascending=False)
print(df_sw.to_string(index=False))

# ═══════════════════════════════════
# Step 5: 当前状态（描述统计，非预测）
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 5: 当前市场状态')
print('以下为描述统计——当前月超额在全期中的位置，以及同标签月份的历史均值。')
print('不构成对未来收益的预测。')

last = df_cond.iloc[-1]
last_dt = df_cond.index[-1]
last_dt_ts = pd.Timestamp(last_dt) if not isinstance(last_dt, pd.Timestamp) else last_dt
recent3 = df_cond.iloc[-3:]

print(f'\n最新数据: {last_dt.date()}')
print(f'  组合月收益: {df_port.loc[last_dt_ts,"ret"]*100:+.1f}%')
print(f'  CSI1000:    {csi1k_m.loc[last_dt_ts]*100:+.1f}%')
print(f'  CSI300:     {csi300_m.loc[last_dt_ts]*100:+.1f}%')
print(f'  超额vs CSI1000: {last["excess_1k"]*100:+.1f}%')
print(f'  超额vs CSI300:  {last["excess_300"]*100:+.1f}%')

cur_s = last['style']; cur_v = last['vol']
match = df_cond[(df_cond['style']==cur_s) & (df_cond['vol']==cur_v)]
print(f'\n  当前环境标签: {cur_s} × {cur_v} (风格基于12月MA方向, 波动率基于中位数二分)')
print(f'  历史相同标签月份 (n={len(match)}, 仅描述统计):')
if len(match) > 0:
    print(f'    月均超额: {match["excess_1k"].mean()*100:+.1f}% (中位数 {match["excess_1k"].median()*100:+.1f}%)')
    print(f'    超额为正的比例: {(match["excess_1k"]>0).mean()*100:.0f}%')

print(f'\n  近3个月:')
for dt, row in recent3.iterrows():
    d = '+' if row['excess_1k'] > 0 else '-'
    print(f'    {dt.date()}: {row["excess_1k"]*100:+.1f}% {d}')

full_excess = excess_1k.dropna()
cur_pct = (full_excess < full_excess.iloc[-1]).mean() * 100
print(f'\n  全期{len(full_excess)}个月中，当前月超额处于 {cur_pct:.0f}% 分位')

print('\nDone.')
