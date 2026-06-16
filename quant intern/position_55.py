"""
Step 4: 55只在龙头池中的定位（申万行业版）
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings, os, glob as _glob
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE55 = Path('research_cache/55stock')

print('=' * 60)
print('Step 1: 加载数据')

df55 = pd.read_csv('55个股票.md', sep='\t', header=None)
df55.columns = ['code','name']
df55['code'] = df55['code'].astype(str).str.zfill(6)

# 申万行业: 34只从xlsx + 21只从cninfo映射
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

xlsx_files = _glob.glob('成分详情*.xlsx')
csi300 = pd.read_excel([f for f in xlsx_files if '000300' in f][0])
csi500 = pd.read_excel([f for f in xlsx_files if '000905' in f][0])

cols_use = {1: 'code', 2: 'name', 11: 'total_mv', 14: 'industry'}
csi300 = csi300.iloc[:, list(cols_use.keys())].copy()
csi300.columns = list(cols_use.values())
csi500 = csi500.iloc[:, list(cols_use.keys())].copy()
csi500.columns = list(cols_use.values())

df_all = pd.concat([csi300, csi500], ignore_index=True)
df_all['code'] = df_all['code'].astype(str).str.replace('.SZ','').str.replace('.SH','').str.zfill(6)
df_all = df_all.drop_duplicates(subset=['code']).dropna(subset=['industry','total_mv'])
df_all['total_mv'] = pd.to_numeric(df_all['total_mv'], errors='coerce')
df_all = df_all[df_all['total_mv'] > 0]
df_all = df_all[df_all['industry'] != '—']

leaders = []
for ind, grp in df_all.groupby('industry'):
    grp_sorted = grp.sort_values('total_mv', ascending=False)
    for rank, (_, row) in enumerate(grp_sorted.head(3).iterrows()):
        leaders.append({'industry': ind, 'rank': rank+1, 'code': row['code'],
                        'name': row['name'], 'total_mv': row['total_mv']})
df_leader = pd.DataFrame(leaders)

df_panel = pd.read_csv(CACHE55 / 'leader_panel.csv', parse_dates=['date'])
print(f'55只: {len(df55)}只, 龙头池: {len(df_leader)}只, 面板: {len(df_panel)}条')

# ═════════════════════════════════════
# Step 2: 匹配
# ═════════════════════════════════════
print('\n' + '=' * 60)
print('Step 2: 55只 vs 龙头池匹配')

codes55 = set(df55['code'])
leader_codes = set(df_leader['code'])
overlap = codes55 & leader_codes
not_in_leader = codes55 - leader_codes

print(f'在龙头池中: {len(overlap)}/{len(df55)} ({100*len(overlap)/len(df55):.0f}%)')
print(f'不在龙头池中: {len(not_in_leader)}')

if not_in_leader:
    print('\n不在龙头池中的股票:')
    for code in sorted(not_in_leader):
        row = df55[df55['code'] == code].iloc[0]
        row_all = df_all[df_all['code'] == code]
        if len(row_all) > 0:
            row_all = row_all.iloc[0]
            ind_grp = df_all[df_all['industry'] == row_all['industry']].sort_values('total_mv', ascending=False)
            rank = ind_grp[ind_grp['code'] == code].index
            rank_num = ind_grp.index.get_loc(rank[0]) + 1 if len(rank) > 0 else '?'
            print(f'  {code} {row["name"]} | {row_all["industry"]} | 市值{row_all["total_mv"]:.0f}亿 | 行业排名#{rank_num}')
        else:
            print(f'  {code} {row["name"]} | 不在CSI300/500成分中')

# ═════════════════════════════════════
# Step 3: 申万行业分布 vs 龙头池覆盖
# ═════════════════════════════════════
print('\n' + '=' * 60)
print('Step 3: 申万行业 x 龙头池覆盖')

df55['is_leader'] = df55['code'].isin(leader_codes)
df55_m = df55.merge(df_all[['code','total_mv']], on='code', how='left')

sw_summary = df55_m.groupby('sw').agg(
    n=('code', 'size'),
    在龙头池=('is_leader', 'sum'),
    不在龙头池=('is_leader', lambda x: (~x).sum()),
    平均市值=('total_mv', 'mean'),
).sort_values('n', ascending=False)
sw_summary['平均市值'] = sw_summary['平均市值'].round(0)
sw_summary['龙头覆盖率'] = (sw_summary['在龙头池'] / sw_summary['n'] * 100).round(0).astype(int)
print(sw_summary.to_string())

# ═════════════════════════════════════
# Step 4: 所在行业历史龙头失效率
# ═════════════════════════════════════
print('\n' + '=' * 60)
print('Step 4: 所在行业 x 风格 龙头失效率')

industries_55 = df55['sw'].unique().tolist()

df_panel['style_simple'] = '中性'
monthly_excess = df_panel.groupby('date')['excess_ret'].mean()
style_map = {}
for dt, val in monthly_excess.items():
    if val > 0.003:
        style_map[dt] = '小盘偏强'
    elif val < -0.003:
        style_map[dt] = '大盘偏强'
    else:
        style_map[dt] = '中性'
df_panel['style_simple'] = df_panel['date'].map(style_map)

# The leader panel uses 'industry' column (Shenwan L1), match to our 'sw'
for ind in sorted(industries_55):
    sub = df_panel[df_panel['industry'] == ind]
    if len(sub) == 0:
        print(f'\n  {ind}: 无龙头数据')
        continue
    print(f'\n  {ind}:')
    for style in ['小盘偏强','大盘偏强','中性']:
        ssub = sub[sub['style_simple'] == style]
        if len(ssub) < 3: continue
        fr = ssub['is_fail'].mean() * 100
        me = ssub['excess_ret'].mean() * 100
        tag = '危险' if fr > 55 else ('安全' if fr < 45 else '中性')
        print(f'    [{tag}] {style:6s} | 失效率{fr:.0f}% | 超额{me:+.2f}% (n={len(ssub)})')

print('\nDone.')
