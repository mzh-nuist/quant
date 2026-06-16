"""
从800只CSI300+500成分股中找55只赵哲股的同质标的
==================================================
方法：
1. 特征工程：从日K提取~50维特征（动量/波动率/趋势/相关性/流动性/回撤/市值）
2. 分组：按SW二级行业，≥3只的独立成组，1-2只的合并到SW一级大类
3. 相似度：Mahalanobis距离(可解释) + OneClassSVM(黑箱)
4. 输出：每个SW2组的Top-15候选 + 综合得分

数据来源：
- 800只CSI300+500日K缓存（2021-2026）
- SW二级行业映射（sw_secondary_map.csv，akshare批量拉取）
- 55只赵哲股列表（55个股票.md + sw_secondary_map.csv）
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import mahalanobis
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings, os, re

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')

# ═══════════════════════════════════════
# Step 0: 加载股票池 + 二级行业 + 55只标签
# ═══════════════════════════════════════
print('=' * 60)
print('Step 0: 加载股票池 + SW二级行业映射')

# 800只缓存中的股票
cached_codes = set()
for f in DATA.glob('stock_tx_*.csv'):
    m = re.match(r'stock_tx_(\d+)_', f.name)
    if m:
        cached_codes.add(m.group(1))
print(f'Cached stocks: {len(cached_codes)}')

# SW二级行业映射
df_map = pd.read_csv(DATA / 'sw_secondary_map.csv', dtype={'code': str})
df_map['code'] = df_map['code'].str.zfill(6)
print(f'SW2 mapping: {len(df_map)} stocks, {df_map["sw2"].nunique()} industries')

# CSI300+500成分股
import glob as _glob
xlsx_files = _glob.glob('成分详情*.xlsx')
all_members = set()
for f in xlsx_files:
    df_xl = pd.read_excel(f)
    df_xl['code'] = df_xl.iloc[:, 1].astype(str).str.replace('.SZ', '').str.replace('.SH', '').str.zfill(6)
    all_members.update(df_xl['code'].tolist())
print(f'CSI300+500 members: {len(all_members)}')

# 可用的：有缓存 + 有SW二级 + 在成分股中
usable = cached_codes & set(df_map['code']) & all_members
print(f'Usable stocks: {len(usable)}')

# 55只赵哲股
df55 = pd.read_csv('55个股票.md', sep='\t', header=None)
df55.columns = ['code', 'name']
df55['code'] = df55['code'].astype(str).str.zfill(6)
df55 = df55.merge(df_map[['code', 'sw1', 'sw2']], on='code', how='left')
print(f'55 stocks: {len(df55)}, SW2 unique: {df55["sw2"].nunique()}')

# 分组规则：SW2内≥3只 → 独立组；1-2只 → 合并到SW1大类
sw2_counts = df55.groupby('sw2').size()
standalone_groups = sw2_counts[sw2_counts >= 3].index.tolist()
merged_map = {}
for sw2, cnt in sw2_counts.items():
    if cnt >= 3:
        merged_map[sw2] = sw2  # 独立组名=SW2
    else:
        sw1 = df55[df55['sw2'] == sw2]['sw1'].iloc[0]
        merged_map[sw2] = f'{sw1}(合并)'  # 合并组名=SW1(合并)

df55['group'] = df55['sw2'].map(merged_map)
groups = df55.groupby('group').size().sort_values(ascending=False)
print(f'\nGroup definition (SW2 >= 3 standalone, else merged to SW1):')
for grp, cnt in groups.items():
    sw1s = df55[df55['group'] == grp]['sw1'].unique()
    sw2s = df55[df55['group'] == grp]['sw2'].unique()
    print(f'  {grp}: {cnt} stocks, SW1={list(sw1s)}, SW2={list(sw2s)}')

TARGET_CODES = set(df55['code'])
NON_TARGET = sorted(usable - TARGET_CODES)  # candidates to score
GROUP_NAMES = groups.index.tolist()


# ═══════════════════════════════════════
# Step 1: 特征工程
# ═══════════════════════════════════════
print('\n' + '=' * 60)
print('Step 1: Feature engineering (~50 dims)')

def load_daily(code):
    files = sorted(DATA.glob(f'stock_tx_{code}_*.csv'))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if 'close' in df.columns and len(df) > 10:
                dfs.append(df[['close', 'amount']])
        except Exception:
            continue
    if not dfs:
        return None
    daily = pd.concat(dfs).sort_index()
    daily = daily[~daily.index.duplicated(keep='last')]
    return daily


def extract_features(code, daily, ref_date):
    """
    在 ref_date 处，基于最近12个月日K提取~50维特征。
    所有特征有前例可循（Q1-Q6、leader_trend_failure等）。
    """
    if daily is None or len(daily) < 252:
        return None

    # 截断到 ref_date
    daily = daily[daily.index <= ref_date].copy()
    if len(daily) < 252:
        return None

    close = daily['close']
    rets = close.pct_change().dropna()

    # 窗口定义
    w_1m = close.last('1ME')   # approx 21 trading days
    w_3m = close.last('3ME')
    w_6m = close.last('6ME')
    w_12m = close.last('12ME')

    feats = {}

    # ── 市值 ──
    feats['log_close'] = np.log(max(close.iloc[-1], 0.01))

    # ── 动量类 (10 dims) ──
    if len(close) >= 21:
        feats['ret_1m'] = close.iloc[-1] / close.iloc[-21] - 1
    else:
        feats['ret_1m'] = 0.0
    if len(close) >= 63:
        feats['ret_3m'] = close.iloc[-1] / close.iloc[-63] - 1
    else:
        feats['ret_3m'] = 0.0
    if len(close) >= 126:
        feats['ret_6m'] = close.iloc[-1] / close.iloc[-126] - 1
    else:
        feats['ret_6m'] = 0.0
    if len(close) >= 252:
        feats['ret_12m'] = close.iloc[-1] / close.iloc[-252] - 1
    else:
        feats['ret_12m'] = 0.0

    feats['ret_12m_6m_ratio'] = feats['ret_12m'] / feats['ret_6m'] if abs(feats['ret_6m']) > 0.01 else 1.0
    feats['ret_3m_1m_ratio'] = feats['ret_3m'] / feats['ret_1m'] if abs(feats['ret_1m']) > 0.01 else 1.0

    # 月度正收益占比
    monthly = close.resample('ME').last().pct_change().dropna()
    if len(monthly) >= 3:
        feats['monthly_win_pct'] = (monthly > 0).mean()
        feats['monthly_ret_mean'] = monthly.mean()
        feats['monthly_ret_std'] = monthly.std()
    else:
        feats['monthly_win_pct'] = 0.5
        feats['monthly_ret_mean'] = 0.0
        feats['monthly_ret_std'] = 0.0

    # ── 波动率类 (8 dims) ──
    if len(rets) >= 20:
        feats['vol_20d'] = rets.iloc[-20:].std() * np.sqrt(252)
    else:
        feats['vol_20d'] = np.nan
    if len(rets) >= 60:
        feats['vol_60d'] = rets.iloc[-60:].std() * np.sqrt(252)
    else:
        feats['vol_60d'] = np.nan
    if len(rets) >= 252:
        feats['vol_250d'] = rets.iloc[-252:].std() * np.sqrt(252)
    else:
        feats['vol_250d'] = np.nan

    feats['vol_20_60_ratio'] = feats['vol_20d'] / feats['vol_60d'] if feats['vol_60d'] and feats['vol_60d'] > 0 else 1.0
    feats['vol_60_250_ratio'] = feats['vol_60d'] / feats['vol_250d'] if feats['vol_250d'] and feats['vol_250d'] > 0 else 1.0

    # 下行 vs 上行波动率
    dn = rets[rets < 0]
    up = rets[rets > 0]
    if len(dn) >= 10 and len(up) >= 10:
        feats['down_vol'] = dn.iloc[-60:].std() * np.sqrt(252) if len(dn) >= 20 else dn.std() * np.sqrt(252)
        feats['up_vol'] = up.iloc[-60:].std() * np.sqrt(252) if len(up) >= 20 else up.std() * np.sqrt(252)
        feats['down_up_vol_ratio'] = feats['down_vol'] / feats['up_vol'] if feats['up_vol'] > 0 else 1.0
    else:
        feats['down_up_vol_ratio'] = 1.0

    # ── 趋势结构类 (10 dims) ──
    close_12m = close.iloc[-252:]
    ma5 = close_12m.rolling(5).mean()
    ma10 = close_12m.rolling(10).mean()
    ma20 = close_12m.rolling(20).mean()
    ma60 = close_12m.rolling(60).mean()

    feats['close_above_ma10'] = float(close_12m.iloc[-1] > ma10.iloc[-1]) if len(ma10.dropna()) > 0 else 0.5
    feats['close_above_ma20'] = float(close_12m.iloc[-1] > ma20.iloc[-1]) if len(ma20.dropna()) > 0 else 0.5
    feats['close_above_ma60'] = float(close_12m.iloc[-1] > ma60.iloc[-1]) if len(ma60.dropna()) > 0 else 0.5

    m = pd.DataFrame({'m5': ma5, 'm10': ma10, 'm20': ma20, 'm60': ma60}).dropna()
    if len(m) > 0:
        feats['bull_align_pct'] = float(((m['m5'] > m['m10']) & (m['m10'] > m['m20']) & (m['m20'] > m['m60'])).mean())
    else:
        feats['bull_align_pct'] = 0.0

    feats['pct_above_ma10'] = float((close_12m > ma10).mean()) if len(ma10.dropna()) > 100 else 0.5
    feats['pct_above_ma20'] = float((close_12m > ma20).mean()) if len(ma20.dropna()) > 100 else 0.5
    feats['pct_above_ma60'] = float((close_12m > ma60).mean()) if len(ma60.dropna()) > 100 else 0.5

    # MA spread (normalized)
    if ma60.iloc[-1] > 0:
        feats['ma5_60_spread'] = float((ma5.iloc[-1] - ma60.iloc[-1]) / ma60.iloc[-1])
    else:
        feats['ma5_60_spread'] = 0.0

    # ── 回撤类 (6 dims) ──
    peak_12m = close_12m.max()
    feats['dd_12m'] = float(close_12m.iloc[-1] / peak_12m - 1) if peak_12m > 0 else 0.0
    dd_series = close_12m / close_12m.cummax() - 1
    feats['dd_max_12m'] = float(dd_series.min())
    feats['dd_mean_12m'] = float(dd_series.mean())
    feats['dd_std_12m'] = float(dd_series.std())
    feats['dd_days_in_dd'] = float((dd_series < -0.05).mean())

    # 回撤恢复速度：当前回撤 / 最大回撤
    feats['dd_recovery_ratio'] = feats['dd_12m'] / feats['dd_max_12m'] if feats['dd_max_12m'] < -0.01 else 1.0

    # ── 成交额类 (6 dims) ──
    if 'amount' in daily.columns:
        amt = daily['amount'].astype(float)
        amt_20 = amt.rolling(20).mean()
        amt_60 = amt.rolling(60).mean()
        if amt_60.iloc[-1] > 0:
            feats['amt_20_60_ratio'] = float(amt_20.iloc[-1] / amt_60.iloc[-1])
        else:
            feats['amt_20_60_ratio'] = 1.0
        feats['amt_log_mean'] = np.log(max(amt.iloc[-60:].mean(), 1))
        feats['amt_log_std'] = np.log(max(amt.iloc[-60:].std(), 1))
        amt_cv = amt.iloc[-60:].std() / amt.iloc[-60:].mean() if amt.iloc[-60:].mean() > 0 else 0
        feats['amt_cv_60d'] = float(amt_cv)
        feats['amt_trend'] = float(amt.iloc[-60:].mean() / amt.iloc[-120:-60].mean()) if len(amt) >= 120 else 1.0
    else:
        feats['amt_20_60_ratio'] = 1.0
        feats['amt_log_mean'] = 0.0
        feats['amt_log_std'] = 0.0
        feats['amt_cv_60d'] = 0.0
        feats['amt_trend'] = 1.0

    # ── 与市场指数的相关性类 (8 dims) ──
    for idx_name in ['csi300', 'csi1k']:
        idx_file = DATA / f'{idx_name.upper()}_price.csv' if idx_name == 'csi1k' else DATA / 'CSI300_price.csv'
        if idx_file.exists():
            idx_df = pd.read_csv(idx_file, index_col=0, parse_dates=True)
            idx_df = idx_df[idx_df.index <= ref_date]
            idx_rets = idx_df['close'].pct_change().dropna()
            common = rets.index.intersection(idx_rets.index)
            if len(common) >= 60:
                r = rets.loc[common]
                i = idx_rets.loc[common]
                # 60d beta
                cov = np.cov(r.iloc[-60:], i.iloc[-60:])
                if cov[1, 1] > 0:
                    beta_60 = cov[0, 1] / cov[1, 1]
                else:
                    beta_60 = 1.0
                feats[f'beta_60d_{idx_name}'] = float(beta_60)
                # 60d correlation
                corr_60 = r.iloc[-60:].corr(i.iloc[-60:])
                feats[f'corr_60d_{idx_name}'] = float(corr_60)
                # 250d beta
                if len(common) >= 250:
                    cov250 = np.cov(r.iloc[-250:], i.iloc[-250:])
                    feats[f'beta_250d_{idx_name}'] = float(cov250[0, 1] / cov250[1, 1]) if cov250[1, 1] > 0 else 1.0
                else:
                    feats[f'beta_250d_{idx_name}'] = beta_60
            else:
                feats[f'beta_60d_{idx_name}'] = 1.0
                feats[f'corr_60d_{idx_name}'] = 0.5
                feats[f'beta_250d_{idx_name}'] = 1.0
        else:
            feats[f'beta_60d_{idx_name}'] = 1.0
            feats[f'corr_60d_{idx_name}'] = 0.5
            feats[f'beta_250d_{idx_name}'] = 1.0

    return feats


# 加载市场指数数据
csi1k_idx = pd.read_csv(DATA / 'CSI1000_price.csv', index_col=0, parse_dates=True)
csi300_idx = pd.read_csv(DATA / 'CSI300_price.csv', index_col=0, parse_dates=True)
REF_DATE = csi1k_idx.index[-1]  # 使用最新可用日期
print(f'Reference date (latest data): {REF_DATE.date()}')

# 提取所有可用股票的特征
from tqdm import tqdm

FEATURES = {}
print(f'Extracting features from {len(usable)} stocks...')
for code in tqdm(sorted(usable)):
    daily = load_daily(code)
    feats = extract_features(code, daily, REF_DATE)
    if feats is not None:
        FEATURES[code] = feats

print(f'Features extracted for {len(FEATURES)} stocks')

# 构建特征矩阵
df_feat = pd.DataFrame(FEATURES).T
df_feat.index.name = 'code'

# 填充缺失值
df_feat = df_feat.fillna(df_feat.median())

print(f'Feature matrix: {df_feat.shape}')
print(f'Feature names: {list(df_feat.columns)}')

# ═══════════════════════════════════════
# Step 2: 标准化 + 分组画像
# ═══════════════════════════════════════
print('\n' + '=' * 60)
print('Step 2: Standardization + Group Profiling')

scaler = StandardScaler()
X = scaler.fit_transform(df_feat)
df_scaled = pd.DataFrame(X, index=df_feat.index, columns=df_feat.columns)

# 分组中心
target_in_features = df55[df55['code'].isin(df_scaled.index)]
print(f'55 stocks with features: {len(target_in_features)}')

# 每个组的关键特征
GROUP_CENTROIDS = {}
for grp in GROUP_NAMES:
    codes_in_grp = target_in_features[target_in_features['group'] == grp]['code'].tolist()
    codes_in_grp = [c for c in codes_in_grp if c in df_scaled.index]
    if len(codes_in_grp) == 0:
        continue
    centroid = df_scaled.loc[codes_in_grp].mean()
    GROUP_CENTROIDS[grp] = {'centroid': centroid, 'codes': codes_in_grp, 'n': len(codes_in_grp)}

    # Top-5 区分特征（与该组均价 vs 全市场均价的差异）
    diff = centroid - df_scaled.mean()
    top5 = diff.abs().sort_values(ascending=False).head(5)
    print(f'\n  {grp} (n={len(codes_in_grp)}):')
    for feat, val in top5.items():
        direction = '+' if diff[feat] > 0 else '-'
        print(f'    {feat}: {direction}{val:.2f}σ')

# ═══════════════════════════════════════
# Step 3a: Mahalanobis 距离评分
# ═══════════════════════════════════════
print('\n' + '=' * 60)
print('Step 3a: Mahalanobis Distance Scoring')

# 对每个组，计算 pooled covariance（用全量样本的协方差，避免小样本问题）
global_cov = np.cov(df_scaled.values.T)
global_cov_inv = np.linalg.pinv(global_cov + np.eye(len(df_scaled.columns)) * 1e-4)

maha_scores = {}
for grp, info in GROUP_CENTROIDS.items():
    centroid = info['centroid']
    scores = []
    for code in df_scaled.index:
        d = mahalanobis(df_scaled.loc[code], centroid, global_cov_inv)
        scores.append({'code': code, 'score': -d})  # negative → higher = closer
    df_scores = pd.DataFrame(scores).sort_values('score', ascending=False)
    maha_scores[grp] = df_scores

# 合并：每个 stock 取到最近组的距离（最大 score）
all_codes = df_scaled.index.tolist()
maha_best = {}
for code in all_codes:
    best_score = max(maha_scores[grp].set_index('code').loc[code, 'score'] for grp in GROUP_CENTROIDS)
    maha_best[code] = best_score

df_maha = pd.DataFrame({'code': list(maha_best.keys()), 'maha_score': list(maha_best.values())})
df_maha = df_maha.sort_values('maha_score', ascending=False)
print(f'Mahalanobis scoring done. Top-20 candidates:')
for _, row in df_maha.head(20).iterrows():
    code = row['code']
    name = df55[df55['code'] == code]['name'].values
    label = f' (55: {name[0]})' if len(name) > 0 else ''
    m_score = row['maha_score']
    print(f'  {code}{label}: {m_score:.4f}')


# ═══════════════════════════════════════
# Step 3b: OneClassSVM
# ═══════════════════════════════════════
print('\n' + '=' * 60)
print('Step 3b: OneClassSVM Scoring')

target_codes_in_matrix = [c for c in TARGET_CODES if c in df_scaled.index]
X_train = df_scaled.loc[target_codes_in_matrix].values
X_all = df_scaled.values

# 训练 OCSVM
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
svm.fit(X_train)

# 对所有股票打分
svm_scores = svm.decision_function(X_all)

df_svm = pd.DataFrame({
    'code': df_scaled.index,
    'svm_score': svm_scores
}).sort_values('svm_score', ascending=False)

print(f'OCSVM trained on {len(target_codes_in_matrix)} positive samples.')
print(f'Top-20 SVM candidates:')
for _, row in df_svm.head(20).iterrows():
    code = row['code']
    is_target = code in TARGET_CODES
    tag = ' [55]' if is_target else ''
    svm_val = row['svm_score']
    print(f'  {code}{tag}: {svm_val:.4f}')

# 55只在SVM排名中的分布
target_svm = df_svm[df_svm['code'].isin(TARGET_CODES)]
print(f'\n55只在SVM中的排名分布:')
print(f'  P25 rank: {(target_svm.index < len(df_svm)*0.25).sum()}/{len(target_svm)}')
print(f'  P50 rank: {(target_svm.index < len(df_svm)*0.5).sum()}/{len(target_svm)}')
print(f'  Top-100: {len(target_svm[target_svm.index < 100])}/{len(target_svm)}')


# ═══════════════════════════════════════
# Step 4: 综合排序 + 按组输出
# ═══════════════════════════════════════
print('\n' + '=' * 60)
print('Step 4: Combined Ranking by Group')

# 合并两个分数（标准化后取平均）
df_maha['maha_z'] = (df_maha['maha_score'] - df_maha['maha_score'].mean()) / df_maha['maha_score'].std()
df_svm['svm_z'] = (df_svm['svm_score'] - df_svm['svm_score'].mean()) / df_svm['svm_score'].std()

df_combined = df_maha[['code', 'maha_score', 'maha_z']].merge(
    df_svm[['code', 'svm_score', 'svm_z']], on='code', how='inner'
)
df_combined['combined'] = (df_combined['maha_z'] + df_combined['svm_z']) / 2
df_combined = df_combined.sort_values('combined', ascending=False)

# 加上标签
code_to_name = {}
for _, row in df55.iterrows():
    code_to_name[row['code']] = row['name']

code_to_group = {}
for _, row in df55.iterrows():
    code_to_group[row['code']] = row['group']

# 输出：每个组的 Top-N 候选（排除原55只）
print('\n' + '=' * 60)
print('TOP CANDIDATES PER GROUP (excluding original 55)')
print('=' * 60)

for grp in GROUP_NAMES:
    grp_codes = set(target_in_features[target_in_features['group'] == grp]['code'])
    n_in_grp = len(grp_codes & set(df_scaled.index))

    # 对于合并组，选择所有相关SW2的候选
    sw2s_in_grp = df55[df55['group'] == grp]['sw2'].unique()
    # 在同一SW1下的候选池中找
    sw1_of_grp = df55[df55['group'] == grp]['sw1'].iloc[0]

    candidates = df_combined[~df_combined['code'].isin(TARGET_CODES)].copy()
    candidates['sw2'] = candidates['code'].map(
        lambda c: df_map[df_map['code'] == c]['sw2'].values[0] if c in df_map['code'].values else 'N/A'
    )
    candidates['sw1'] = candidates['code'].map(
        lambda c: df_map[df_map['code'] == c]['sw1'].values[0] if c in df_map['code'].values else 'N/A'
    )

    # 优先同SW2，其次同SW1
    candidates['sw_priority'] = candidates['sw2'].apply(lambda s: 2 if s in sw2s_in_grp else (1 if s == sw1_of_grp else 0))

    top = candidates.sort_values(['sw_priority', 'combined'], ascending=[False, False]).head(15)

    print(f'\n{grp} ({n_in_grp} original 55, {len(sw2s_in_grp)} SW2)')
    print(f'  SW2: {list(sw2s_in_grp)}')
    print(f'  Top candidates:')
    for _, row in top.iterrows():
        code = row['code']
        sw2 = row['sw2']
        name_info = code_to_name.get(code, '')
        c_score = row['combined']
        m_score = row['maha_z']
        s_score = row['svm_z']
        print(f'    {code} {name_info:<10s} SW2={sw2:<12s}  combined={c_score:+.3f}  maha={m_score:+.2f}  svm={s_score:+.2f}')

# 保存完整评分表
df_combined.to_csv('research_cache/similarity_scores.csv', index=False)
print(f'\nFull scores saved to research_cache/similarity_scores.csv')

print('\nDone.')
