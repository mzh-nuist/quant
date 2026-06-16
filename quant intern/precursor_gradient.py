"""
龙头趋势失效前兆深挖 — 时间梯度 + 成交额维度
=================================================
在现有 analyze_leader_failure.py 模块3的基础上做三件事：
  A. 扩展信号到 T-3, T-2, T-1, T 四个时间点
  B. 加入成交额维度（amount ratio, amount CV）
  C. 检测恶化模式（单调恶化、最后突变、恶化计数）
  D. 复合预警评分

所有方法有前例可循。无自编指标。非预测——仅描述统计。
"""
import numpy as np
import pandas as pd
from pathlib import Path
import warnings, os

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA = Path('research_cache')
CACHE = Path('research_cache/leader_trend')

# ═══════════════════════════════════
# A. 统一版 load_daily（加载 close + amount）
# ═══════════════════════════════════
def load_daily(code, cols=None):
    """加载个股日K线。同 Q5/Q6 extend_years.py 逻辑，扩展到多列。"""
    if cols is None:
        cols = ['close', 'amount']
    files = sorted(DATA.glob(f'stock_tx_{code}_*.csv'))
    if not files:
        return None, None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            available = [c for c in cols if c in df.columns]
            if available and len(df) > 10:
                dfs.append(df[available])
        except Exception:
            continue
    if not dfs:
        return None, None
    daily = pd.concat(dfs).sort_index()
    daily = daily[~daily.index.duplicated(keep='last')]
    monthly = daily.resample('ME').last()
    return daily, monthly


# ═══════════════════════════════════
# Step 0: 加载基础数据
# ═══════════════════════════════════
print('=' * 60)
print('Step 0: 加载龙头面板 + 市场指标 + 行业指数')

df_panel = pd.read_csv(CACHE / 'leader_panel_monthly.csv', parse_dates=['date'])
print(f'Leader panel: {len(df_panel)} rows, {df_panel["code"].nunique()} stocks')

# 构建趋势段（同 analyze_leader_failure.py 逻辑）
df = df_panel.sort_values(['code', 'date']).copy()
df['month_diff'] = df.groupby('code')['date'].diff().dt.days.fillna(999)
df['episode_id'] = (df['month_diff'] > 40).cumsum()

episodes = df.groupby(['episode_id', 'code', 'sw']).agg(
    start_date=('date', 'min'),
    end_date=('date', 'max'),
    months=('date', 'count'),
    avg_ret_12m=('ret_12m', 'mean'),
).reset_index()

print(f'Trend episodes: {len(episodes)}')

# 市场指标
csi1k = pd.read_csv(DATA / 'CSI1000_price.csv', index_col=0, parse_dates=True)
csi300 = pd.read_csv(DATA / 'CSI300_price.csv', index_col=0, parse_dates=True)

# 风格方向：CSI1000/CSI300 12月MA方向（Q1 1D方法论）
ratio = csi1k['close'] / csi300['close']
ratio_ma12 = ratio.rolling(252, min_periods=126).mean()
ratio_ma12_m = ratio_ma12.resample('ME').last().dropna()
ma_direction = ratio_ma12_m.diff()  # >0 = 小盘偏强, <0 = 大盘偏强

# CSI300 波动率
csi300_vol = csi300['close'].pct_change().rolling(60).std() * np.sqrt(252)
vol_m = csi300_vol.resample('ME').last()

# SW 行业等权日收益（同 Q5/Q6 方法，自建）
df_sw = pd.read_csv(CACHE / 'sw_industry_daily_ew.csv', index_col=0, parse_dates=True)
sw_monthly_ret = df_sw.resample('ME').agg(lambda x: (1 + x).prod() - 1)

print(f'SW industries: {list(df_sw.columns)}')
print(f'Market data range: {ratio_ma12_m.index[0].date()} ~ {ratio_ma12_m.index[-1].date()}')


# ═══════════════════════════════════
# B. 时间梯度信号计算
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 1: 时间梯度信号计算 (T-3, T-2, T-1, T)')

# 信号定义
# 对每个趋势段，在 end_date 前 offset=0,-1,-2,-3 个月分别计算
# offset=0 是结束当月（T），offset=-3 是结束前 3 个月（T-3）

SIGNAL_OFFSETS = [0, -1, -2, -3]  # T, T-1, T-2, T-3

# 恶化方向定义：True = 值越小越恶化（需要反转的在下游处理）
# 以下信号"值越小 = 越恶化"：
#   mom_ratio, align_pct, amt_ratio, excess_sw, drawdown
# 以下信号"值越大 = 越恶化"（需要取反）：
#   below_ma10, below_ma20, vol_ratio, amt_cv


def compute_signals_at(daily, check_dt, trend_high):
    """
    在给定时间点计算所有前兆信号。
    daily: 个股日K DataFrame (index=date, columns 含 close, amount)
    check_dt: 检查的时间点 (Timestamp)
    trend_high: 趋势段内最高收盘价
    返回 dict
    """
    if daily is None or len(daily) < 252:
        return None

    # 找最近交易日
    idx = daily.index.get_indexer([check_dt], method='ffill')
    if idx[0] < 0:
        return None
    end_loc = idx[0]
    if end_loc < 120:  # 需要至少 120 个交易日做滚动计算
        return None

    window = daily.iloc[max(0, end_loc - 120):end_loc + 1]
    close = window['close']

    # ── 价格结构 ──
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    below_ma10 = float(close.iloc[-1] < ma10.iloc[-1])
    below_ma20 = float(close.iloc[-1] < ma20.iloc[-1])

    # ── MA 多头排列占比 ──
    ma5 = close.rolling(5).mean()
    ma60_roll = close.rolling(60).mean()
    aligned = pd.DataFrame({'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma60': ma60_roll}).dropna()
    bullish = (aligned['ma5'] > aligned['ma10']) & \
              (aligned['ma10'] > aligned['ma20']) & \
              (aligned['ma20'] > aligned['ma60'])
    align_pct = float(bullish.mean()) if len(bullish) > 0 else 0.0

    # ── 动量衰减 ──
    if len(close) >= 250:
        ret_60 = close.iloc[-1] / close.iloc[-61] - 1
        ret_250 = close.iloc[-1] / close.iloc[0] - 1
        mom_ratio = ret_60 / ret_250 if abs(ret_250) > 0.01 else 1.0
    elif len(close) >= 61:
        ret_60 = close.iloc[-1] / close.iloc[-61] - 1
        mom_ratio = ret_60 / (close.iloc[-1] / close.iloc[0] - 1) if abs(close.iloc[-1] / close.iloc[0] - 1) > 0.01 else 1.0
    else:
        mom_ratio = 1.0

    # ── 波动率突变 ──
    rets = close.pct_change().dropna()
    if len(rets) >= 60:
        v20 = rets.iloc[-20:].std() * np.sqrt(252)
        v60 = rets.iloc[-60:].std() * np.sqrt(252)
        vol_ratio = float(v20 / v60) if v60 > 0 else 1.0
    else:
        vol_ratio = np.nan

    # ── 成交额信号（新） ──
    amt_ratio = np.nan
    amt_cv = np.nan
    if 'amount' in window.columns:
        amt = window['amount'].astype(float)
        amt_ma20 = amt.rolling(20).mean()
        amt_ma60 = amt.rolling(60).mean()
        if amt_ma60.iloc[-1] > 0:
            amt_ratio = float(amt_ma20.iloc[-1] / amt_ma60.iloc[-1])
        amt_20_std = amt.rolling(20).std()
        amt_20_mean = amt.rolling(20).mean()
        if amt_20_mean.iloc[-1] > 0:
            amt_cv = float(amt_20_std.iloc[-1] / amt_20_mean.iloc[-1])

    # ── 回撤 ──
    drawdown = float(close.iloc[-1] / trend_high - 1) if trend_high > 0 else 0.0

    return {
        'below_ma10': below_ma10,
        'below_ma20': below_ma20,
        'align_pct': align_pct,
        'mom_ratio': mom_ratio if np.isfinite(mom_ratio) else 1.0,
        'vol_ratio': vol_ratio if np.isfinite(vol_ratio) else np.nan,
        'amt_ratio': amt_ratio if np.isfinite(amt_ratio) else np.nan,
        'amt_cv': amt_cv if np.isfinite(amt_cv) else np.nan,
        'drawdown': drawdown if np.isfinite(drawdown) else 0.0,
    }


def get_style(excess_sw_val, check_dt):
    """获取市场风格标签"""
    style = '中性'
    if check_dt in ma_direction.index:
        d = ma_direction.loc[check_dt]
        if d > 0:
            style = '小盘偏强'
        elif d < 0:
            style = '大盘偏强'
    return style


# 对每个趋势段计算时间梯度
gradient_rows = []
n_skipped = 0
n_processed = 0

# 按 code 分组加载，避免重复 IO
codes_with_episodes = episodes['code'].unique()
print(f'Processing {len(codes_with_episodes)} unique stocks across {len(episodes)} episodes...')

# 先按 code 加载日K并缓存
daily_cache = {}
for code in codes_with_episodes:
    daily, _ = load_daily(code)
    if daily is not None and 'close' in daily.columns:
        daily_cache[code] = daily

print(f'Loaded daily data for {len(daily_cache)}/{len(codes_with_episodes)} stocks')

for _, ep in episodes.iterrows():
    code = ep['code']
    sw = ep['sw']
    end_dt = pd.Timestamp(ep['end_date'])
    start_dt = pd.Timestamp(ep['start_date'])

    daily = daily_cache.get(code)
    if daily is None:
        n_skipped += 1
        continue

    # 找 trend_high：趋势段内（start 到 end）的最高收盘价
    trend_period = daily.loc[start_dt:end_dt]
    if len(trend_period) == 0:
        # fallback：扩展到 end_date 之前的数据
        trend_period = daily.loc[:end_dt].iloc[-120:]
    trend_high = float(trend_period['close'].max()) if len(trend_period) > 0 else 0.0

    # 计算 4 个时间点的信号
    signals_at_t = {}
    valid_offsets = []
    for offset in SIGNAL_OFFSETS:
        check_dt = end_dt + pd.DateOffset(months=offset)
        sig = compute_signals_at(daily, check_dt, trend_high)
        if sig is not None:
            sig['style'] = get_style(None, check_dt)
            signals_at_t[offset] = sig
            valid_offsets.append(offset)

    if len(valid_offsets) < 2:  # 至少需要 2 个时间点才能看趋势
        n_skipped += 1
        continue

    # 行业超额（在月度频率上计算）
    excess_sw_vals = {}
    if sw in sw_monthly_ret.columns:
        stock_monthly = daily['close'].resample('ME').last().pct_change().dropna()
        ind_monthly = sw_monthly_ret[sw].dropna()
        for offset in valid_offsets:
            check_dt = end_dt + pd.DateOffset(months=offset)
            if check_dt in stock_monthly.index and check_dt in ind_monthly.index:
                excess_sw_vals[offset] = float(
                    stock_monthly.loc[check_dt] - ind_monthly.loc[check_dt])

    # 收集该 episode 的所有时间点数据
    row_base = {
        'episode_id': ep['episode_id'],
        'code': code,
        'sw': sw,
        'start_date': start_dt,
        'end_date': end_dt,
        'months': ep['months'],
    }

    for offset in valid_offsets:
        sig = signals_at_t[offset]
        row = row_base.copy()
        row['offset'] = offset
        row['time_label'] = f'T{offset}' if offset < 0 else 'T'
        for k, v in sig.items():
            row[k] = v
        row['excess_sw'] = excess_sw_vals.get(offset, np.nan)
        gradient_rows.append(row)

    n_processed += 1

print(f'\nProcessed: {n_processed} episodes, Skipped: {n_skipped}')
print(f'Gradient rows: {len(gradient_rows)}')


# ═══════════════════════════════════
# C. 恶化模式检测
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 2: 恶化模式检测')

df_grad = pd.DataFrame(gradient_rows)

# 定义每个信号的恶化方向
# direction=1: 值上升 = 恶化 (如 below_ma10, vol_ratio)
# direction=-1: 值下降 = 恶化 (如 align_pct, mom_ratio, amt_ratio, drawdown)
SIGNAL_DIRECTION = {
    'below_ma10': 1,       # 跌破均线 = 恶化
    'below_ma20': 1,       # 跌破均线 = 恶化
    'align_pct': -1,       # 多头排列占比下降 = 恶化
    'mom_ratio': -1,       # 动量比率下降 = 恶化
    'vol_ratio': 1,        # 波动率飙升 = 恶化（>1.2 阈值）
    'amt_ratio': -1,       # 成交额收缩 = 恶化
    'amt_cv': 1,           # 成交额离散上升 = 恶化
    'drawdown': -1,        # 回撤加深(更负) = 恶化
    'excess_sw': -1,       # 行业超额下降 = 恶化
}

# 特殊阈值信号（不是纯方向性的）
# vol_ratio > 1.2 才算恶化（轻微波动不算）
# amt_cv > 某个阈值才算恶化

# 对每个 episode，整理 T-3 → T-2 → T-1 → T 的信号变化
ep_deterioration = []

for ep_id, grp in df_grad.groupby('episode_id'):
    grp = grp.sort_values('offset', ascending=False)  # T-3, T-2, T-1, T
    offsets = grp['offset'].tolist()
    if len(offsets) < 2:
        continue

    record = {
        'episode_id': ep_id,
        'code': grp['code'].iloc[0],
        'sw': grp['sw'].iloc[0],
        'months': grp['months'].iloc[0],
    }

    for sig_name, direction in SIGNAL_DIRECTION.items():
        vals = grp[sig_name].dropna().tolist()
        if len(vals) < 2:
            record[f'{sig_name}_mono'] = 0
            record[f'{sig_name}_jump'] = 0
            record[f'{sig_name}_bad_count'] = 0
            record[f'{sig_name}_T'] = np.nan
            record[f'{sig_name}_Tminus3'] = np.nan
            continue

        # 记录 T 和 T-3 的值
        t_vals = grp[grp['offset'] == 0][sig_name].dropna()
        t3_vals = grp[grp['offset'] == -3][sig_name].dropna()
        record[f'{sig_name}_T'] = float(t_vals.iloc[0]) if len(t_vals) > 0 else np.nan
        record[f'{sig_name}_Tminus3'] = float(t3_vals.iloc[0]) if len(t3_vals) > 0 else np.nan

        # 1. 单调恶化：是否持续朝坏方向走
        # 对方向性信号，检查相邻差分是否都同向（恶化方向）
        diffs = []
        for i in range(1, len(vals)):
            d = vals[i] - vals[i - 1]
            diffs.append(d * direction)  # 正 = 朝恶化方向变化
        # 单调恶化 = 所有相邻差分都 >= 0 且至少一个 > 0
        # 注意：below_ma10/ma20 是 0/1 值，略有不同
        if sig_name in ['below_ma10', 'below_ma20']:
            # 二元信号：0→0→1→1 算单调恶化
            mono = all(d >= 0 for d in diffs) and any(d > 0 for d in diffs)
        else:
            # 连续信号：要求每个 step 都朝恶化方向变化，且总量 > 0
            total_change = direction * (vals[-1] - vals[0])
            mono = all(d >= 0 for d in diffs) and total_change > 0
        record[f'{sig_name}_mono'] = int(mono)

        # 2. 最后突变：T-1 → T 的跳变幅度 > 前两期平均跳变的 2 倍
        if len(diffs) >= 2:
            last_jump = abs(diffs[-1])
            avg_prev = np.mean([abs(d) for d in diffs[:-1]]) if len(diffs) > 1 else 0
            jump = last_jump > 2.0 * avg_prev if avg_prev > 1e-8 else (last_jump > 0.1)
            record[f'{sig_name}_jump'] = int(jump)
        else:
            record[f'{sig_name}_jump'] = 0

        # 3. 恶化计数：4 个时间点中几个在"坏"状态
        if sig_name in ['below_ma10', 'below_ma20']:
            bad = sum(v > 0.5 for v in vals)  # > 0.5 即 True
        elif sig_name in ['vol_ratio']:
            bad = sum(v > 1.2 for v in vals)  # 波动率飙升阈值
        elif sig_name in ['amt_cv']:
            # 变异系数：超过全样本中位数算恶化（动态阈值）
            med = df_grad['amt_cv'].dropna().median()
            bad = sum(v > med for v in vals) if not np.isnan(med) else 0
        elif sig_name in ['drawdown']:
            bad = sum(v < -0.05 for v in vals)  # 回撤 > 5%
        elif sig_name in ['excess_sw']:
            bad = sum(v < 0 for v in vals)  # 超额为负
        else:
            # 连续信号：比 T-3 更差就算坏
            if len(vals) >= 2 and direction == -1:
                bad = sum(1 for v in vals if v < vals[0]) + (1 if direction == -1 and vals[-1] < vals[0] else 0)
                bad = min(bad, sum(1 for v in vals if v < (vals[0] if len(vals) >= 2 else 0)))
            else:
                bad = sum(1 for v in vals if v > vals[0]) if direction == 1 else 0
        record[f'{sig_name}_bad_count'] = bad

    ep_deterioration.append(record)

df_det = pd.DataFrame(ep_deterioration)
print(f'Deterioration records: {len(df_det)} episodes')

# ── 单信号恶化覆盖率 ──
print('\n单信号恶化模式覆盖失效的比例:')
print(f'{"信号":<18s} {"单调恶化":>8s} {"最后突变":>8s} {"≥2月恶化":>8s} {"T时异常":>8s}')
print('-' * 56)

for sig_name in SIGNAL_DIRECTION:
    mono_col = f'{sig_name}_mono'
    jump_col = f'{sig_name}_jump'
    bad_col = f'{sig_name}_bad_count'
    t_col = f'{sig_name}_T'

    if mono_col not in df_det.columns:
        continue

    n = len(df_det)
    mono_pct = df_det[mono_col].mean() * 100
    jump_pct = df_det[jump_col].mean() * 100
    bad2_pct = (df_det[bad_col] >= 2).mean() * 100

    # "T时异常"的定义
    if sig_name == 'below_ma10':
        t_bad = df_det[t_col].mean() * 100  # T 时跌破 MA10 的比例
    elif sig_name == 'below_ma20':
        t_bad = df_det[t_col].mean() * 100
    elif sig_name == 'drawdown':
        t_bad = (df_det[t_col] < -0.05).mean() * 100
    elif sig_name == 'excess_sw':
        t_bad = (df_det[t_col] < 0).mean() * 100
    elif sig_name == 'vol_ratio':
        t_bad = (df_det[t_col] > 1.2).mean() * 100
    else:
        t_bad = np.nan

    t_str = f'{t_bad:.0f}%' if not np.isnan(t_bad) else 'N/A'
    print(f'{sig_name:<18s} {mono_pct:7.0f}% {jump_pct:7.0f}% {bad2_pct:7.0f}% {t_str:>8s}')


# ═══════════════════════════════════
# D. 复合预警评分
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 3: 复合预警评分')

# 合并多个恶化信号
# 每个信号贡献 0/1：是否存在任意恶化模式（单调 或 突变 或 ≥2月坏）
# 然后汇总为 0-N 分

for sig_name in SIGNAL_DIRECTION:
    mono_col = f'{sig_name}_mono'
    jump_col = f'{sig_name}_jump'
    bad_col = f'{sig_name}_bad_count'
    if mono_col not in df_det.columns:
        continue
    df_det[f'{sig_name}_warn'] = (
        (df_det[mono_col] == 1) |
        (df_det[jump_col] == 1) |
        (df_det[bad_col] >= 2)
    ).astype(int)

# 选出最有效的信号（恶化覆盖率 > 20% 的信号才算有用）
warn_cols = []
for sig_name in SIGNAL_DIRECTION:
    col = f'{sig_name}_warn'
    if col in df_det.columns:
        coverage = df_det[col].mean()
        if coverage > 0.20:
            warn_cols.append(col)
            print(f'  {sig_name}: {coverage*100:.0f}% 覆盖 → 纳入评分')
        else:
            print(f'  {sig_name}: {coverage*100:.0f}% 覆盖 → 剔除（<20%）')

if warn_cols:
    df_det['warn_score'] = df_det[warn_cols].sum(axis=1)
    max_score = len(warn_cols)
    print(f'\n复合预警评分 (0-{max_score}分，{len(warn_cols)}个有效信号):')
    for score in range(max_score + 1):
        n = (df_det['warn_score'] >= score).sum()
        pct = n / len(df_det) * 100
        print(f'  ≥{score}分: {n}/{len(df_det)} ({pct:.0f}%)')

    # 高分段特征
    high_warn = df_det[df_det['warn_score'] >= max(1, max_score // 2)]
    if len(high_warn) > 0:
        print(f'\n高预警段 (≥{max(1, max_score//2)}分): {len(high_warn)} 段')
        print(f'  平均持续月数: {high_warn["months"].mean():.1f}')
        print(f'  行业分布(top 5):')
        for sw, cnt in high_warn.groupby('sw').size().sort_values(ascending=False).head(5).items():
            print(f'    {sw}: {cnt}')
else:
    print('\n警告：没有恶化覆盖率 > 20% 的信号，无法构建有效复合评分。')
    df_det['warn_score'] = 0


# ═══════════════════════════════════
# E. 总结
# ═══════════════════════════════════
print('\n' + '=' * 60)
print('Step 4: 总结')

print(f'\n分析样本: {len(df_det)} 段龙头趋势')
print(f'T-3 → T 时间梯度信号: {len(df_grad)} 条记录')

# 统计能被至少 1 个恶化信号捕捉的失效比例
if warn_cols:
    any_warn = (df_det[warn_cols].sum(axis=1) > 0).sum()
    print(f'至少 1 个恶化信号覆盖: {any_warn}/{len(df_det)} ({any_warn/len(df_det)*100:.0f}%)')

# 各信号在 T-3 vs T 的变化
print('\n信号在 T-3 → T 的中位数变化:')
print(f'{"信号":<18s} {"T-3 中位":>10s} {"T 中位":>10s} {"变化":>8s} {"恶化方向":>8s}')
print('-' * 58)
for sig_name in SIGNAL_DIRECTION:
    t3_col = f'{sig_name}_Tminus3'
    t_col = f'{sig_name}_T'
    if t3_col not in df_det.columns:
        continue
    t3_med = df_det[t3_col].dropna().median()
    t_med = df_det[t_col].dropna().median()
    chg = t_med - t3_med
    direction = SIGNAL_DIRECTION[sig_name]
    worse = 'Y' if (direction * chg > 0) else 'N'
    print(f'{sig_name:<18s} {t3_med:10.3f} {t_med:10.3f} {chg:+8.3f} {worse:>8s}')

# 波动率比率特殊处理（> 1.2 为异常阈值）
if 'vol_ratio_T' in df_det.columns:
    vr_t3 = df_det['vol_ratio_Tminus3'].dropna()
    vr_t = df_det['vol_ratio_T'].dropna()
    print(f'\n波动率比率 >1.2: T-3: {(vr_t3>1.2).mean()*100:.0f}% → T: {(vr_t>1.2).mean()*100:.0f}%')

# 成交量信号总结
for sig in ['amt_ratio', 'amt_cv']:
    t3_col = f'{sig}_Tminus3'
    t_col = f'{sig}_T'
    if t3_col in df_det.columns:
        valid = df_det[[t3_col, t_col]].dropna()
        if len(valid) > 0:
            print(f'\n{sig}:')
            print(f'  T-3: median={valid[t3_col].median():.3f}, mean={valid[t3_col].mean():.3f}')
            print(f'  T:   median={valid[t_col].median():.3f}, mean={valid[t_col].mean():.3f}')
            # 缩量信号：T 的 amt_ratio < 0.8（成交额比 60 日均缩 20%+）
            if sig == 'amt_ratio':
                shrink = (valid[t_col] < 0.8).mean() * 100
                print(f'  T 时缩量(ratio<0.8): {shrink:.0f}%')
            if sig == 'amt_cv':
                high_cv = (valid[t_col] > valid[t_col].median()).mean() * 100
                print(f'  T 时高离散(CV>中位): {high_cv:.0f}%')

print('\nDone.')
