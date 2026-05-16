import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings

# 尝试引入 IPython 的 display 以支持 DataFrame 渲染显示
try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings('ignore')

# ==========================================
# 核心指标计算模块 (保持原基础逻辑不变)
# ==========================================

def _normalize_time_index(df):
    """Try to coerce common date-like indices to a normalized DatetimeIndex."""
    df_copy = df.copy()

    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = df_copy.index.normalize()
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_localize(None)
        return df_copy

    idx_series = pd.Index(df_copy.index).astype(str).str.strip()
    idx_series = idx_series.str.replace(r'\.0$', '', regex=True)
    idx_series = idx_series.str.replace(r'[-/:\s]', '', regex=True).str[:8]
    parsed = pd.to_datetime(idx_series, format='%Y%m%d', errors='coerce')

    if parsed.notna().sum() == 0:
        return df_copy

    valid_mask = parsed.notna()
    df_copy = df_copy.loc[valid_mask].copy()
    df_copy.index = pd.DatetimeIndex(parsed[valid_mask]).normalize()
    return df_copy


def _align_data(pred_df, true_df):
    """时间轴对齐，剔除两端多余的 NaN"""
    pred_df = _normalize_time_index(pred_df)
    true_df = _normalize_time_index(true_df)
    pred_df, true_df = pred_df.align(true_df, join='inner')
    return pred_df, true_df


def _resolve_horizons(trajectory_df, fallback_horizons=None):
    """Resolve the actual horizon mapping stored on a trajectory matrix."""
    if hasattr(trajectory_df, 'attrs') and trajectory_df.attrs.get('horizons'):
        return list(trajectory_df.attrs['horizons'])
    if fallback_horizons is not None:
        return list(fallback_horizons)
    return [1, 2, 3, 4, 5, 6, 7]

def calculate_ic_series(pred_df, true_df, method='spearman'):
    """计算截面 Rank IC 或 Normal IC 序列。"""
    pred_df, true_df = _align_data(pred_df, true_df)
    ic_series = pd.Series(index=pred_df.index, dtype=float)
    
    for date in pred_df.index:
        p = pred_df.loc[date].dropna()
        t = true_df.loc[date].dropna()
        
        idx = p.index.intersection(t.index)
        if len(idx) < 2:
            continue
            
        p_val = p[idx].values
        t_val = t[idx].values
        
        if method == 'spearman':
            r, _ = spearmanr(p_val, t_val)
            ic_series.loc[date] = r
        else:
            ic_series.loc[date] = np.corrcoef(p_val, t_val)[0, 1]
            
    return ic_series.dropna()

def calculate_statistical_metrics(pred_df, true_df):
    """计算整体统计学指标：MAE, RMSE, Hit Rate, OOS R-squared。"""
    pred_df, true_df = _align_data(pred_df, true_df)
    
    p_flat = pred_df.values.flatten()
    t_flat = true_df.values.flatten()
    
    valid_mask = ~(np.isnan(p_flat) | np.isnan(t_flat))
    p_valid = p_flat[valid_mask]
    t_valid = t_flat[valid_mask]
    
    if len(p_valid) == 0:
        return {}
    
    mae = np.mean(np.abs(p_valid - t_valid))
    rmse = np.sqrt(np.mean((p_valid - t_valid)**2))
    hit_rate = np.mean(np.sign(p_valid) == np.sign(t_valid))
    
    tss = np.sum((t_valid - np.mean(t_valid))**2)
    rss = np.sum((t_valid - p_valid)**2)
    oos_r2 = 1 - (rss / tss) if tss != 0 else np.nan
    
    return {'MAE': mae, 'RMSE': rmse, 'Hit Rate': hit_rate, 'OOS R2': oos_r2}

def calculate_quantile_transition(pred_df, true_df, q=10):
    """计算预测分位数与真实分位数转移矩阵 (Quantile Confusion Matrix)。"""
    pred_df, true_df = _align_data(pred_df, true_df)
    
    p_stacked = pred_df.stack().reset_index()
    t_stacked = true_df.stack().reset_index()
    
    p_stacked.columns = ['Date', 'Asset', 'Pred']
    t_stacked.columns = ['Date', 'Asset', 'True']
    
    df_merged = pd.merge(p_stacked, t_stacked, on=['Date', 'Asset']).dropna()
    
    def assign_quantile(series):
        if len(series) < q:
            return pd.Series(np.nan, index=series.index)
        ranks = series.rank(pct=True, method='first')
        return pd.cut(ranks, bins=q, labels=range(1, q + 1))
    
    df_merged['Pred_Q'] = df_merged.groupby('Date')['Pred'].transform(assign_quantile)
    df_merged['True_Q'] = df_merged.groupby('Date')['True'].transform(assign_quantile)
    
    df_merged = df_merged.dropna()
    transition_matrix = pd.crosstab(df_merged['Pred_Q'], df_merged['True_Q'], normalize='all')
    return transition_matrix

def generate_evaluation_report(pred_df, true_df, q=10):
    """单矩阵评估接口。"""
    ic_series = calculate_ic_series(pred_df, true_df, method='spearman')
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic != 0 else np.nan
    
    stat_metrics = calculate_statistical_metrics(pred_df, true_df)
    transition_matrix = calculate_quantile_transition(pred_df, true_df, q=q)
    
    report = {
        'Rank IC (Mean)': mean_ic,
        'ICIR (Annualized)': icir,
        'Win Rate (IC > 0)': (ic_series > 0).mean() if len(ic_series) > 0 else np.nan,
        'MAE': stat_metrics.get('MAE', np.nan),
        'RMSE': stat_metrics.get('RMSE', np.nan),
        'Directional Hit Rate': stat_metrics.get('Hit Rate', np.nan),
        'OOS R2': stat_metrics.get('OOS R2', np.nan)
    }
    return pd.DataFrame([report]).T, transition_matrix, ic_series

# ==========================================
# 多期预测评估与可视化模块 (新增)
# ==========================================

def extract_horizon_matrix(trajectory_df, h, horizons=None):
    """
    将带列表的 3D 矩阵，转化为只包含第 h 期的纯数值矩阵
    """
    horizons = _resolve_horizons(trajectory_df, horizons)
    if h not in horizons:
        raise ValueError(f"给定的 h={h} 不在模型的 horizons 属性 {horizons} 中。")

    idx = horizons.index(h)

    def get_h_value(val):
        if isinstance(val, list) and len(val) > idx:
            return val[idx]
        return np.nan

    return trajectory_df.applymap(get_h_value).astype(float)


def evaluate_multi_horizon(trajectory_matrix, true_df, horizons=[1, 2, 3, 4, 5, 6, 7], q=10):
    """
    一键评估多期预测矩阵。
    自动将 t 时刻对 t+h 期的预测与 t+h 期的真实收益对齐，并输出横向对比表。
    """
    reports = {}
    
    for h in horizons:
        pred_h = extract_horizon_matrix(trajectory_matrix, h, _resolve_horizons(trajectory_matrix, horizons))
        true_h_shifted = true_df.shift(-h)
        report_df, _, _ = generate_evaluation_report(pred_h, true_h_shifted, q=q)
        reports[f't+{h}'] = report_df.iloc[:, 0]
        
    combined_report = pd.DataFrame(reports)
    return combined_report

def plot_prediction_vs_true(trajectory_matrix, true_df, asset, h, horizons=[1, 2, 3, 4, 5, 6, 7], figsize=(12, 5)):
    """
    绘制指定品种、指定期数的预测曲线与未来真实收益率曲线的对比图。
    """
    pred_h = extract_horizon_matrix(trajectory_matrix, h, _resolve_horizons(trajectory_matrix, horizons))
    
    if asset not in pred_h.columns or asset not in true_df.columns:
        print(f"找不到品种 '{asset}'，请检查资产名称是否正确。")
        return
        
    p_series = pred_h[asset]
    t_series = true_df[asset].shift(-h)
    
    df_plot = pd.DataFrame({
        f'Predicted (t+{h})': p_series, 
        f'True Future (t+{h})': t_series
    }).dropna()
    
    if len(df_plot) == 0:
        print(f"品种 '{asset}' 在给定的期数 h={h} 下没有足够的不为空的对齐数据以供绘图。")
        return
        
    plt.figure(figsize=figsize)
    plt.plot(df_plot.index, df_plot[f'True Future (t+{h})'], 
             label='True Return', alpha=0.5, color='gray', linestyle='--')
    plt.plot(df_plot.index, df_plot[f'Predicted (t+{h})'], 
             label='Predicted Return', alpha=0.9, color='blue')
             
    plt.axhline(0, color='red', alpha=0.3, linestyle='-')
             
    plt.title(f'[{asset}] Horizon t+{h}: Predicted vs True Return')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    
# 确保在文件开头 import 你的 my_backtest 包
import my_backtest

def evaluate_portfolio_backtest(models_dict, true_df, h, horizons=[1, 2, 3, 4, 5, 6, 7], 
                                top_pct_extreme=0.1, top_pct_moderate=0.3):
    """
    多模型回测大比拼，并将最终生成的表格自动智能标绿显示！
    """
    factors_2d = {}
    lags_dict = {}
    
    # ---------------------------------------------------------
    # 1. 提取出对应 h 期的纯数字 2D 矩阵
    # ---------------------------------------------------------
    if isinstance(h, int):
        for name, mat in models_dict.items():
            model_horizons = _resolve_horizons(mat, horizons)
            if h not in model_horizons:
                continue
            factors_2d[name] = extract_horizon_matrix(mat, h, model_horizons)
            lags_dict[name] = h
            
    elif isinstance(h, list):
        for name, mat in models_dict.items():
            model_horizons = _resolve_horizons(mat, horizons)
            for lag in h:
                if lag not in model_horizons:
                    continue
                label = f"{name} (t+{lag})"
                factors_2d[label] = extract_horizon_matrix(mat, lag, model_horizons)
                lags_dict[label] = lag
                
    elif isinstance(h, dict):
        for name, lag in h.items():
            if name in models_dict:
                model_horizons = _resolve_horizons(models_dict[name], horizons)
                if lag not in model_horizons:
                    continue
                label = f"{name} (t+{lag})"
                factors_2d[label] = extract_horizon_matrix(models_dict[name], lag, model_horizons)
                lags_dict[label] = lag
    else:
        raise ValueError("传入的 h 格式不被支持！")
        
    # ---------------------------------------------------------
    # 2. 调用 my_backtest 计算多空收益
    # ---------------------------------------------------------
    stats_list = []
    port_cum_dict = {}
    ic_cum_dict = {}
    dd_cum_dict = {}
    
    for label, factor_df in factors_2d.items():
        lag = lags_dict[label]
        
        port_series, ic_series = my_backtest.backtest_long_short(
            factor_df, true_df, lag=lag, 
            top_pct_extreme=top_pct_extreme, 
            top_pct_moderate=top_pct_moderate
        )
        
        s = port_series.dropna()
        if len(s) == 0:
            print(f"⚠️ 警告：'{label}' 未生成有效回测数据。")
            continue
            
        cum_series = (1 + s).cumprod()
        total_ret = cum_series.iloc[-1] if len(cum_series) > 0 else np.nan
        
        total_for_ann = (1 + s).prod() - 1
        years = len(s) / 252
        ann_ret = (1 + total_for_ann) ** (1 / years) - 1 if years > 0 else 0
        
        mean_ann = s.mean() * 252
        std_ann = s.std(ddof=1) * np.sqrt(252)
        sharpe = mean_ann / std_ann if std_ann != 0 else np.nan
        
        peak = cum_series.cummax()
        drawdowns = (peak - cum_series) / peak
        max_dd = drawdowns.max()
        
        calmar = ann_ret / max_dd if (max_dd != 0 and not np.isnan(max_dd)) else np.nan
        win_rate = (s > 0).mean()
        
        stats_list.append({
            'Model_Horizon': label,
            'Win Rate': win_rate,
            'Cum Return': total_ret,
            'Ann Return': ann_ret,
            'Sharpe': sharpe,
            'Calmar': calmar,
            'Max Drawdown': max_dd,
            'IC Mean': ic_series.mean()
        })
        
        port_cum_dict[label] = cum_series
        ic_cum_dict[label] = ic_series.dropna().cumsum()
        dd_cum_dict[label] = (cum_series - peak) / peak
        
    if not stats_list:
        return pd.DataFrame()
        
    stats_df = pd.DataFrame(stats_list).set_index('Model_Horizon')
    
    # ---------------------------------------------------------
    # 3. 结果作图
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    for name, cum_ret in port_cum_dict.items():
        plt.plot(cum_ret.index, cum_ret.values, label=name, linewidth=1.5)
    plt.title('Cumulative Returns (Prediction Portfolios)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for name, cum_ic in ic_cum_dict.items():
        plt.plot(cum_ic.index, cum_ic.values, label=name, linewidth=1.5)
    plt.title('Cumulative Rank IC')
    plt.axhline(0, color='k', lw=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for name, dd in dd_cum_dict.items():
        line, = plt.plot(dd.index, dd.values, label=name, linewidth=1)
        plt.fill_between(dd.index, dd.values, 0, color=line.get_color(), alpha=0.3)
    plt.title('Drawdown Series')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------------------------
    # 4. 🚀 核心优化：智能内嵌输出并高亮极值表格
    # ---------------------------------------------------------
    sorted_stats = stats_df.sort_values('Sharpe', ascending=False)
    
    try:
        max_cols = [col for col in sorted_stats.columns if col != 'Max Drawdown']
        min_cols = ['Max Drawdown'] if 'Max Drawdown' in sorted_stats.columns else []
        
        styled = sorted_stats.style.format("{:.4f}")
        if max_cols:
            styled = styled.highlight_max(subset=max_cols, color='lightgreen')
        if min_cols:
            # 针对最大回撤，去寻找最小值标绿
            styled = styled.highlight_min(subset=min_cols, color='lightgreen')
            
        print("\n=== 📊 核心指标回测结果对比 ===")
        display(styled)
    except Exception:
        # 如果当前环境不支持渲染 (比如直接跑 py 脚本)，则退化为普通打印
        print("\n=== 📊 核心指标回测结果对比 ===")
        print(sorted_stats)

    return sorted_stats


def plot_cumulative_ic_comparison(models_dict, true_df, h=1, horizons=[1, 2, 3, 4, 5, 6, 7], 
                                  top_pct_extreme=0.1, top_pct_moderate=0.3):
    """
    独立对比所有模型的累计 IC 图表
    """
    factors_2d = {}
    lags_dict = {}
    
    if isinstance(h, int):
        for name, mat in models_dict.items():
            model_horizons = _resolve_horizons(mat, horizons)
            if h not in model_horizons:
                continue
            factors_2d[name] = extract_horizon_matrix(mat, h, model_horizons)
            lags_dict[name] = h
    elif isinstance(h, list):
        for name, mat in models_dict.items():
            model_horizons = _resolve_horizons(mat, horizons)
            for lag in h:
                if lag not in model_horizons:
                    continue
                label = f"{name} (t+{lag})"
                factors_2d[label] = extract_horizon_matrix(mat, lag, model_horizons)
                lags_dict[label] = lag
    else:
        raise ValueError("传入的 h 格式不被支持！")
        
    ic_cum_dict = {}
    stats_list = []  
    
    for label, factor_df in factors_2d.items():
        lag = lags_dict[label]
        
        _, ic_series = my_backtest.backtest_long_short(
            factor_df, true_df, lag=lag, 
            top_pct_extreme=top_pct_extreme, 
            top_pct_moderate=top_pct_moderate
        )
        
        valid_ic_series = ic_series.dropna()
        if len(valid_ic_series) == 0:
            continue
            
        ic_cum_dict[label] = valid_ic_series.cumsum()
        
        mean_ic = valid_ic_series.mean()
        std_ic = valid_ic_series.std()
        icir = (mean_ic / std_ic * np.sqrt(252)) if std_ic != 0 else np.nan
        
        stats_list.append({
            'Model_Horizon': label,
            'Mean IC': mean_ic,
            'ICIR': icir,
            'IC Win Rate': (valid_ic_series > 0).mean(),
            'Num Observations': len(valid_ic_series)
        })

    plt.figure(figsize=(12, 6))
    for name, cum_ic in ic_cum_dict.items():
        plt.plot(cum_ic.index, cum_ic.values, label=name, linewidth=1.5)
    
    plt.title('Cumulative Rank IC Comparison')
    plt.axhline(0, color='k', lw=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if not stats_list:
        return pd.DataFrame()
        
    stats_df = pd.DataFrame(stats_list).set_index('Model_Horizon')
    sorted_stats = stats_df.sort_values('ICIR', ascending=False)
    
    # 智能标绿并输出
    try:
        styled = sorted_stats.style.format("{:.4f}").highlight_max(color='lightgreen')
        print("\n=== 📊 核心 IC 指标对比 ===")
        display(styled)
    except Exception:
        print("\n=== 📊 核心 IC 指标对比 ===")
        print(sorted_stats)
        
    return sorted_stats


def evaluate_split_periods(models_dict, true_df, h=1, n_splits=5, horizons=[1, 2, 3, 4, 5, 6, 7], q=10):
    """
    ========================================================
    多模型 & 分段时序评估 (Multi-Model & Time-Split Evaluation)
    ========================================================
    """
    common_dates = None
    for model, traj in models_dict.items():
        if common_dates is None:
            common_dates = traj.index
        else:
            common_dates = common_dates.intersection(traj.index)
            
    all_dates = pd.to_datetime(common_dates.unique().sort_values())
    date_chunks = np.array_split(all_dates, n_splits)
    true_h_shifted = true_df.shift(-h)
    
    all_reports = []
    
    for i, dates in enumerate(date_chunks):
        if len(dates) == 0: continue
        start_dt = dates[0]
        end_dt = dates[-1]
        
        start_str = start_dt.strftime('%Y-%m-%d')
        end_str = end_dt.strftime('%Y-%m-%d')
        period_name = f"Split {i+1} ({start_str} ~ {end_str})"
        
        true_chunk = true_h_shifted.loc[start_dt:end_dt]
        chunk_report = {}
        
        for model_name, trajectory_matrix in models_dict.items():
            pred_h = extract_horizon_matrix(trajectory_matrix, h, _resolve_horizons(trajectory_matrix, horizons))
            pred_chunk = pred_h.loc[start_dt:end_dt]
            report_df, _, _ = generate_evaluation_report(pred_chunk, true_chunk, q=q)
            chunk_report[model_name] = report_df.iloc[:, 0]
            
        chunk_df = pd.DataFrame(chunk_report)
        chunk_df.index = pd.MultiIndex.from_product([[period_name], chunk_df.index], names=['Time Segment', 'Metric'])
        all_reports.append(chunk_df)
        
    final_report = pd.concat(all_reports)
    return final_report