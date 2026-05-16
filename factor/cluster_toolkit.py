import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 辅助函数：构建标准答案映射
# ==========================================
def _get_true_labels(sector_map, valid_tickers):
    """根据 sector_map 生成自然数标准答案字典"""
    sector_to_id = {sector: i+1 for i, sector in enumerate(sector_map.keys())}
    ticker_to_id = {}
    for sector, tickers in sector_map.items():
        for ticker in tickers:
            ticker_to_id[ticker] = sector_to_id[sector]
    return [ticker_to_id.get(t, np.nan) for t in valid_tickers]

# ==========================================
# 聚类方法 1：固定标准分类
# ==========================================
def method_fixed(returns, sector_map):
    """基于固定设定的品种进行分类，仅在有数据的时间点填充"""
    result = pd.DataFrame(index=returns.index, columns=returns.columns)
    sector_to_id = {sector: i+1 for i, sector in enumerate(sector_map.keys())}
    ticker_map = {t: sector_to_id[s] for s, tkrs in sector_map.items() for t in tkrs}
    
    for col in returns.columns:
        if col in ticker_map:
            mask = returns[col].notna()
            result.loc[mask, col] = ticker_map[col]
            
    return result.astype(float)

# ==========================================
# 聚类方法 2：滚动层次聚类 (轮廓系数寻优)
# ==========================================
def method_hierarchical(returns, lookback, k_range):
    """用过去 lookback 期(包含t)数据进行层次聚类，作为 t+1 期的结果"""
    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    min_k, max_k = k_range
    
    for i in range(lookback - 1, len(returns) - 1):
        window = returns.iloc[i - lookback + 1 : i + 1]
        valid_cols = window.columns[window.notna().all()]
        actual_max_k = min(max_k, len(valid_cols) - 1)
        
        if len(valid_cols) < max(min_k, 2):
            continue 
            
        corr = window[valid_cols].corr().fillna(0)
        dist_matrix = np.sqrt(2 * (1 - corr))
        np.fill_diagonal(dist_matrix.values, 0)
        
        dist_links = squareform(dist_matrix)
        Z = linkage(dist_links, method='ward')
        
        best_k = min_k
        best_score = -1
        best_labels = None
        
        for k in range(min_k, actual_max_k + 1):
            labels = fcluster(Z, t=k, criterion='maxclust')
            if len(set(labels)) > 1:
                score = silhouette_score(dist_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    
        if best_labels is not None:
            result.loc[returns.index[i + 1], valid_cols] = best_labels
            
    return result

# ==========================================
# 聚类方法 3：滚动谱聚类 (特征间隙寻优)
# ==========================================
def method_spectral(returns, lookback, k_range):
    """用过去 lookback 期(包含t)数据进行谱聚类，作为 t+1 期的结果"""
    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    min_k, max_k = k_range
    
    for i in range(lookback - 1, len(returns) - 1):
        window = returns.iloc[i - lookback + 1 : i + 1]
        valid_cols = window.columns[window.notna().all()]
        actual_max_k = min(max_k, len(valid_cols) - 1)
        
        if len(valid_cols) < max(min_k, 2):
            continue
            
        corr = window[valid_cols].corr().fillna(0)
        W = (corr + 1) / 2 
        np.fill_diagonal(W.values, 0)
        
        L_norm = laplacian(W.values, normed=True)
        eigenvalues, _ = np.linalg.eigh(L_norm)
        eigenvalues = np.sort(eigenvalues)
        
        best_k = min_k
        max_gap = -1
        
        for k in range(min_k, actual_max_k + 1):
            if k < len(eigenvalues):
                gap = eigenvalues[k] - eigenvalues[k-1]
                if gap > max_gap:
                    max_gap = gap
                    best_k = k
        
        spectral = SpectralClustering(n_clusters=best_k, affinity='precomputed', 
                                      assign_labels='kmeans', random_state=42)
        best_labels = spectral.fit_predict(W.values) + 1 
        result.loc[returns.index[i + 1], valid_cols] = best_labels
            
    return result

# ==========================================
# 评估模块 1：输出数值指标表格
# ==========================================
def evaluate_metrics(cluster_result, returns, sector_map, lookback=120):
    """输出内部与外部评价指标"""
    eval_records = []
    
    for i in range(lookback, len(cluster_result)):
        date = cluster_result.index[i]
        labels_series = cluster_result.iloc[i].dropna()
        
        if len(labels_series) < 3 or len(set(labels_series)) < 2:
            continue
            
        valid_tickers = labels_series.index
        current_labels = labels_series.values
        
        window = returns.iloc[i - lookback : i][valid_tickers]
        corr = window.corr().fillna(0)
        dist_matrix = np.sqrt(2 * (1 - corr))
        np.fill_diagonal(dist_matrix.values, 0)
        
        X_features = corr.values 
        
        sil_score = silhouette_score(dist_matrix, current_labels, metric='precomputed')
        ch_score = calinski_harabasz_score(X_features, current_labels)
        db_score = davies_bouldin_score(X_features, current_labels)
        
        true_labels_full = _get_true_labels(sector_map, valid_tickers)
        valid_mask = ~np.isnan(true_labels_full)
        
        ari_score, nmi_score = np.nan, np.nan
        if valid_mask.sum() >= 2:
            true_l = np.array(true_labels_full)[valid_mask]
            pred_l = current_labels[valid_mask]
            ari_score = adjusted_rand_score(true_l, pred_l)
            nmi_score = normalized_mutual_info_score(true_l, pred_l)
            
        eval_records.append({
            'Date': date, 'Silhouette': sil_score, 
            'CH': ch_score, 'DB': db_score,
            'ARI': ari_score, 'NMI': nmi_score
        })
        
    return pd.DataFrame(eval_records).set_index('Date')

# ==========================================
# 评估模块 2：可视化输出 (单张独立输出)
# ==========================================
def plot_evaluation(cluster_result, returns, target_date, lookback=120):
    """依次独立输出聚类热力图、轮廓图、降维散点图"""
    if target_date not in cluster_result.index:
        print(f"找不到日期 {target_date} 的聚类结果。")
        return
        
    labels_series = cluster_result.loc[target_date].dropna()
    if len(labels_series) < 3:
        print("有效品种不足，无法绘图。")
        return
        
    valid_tickers = labels_series.index
    labels = labels_series.values.astype(int)
    
    idx_pos = returns.index.get_loc(target_date)
    window = returns.iloc[idx_pos - lookback : idx_pos][valid_tickers]
    corr = window.corr().fillna(0)
    
    # -----------------------------------
    # 图1：聚类热力图
    # -----------------------------------
    plt.figure(figsize=(9, 9))
    sort_idx = np.argsort(labels)
    sorted_tickers = valid_tickers[sort_idx]
    sorted_corr = corr.loc[sorted_tickers, sorted_tickers]
    sns.heatmap(sorted_corr, cmap='vlag', center=0, 
                xticklabels=sorted_tickers, yticklabels=sorted_tickers, cbar_kws={"shrink": .8})
    plt.title(f"Clustered Correlation Heatmap ({target_date})")
    plt.tight_layout()
    plt.show()
    
    # -----------------------------------
    # 图2：轮廓图
    # -----------------------------------
    plt.figure(figsize=(8, 5))
    ax2 = plt.gca()
    dist_matrix = np.sqrt(2 * (1 - corr))
    np.fill_diagonal(dist_matrix.values, 0)
    
    from sklearn.metrics import silhouette_samples
    sample_values = silhouette_samples(dist_matrix, labels, metric='precomputed')
    
    y_lower = 10
    n_clusters = len(set(labels))
    for i, k in enumerate(sorted(list(set(labels)))):
        ith_cluster_vals = sample_values[labels == k]
        ith_cluster_vals.sort()
        size_cluster_i = ith_cluster_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {k}")
        y_lower = y_upper + 10
        
    avg_score = silhouette_score(dist_matrix, labels, metric='precomputed')
    ax2.axvline(x=avg_score, color="red", linestyle="--")
    plt.title(f"Silhouette Plot ({target_date} | Avg Score: {avg_score:.3f})")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    plt.tight_layout()
    plt.show()
    
    # -----------------------------------
    # 图3：PCA 散点图
    # -----------------------------------
    plt.figure(figsize=(9, 9))
    pca = PCA(n_components=2)
    xy = pca.fit_transform(corr.values)
    
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap='nipy_spectral', alpha=0.8, s=100)
    for i, txt in enumerate(valid_tickers):
        plt.annotate(txt, (xy[i, 0] + 0.02, xy[i, 1] + 0.02), fontsize=10)
            
    plt.title(f"PCA Dimensionality Reduction ({target_date})")
    plt.colorbar(scatter, label="Cluster Label")
    plt.tight_layout()
    plt.show()

# ==========================================
# 评估模块 3：随时间变化的 K 值对比折线图
# ==========================================
def plot_k_over_time(cluster_results_dict):
    """
    绘制不同聚类方法随时间变化的类簇数量 (K) 的折线图对比。
    """
    plt.figure(figsize=(15, 6))
    
    for name, df in cluster_results_dict.items():
        # 逐行统计每天有效的类别数量 (排除 NaN 的品种)
        k_series = df.apply(lambda row: len(set(row.dropna())), axis=1)
        
        # 只绘制 K 大于等于 1 的有效天数
        valid_k = k_series[k_series >= 1]
        
        # === 核心修改：时间格式处理 ===
        # 尝试将类似 201801030000 的数字转换为正常的 Datetime 对象
        try:
            # 检查是否为 12 位的时间戳格式 (YYYYMMDDHHMM)
            first_val = str(valid_k.index[0]).strip()
            if len(first_val) == 12 and first_val.isdigit():
                plot_x = pd.to_datetime(valid_k.index.astype(str), format='%Y%m%d%H%M')
            else:
                # 尝试通用转换
                plot_x = pd.to_datetime(valid_k.index)
        except Exception as e:
            # 如果转换失败，就使用原始索引
            plot_x = valid_k.index 
            
        plt.plot(plot_x, valid_k.values, label=name, linewidth=2, alpha=0.8)
        
    plt.title('Optimal Number of Clusters (K) Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Clusters (K)', fontsize=12)
    plt.legend(fontsize=11)
    
    # 优化 X 轴显示，避免时间标签重叠
    plt.gcf().autofmt_xdate() 
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_cluster_mapping(cluster_pred_df, returns_df, sector_map, target_date):
    """
    通过交叉表查看动态聚类与固定聚类的真实映射关系
    """
    if target_date not in cluster_pred_df.index:
        print("日期不在聚类结果中")
        return
        
    # 1. 获取当天的动态预测标签
    pred_labels = cluster_pred_df.loc[target_date].dropna()
    valid_tickers = pred_labels.index
    
    # 2. 获取当天的固定真实标签 (沿用之前的函数)
    # 反转 sector_map 以便查询
    ticker_to_sector = {}
    for sector, tickers in sector_map.items():
        for t in tickers:
            ticker_to_sector[t] = sector
            
    # 构建对比 DataFrame
    compare_df = pd.DataFrame({
        'Predicted_Cluster': pred_labels.values.astype(int),
        'Fixed_Sector': [ticker_to_sector.get(t, 'Unknown') for t in valid_tickers]
    }, index=valid_tickers)
    
    # 过滤掉未知品种
    compare_df = compare_df[compare_df['Fixed_Sector'] != 'Unknown']
    
    # 3. 计算交叉表 (Cross-Tab)
    # 行是固定行业，列是模型预测出的类别
    cross_tab = pd.crosstab(compare_df['Fixed_Sector'], compare_df['Predicted_Cluster'])
    
    # 打印具体的包含明细
    print(f"\n=== {target_date} 聚类映射明细 ===")
    for cluster_id in cross_tab.columns:
        members = compare_df[compare_df['Predicted_Cluster'] == cluster_id].index.tolist()
        print(f"动态类别 {cluster_id} 包含的品种: {members}")
    
    # 4. 绘制热力图直观展示
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, annot=True, cmap="YlGnBu", fmt='g', cbar=False)
    plt.title(f"Cluster Mapping: Fixed Sector vs Dynamic Clusters ({target_date})")
    plt.xlabel("Dynamic Predicted Cluster (K)")
    plt.ylabel("Fixed Sector (Ground Truth)")
    plt.tight_layout()
    plt.show()
    
    return cross_tab
