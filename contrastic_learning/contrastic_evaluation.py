import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time

def evaluate_representation(features, labels, n_classes=None):
    """
    评估隐层表征质量的综合函数
    返回包含多个指标的字典
    """
    metrics = {}
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 1. 线性可分性评估（线性SVM交叉验证）
    svm = LinearSVC(max_iter=10000, random_state=42)
    svm_scores = cross_val_score(svm, scaled_features, labels, cv=5, n_jobs=-1)
    metrics['svm_accuracy'] = np.mean(svm_scores)
    
    # 2. 最近邻分类评估（1-NN准确率）
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn_scores = cross_val_score(knn, scaled_features, labels, cv=5, n_jobs=-1)
    metrics['1nn_accuracy'] = np.mean(knn_scores)
    
    # 3. 聚类质量评估（需要类别数量）
    if n_classes is None:
        n_classes = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # 调整兰德指数
    metrics['ari'] = adjusted_rand_score(labels, cluster_labels)
    
    # 标准化互信息
    metrics['nmi'] = normalized_mutual_info_score(labels, cluster_labels)
    
    # 4. 轮廓系数（无监督指标）
    if len(features) > 1000:  # 大数据集使用子采样
        sample_idx = np.random.choice(len(features), 1000, replace=False)
        sample_features = scaled_features[sample_idx]
        sample_labels = labels[sample_idx]
    else:
        sample_features = scaled_features
        sample_labels = labels
    
    metrics['silhouette'] = silhouette_score(
        sample_features, 
        sample_labels,
        metric='euclidean'
    )
    
    return metrics

# 模型路径配置
pathes = {
    'lora': './results/lora_hidden2.csv',
    'k30_alpha0.2': './results/token_level_hiddens_k30_alpha0.2.csv',
    'k30_alpha0.5': './results/token_level_hiddens_k30_alpha0.5.csv',
    'k60_alpha0.2': './results/token_level_hiddens_k60_alpha0.2.csv',
    'k60_alpha0.5': './results/token_level_hiddens_k60_alpha0.5.csv',
    'k90_alpha0.2': './results/token_level_hiddens_k90_alpha0.2.csv',
    'k90_alpha0.5': './results/token_level_hiddens_k90_alpha0.5.csv',
    '90_alpha0.25_nc5': './results/token_level_hiddens_k90_alpha0.25_nc5.csv',
    '90_alpha0.25_nc10': './results/token_level_hiddens_k90_alpha0.25_nc10.csv',
    '90_alpha0.25_nc15': './results/token_level_hiddens_k90_alpha0.25_nc15.csv',
}

# 评估所有模型
results = {}
for model_name, path in pathes.items():
    print(f"Evaluating {model_name}...")
    start_time = time.time()
    
    # 读取数据
    df = pd.read_csv(path)
    features = df.drop(columns=['label']).values
    labels = df['label'].values
    
    # 评估表征质量
    metrics = evaluate_representation(features, labels)
    results[model_name] = metrics
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds\n")

# 创建结果DataFrame并保存
results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
results_df.to_csv('representation_metrics.csv', float_format='%.4f')

# 打印结果
print("="*80)
print("Representation Quality Evaluation Results:")
print("="*80)
print(results_df)