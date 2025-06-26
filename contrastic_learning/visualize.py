import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

def visualize_single_tsne(data_path, output_path='tsne_single.png', title='t-SNE Visualization'):
    """
    对单组数据执行 t-SNE 降维并保存散点图。
    要求 CSV 文件中有一列名为 'label'，其余为特征。
    """
    # 加载数据
    df = pd.read_csv('./out/results/' + data_path)
    features = df.drop(columns=['label']).values
    labels = df['label'].values

    # 标准化特征
    features_scaled = StandardScaler().fit_transform(features)

    # 执行 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(features_scaled)

    # 整理为 DataFrame
    df_vis = pd.DataFrame(X_embedded, columns=['dim1', 'dim2'])
    df_vis['label'] = labels

    # 绘图
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df_vis,
        x='dim1', y='dim2',
        hue='label',
        palette='tab10',
        alpha=0.7
    )
    plt.title(title, fontsize=16)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    pathes = {
        'sentence_0_1_2_3': 'sentence_level_hiddens_layers_0-1-2-3.csv',
        'sentence_4_5_6_7': 'sentence_level_hiddens_layers_4-5-6-7.csv',
        'sentence_8_9_10_11': 'sentence_level_hiddens_layers_8-9-10-11.csv',
        'sentence_2_5_8_11': 'sentence_level_hiddens_layers_2-5-8-11.csv',
        'sentence_5_7_9_11': 'sentence_level_hiddens_layers_5-7-9-11.csv',
        'sentence_11': 'sentence_level_hiddens_layers_11.csv',
        'token_0_1_2_3': 'token_level_hiddens_layers_0-1-2-3.csv',
        'token_4_5_6_7': 'token_level_hiddens_layers_4-5-6-7.csv',
        'token_8_9_10_11': 'token_level_hiddens_layers_8-9-10-11.csv',
        'token_2_5_8_11': 'token_level_hiddens_layers_2-5-8-11.csv',
        'token_5_7_9_11': 'token_level_hiddens_layers_5-7-9-11.csv',
        'token_11': 'token_level_hiddens_layers_11.csv',
    }
    for title,path in pathes.items():
        if not os.path.exists('./out/results/' + path):
            print(f"not found: {path}")
            continue
        visualize_single_tsne(
        data_path=path,
        output_path='./figures/tsne_{}.png'.format(title),
        title='t-SNE of {}'.format(title)
        )
