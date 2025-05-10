import numpy as np
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def compute_v_density(v_0: np.ndarray, n_neighbors: int = 20, r: float = 0.5):
    """
    对单个 V (shape: [3639, 64]) 进行标准化、降维，并用 FAISS 计算局部密度
    """
    scaler = StandardScaler()
    v_std = scaler.fit_transform(v_0)

    pca = PCA(n_components=0.8)
    v_pca = pca.fit_transform(v_std).astype('float32')

    index = faiss.IndexFlatL2(v_pca.shape[1])
    index.add(v_pca)
    distances, _ = index.search(v_pca, n_neighbors)
    density = np.sum(distances[:, 1:] < r, axis=1)  # 局部密度估计
    return density

def compute_attention_score(attn: np.ndarray, gqa_group_id: int, num_heads: int = 8):
    """
    从 attention tensor (24, 3639, 3639) 中取出对应 GQA group 的 heads，
    并对每个 token 聚合出 mean attention sum
    """
    head_ids = [gqa_group_id + i * num_heads for i in range(attn.shape[0] // num_heads)]  # e.g., [0, 8, 16]
    print(head_ids)
    selected_heads = attn[head_ids]  # shape: (3, 3639, 3639)

    attention_sum = selected_heads.sum(axis=0).sum(axis=0)  # mean over heads, sum over query dim
    return attention_sum  # shape: (3639,)

def plot_attention_vs_density(attn_scores: np.ndarray, v_density: np.ndarray, cutoff: int):
    """
    可视化 attention score 与 v density 的关系，附带线性拟合线与 Pearson 相关系数
    """
    # reshape for sklearn
    attn_scores = attn_scores[cutoff:]
    v_density = v_density[cutoff:]
    X = attn_scores.reshape(-1, 1)
    y = v_density

    # 拟合线性模型
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)

    # Pearson 相关
    corr, _ = pearsonr(attn_scores, v_density)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.scatter(attn_scores, v_density, alpha=0.5, s=20, label='Token')
    plt.plot(attn_scores, y_pred, color='red', linewidth=2, label=f'Fit line (R²={r2:.3f})')
    plt.xlabel("Mean Attention Score (token)")
    plt.ylabel("V Density (local)")
    plt.title(f"Attention Score vs V Density (Pearson r = {corr:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_token_attention_colored_by_density(attn_scores: np.ndarray, v_density: np.ndarray, cutoff: int):
    """
    x轴：token index
    y轴：attention sum
    color：density of v
    """
    attn_scores = attn_scores[cutoff:]
    v_density = v_density[cutoff:]
    indices = np.arange(len(attn_scores))

    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(indices, attn_scores, c=v_density, cmap='viridis', s=20)
    plt.xlabel("Token Index")
    plt.ylabel("Attention Score (mean over heads)")
    plt.title("Token Attention Colored by V Density")
    plt.colorbar(scatter, label="V Local Density")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    attn = np.load("qkvs/a_0.npy").squeeze()  # shape: (24, 3639, 3639)
    value = np.load("qkvs/v_0.npy").squeeze()  # shape: (8, 3639, 64)
    print(f"attn shape: {attn.shape}")
    print(f"value shape: {value.shape}")

    v_0 = value[0]  # (3639, 64)
    v_density = compute_v_density(v_0)  # local density of v[0]

    attn_score = compute_attention_score(attn, gqa_group_id=0)  # aggregated attention score

    plot_attention_vs_density(attn_score, v_density, cutoff = 128)

    plot_token_attention_colored_by_density(attn_score, v_density, cutoff = 128)

if __name__ == '__main__':
    main()

