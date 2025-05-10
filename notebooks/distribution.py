import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import faiss

def visualize_density_evolution(value_one: np.ndarray, step: int = 1000, n_neighbors: int = 20, r: float = 0.5):
    import matplotlib.pyplot as plt
    import faiss
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import numpy as np

    n, d = value_one.shape
    num_steps = (n + step - 1) // step  # ceil(n / step)

    scaler = StandardScaler()
    value_std = scaler.fit_transform(value_one)

    pca = PCA(n_components=0.8)
    value_pca = pca.fit_transform(value_std)
    print(f"After PCA, shape: {value_pca.shape}")

    # 全局建索引
    full_data = value_pca.astype('float32')
    index_full = faiss.IndexFlatL2(full_data.shape[1])
    index_full.add(full_data)
    distances_full, _ = index_full.search(full_data, n_neighbors)
    global_density = np.sum(distances_full[:, 1:] < r, axis=1)
    full_2d = full_data[:, :2]

    for i in range(num_steps):
        start_idx = i * step
        end_idx = min((i + 1) * step, n)
        block = full_data[start_idx:end_idx]

        index = faiss.IndexFlatL2(block.shape[1])
        index.add(block)
        distances, _ = index.search(block, n_neighbors)
        local_density = np.sum(distances[:, 1:] < r, axis=1)
        block_2d = block[:, :2]

        # 两列子图：左 current, 右 global
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左：当前 step 的密度图
        sc1 = axes[0].scatter(block_2d[:, 0], block_2d[:, 1], c=local_density, cmap='viridis', s=50)
        axes[0].set_title(f'Current Step {i + 1} ({start_idx}-{end_idx})')
        axes[0].set_xlabel('PCA 1')
        axes[0].set_ylabel('PCA 2')
        fig.colorbar(sc1, ax=axes[0], label='Estimated Density')

        # 右：全局密度图（仅参考）
        sc2 = axes[1].scatter(full_2d[:, 0], full_2d[:, 1], c=global_density, cmap='viridis', s=10)
        axes[1].set_title('Global Density (All 3664)')
        axes[1].set_xlabel('PCA 1')
        axes[1].set_ylabel('PCA 2')
        fig.colorbar(sc2, ax=axes[1], label='Estimated Density')

        plt.suptitle(f'Density Comparison - Step {i + 1}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=True)

def main():
    attns = np.load("qkvs/a_0.npy").squeeze()
    print(attns.shape)
    values = np.load("qkvs/v_0.npy").squeeze()
    print(values.shape)
    visualize_density_evolution(values[0])

if __name__ == '__main__':
    main()