# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# Settings
# ============================================================

REPLICATE = 1          # choose 1, 2, or 3
SAMPLE_FRACTION = 1
SAMPLE_SEED = 2

TIME_END = 1200
DMD_RANK = 200
DELAY_DIM = 200

N_CLUSTERS = 3

GROUP_NAMES = ["Ctrl", "MBCD", "20", "100", "250"]

RESULTS_DIR = Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def sample_cells(data, fraction=0.5, seed=42):
    rng = np.random.default_rng(seed)
    n_cells = data.shape[0]
    n_keep = int(n_cells * fraction)
    idx = rng.choice(n_cells, n_keep, replace=False)
    return data[idx]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.values.T   # cells x time


def DMD_self(data, r):
    X1 = data[:, :-1]
    X2 = data[:, 1:]

    u, s, vh = np.linalg.svd(X1, full_matrices=False)

    A_tilde = (
        u[:, :r].conj().T
        @ X2
        @ vh[:r, :].conj().T
        @ np.diag(1 / s[:r])
    )

    eigval, eigvec = np.linalg.eig(A_tilde)

    Psi = (
        X2
        @ vh[:r, :].conj().T
        @ np.diag(1 / s[:r])
        @ eigvec
    )

    return eigval, eigvec, Psi


def reconstruct_DMD_system(data, r):
    T = data.shape[1]

    eigval, eigvec, Psi = DMD_self(data, r)

    b = np.linalg.pinv(Psi) @ data[:, 0]

    time_dynamics = np.zeros((r, T), dtype=complex)

    for t in range(T):
        time_dynamics[:, t] = np.power(eigval, t) * b

    return (Psi @ time_dynamics).real, time_dynamics.real


def build_delay_embedding(X, delay_dim, end):
    Xaug = []

    for i in range(delay_dim):
        z = X[:, i:end - (delay_dim - i)]
        Xaug.append(z)

    return np.concatenate(Xaug).astype(np.float32)


def plot_3d_clustering(data, labels, n_clusters):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for cluster in range(n_clusters):
        cluster_points = data[labels == cluster]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            label=f"Cluster {cluster + 1}"
        )

    ax.set_title("3D Clustering Results (Kernel PCA + K-Means)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend()
    plt.show()


def summarize_clusters(labels, groups, group_names):
    assert len(labels) == sum(g.shape[0] for g in groups), (
        f"Mismatch: labels={len(labels)}, "
        f"cells in groups={sum(g.shape[0] for g in groups)}"
    )

    index_intervals = []
    start = 0

    for g in groups:
        end = start + g.shape[0]
        index_intervals.append((start, end))
        start = end

    cluster_ids = np.unique(labels)
    counts = np.zeros((len(cluster_ids), len(groups)), dtype=int)

    for j, (start, end) in enumerate(index_intervals):
        sub_labels = labels[start:end]

        for i, cl in enumerate(cluster_ids):
            counts[i, j] = np.sum(sub_labels == cl)

    cluster_names = [f"Cluster {cl + 1}" for cl in cluster_ids]

    count_df = pd.DataFrame(
        counts,
        index=cluster_names,
        columns=group_names
    )

    percent_df = count_df / count_df.sum(axis=0) * 100

    return count_df, percent_df.round(1)


def save_cluster_labels(labels, groups, group_names, rep_dir):
    condition_labels = []
    condition_cell_index = []

    for name, g in zip(group_names, groups):
        condition_labels.extend([name] * g.shape[0])
        condition_cell_index.extend(np.arange(g.shape[0]))

    cluster_df = pd.DataFrame({
        "global_cell_index": np.arange(len(labels)),
        "condition": condition_labels,
        "condition_cell_index": condition_cell_index,
        "cluster": labels
    })

    save_path = rep_dir / "cluster_labels.csv"
    cluster_df.to_csv(save_path, index=False)

    print(f"Saved cluster labels to: {save_path}")

    return cluster_df


# ============================================================
# Load all replicates
# ============================================================

replicate_groups = {
    1: [
        sample_cells(load_data("Data/Control_001_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/MBCD_001_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/20uM_001_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/100uM_004_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/250uM_001_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
    ],
    2: [
        sample_cells(load_data("Data/Control_002_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/MBCD_002_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/20uM_002_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/100uM_005_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
        sample_cells(load_data("Data/250uM_002_raw_signals_normalized.csv"), SAMPLE_FRACTION, SAMPLE_SEED),
    ],
    3: [
        load_data("Data/Control_003_raw_signals_normalized.csv"),
        load_data("Data/MBCD_003_raw_signals_normalized.csv"),
        load_data("Data/20uM_003_raw_signals_normalized.csv"),
        load_data("Data/100uM_003_raw_signals_normalized.csv"),
        load_data("Data/250uM_003_raw_signals_normalized.csv"),
    ],
}


# ============================================================
# Select dataset
# ============================================================

groups = replicate_groups[REPLICATE]
Dataset = np.vstack(groups)

rep_dir = RESULTS_DIR / f"replicate_{REPLICATE}"
rep_dir.mkdir(exist_ok=True)

print(f"Selected replicate: {REPLICATE}")
print("Dataset shape:", Dataset.shape)
print("Group sizes:", dict(zip(GROUP_NAMES, [g.shape[0] for g in groups])))


# ============================================================
# Prepare data
# ============================================================

X = Dataset[:, :TIME_END].astype(np.float32)

Xaug_f = build_delay_embedding(
    X,
    delay_dim=DELAY_DIM,
    end=TIME_END
)

print("Xaug_f:", Xaug_f.shape)

plt.imshow(Xaug_f[::20, :], aspect="auto", vmin=0, vmax=1)
plt.colorbar()
plt.xlabel("Time in sec")
plt.ylabel("Augmented cell number")
plt.title("Time-delay embedded data")
plt.show()


# ============================================================
# DMD
# ============================================================

eigval, eigvec, Psi = DMD_self(Xaug_f, DMD_RANK)

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect("equal", adjustable="box")

plt.plot(eigval.real, eigval.imag, "o", color="red", markersize=6)

circle = plt.Circle((0, 0), 1, color="gray", linewidth=2.5, fill=False)
ax.add_patch(circle)

plt.grid(axis="both", linestyle="--", linewidth=0.1, color="gray")
ax.tick_params(direction="in")
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.title("DMD eigenvalues")
plt.show()


# ============================================================
# DMD reconstruction and amplitude extraction
# ============================================================

rec_system, time_dynamics = reconstruct_DMD_system(Xaug_f, DMD_RANK)

n_cells = Dataset.shape[0]

psi_cut = Psi[:n_cells, :]
amplitude_f = np.abs(psi_cut)

print("amplitude_f:", amplitude_f.shape)

assert amplitude_f.shape[0] == Dataset.shape[0], (
    f"Mismatch: amplitude rows={amplitude_f.shape[0]}, "
    f"dataset cells={Dataset.shape[0]}"
)


# ============================================================
# Kernel PCA
# ============================================================

kernel_pca = KernelPCA(
    n_components=3,
    kernel="rbf",
    gamma=2.5
)

X_transformed = kernel_pca.fit_transform(amplitude_f)

print("X_transformed:", X_transformed.shape)

assert X_transformed.shape[0] == Dataset.shape[0]


# ============================================================
# Elbow and silhouette analysis
# ============================================================

inertia = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=30,
        n_init="auto"
    )

    temp_labels = kmeans.fit_predict(X_transformed)

    inertia.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(X_transformed, temp_labels)
    )


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, "-o", label="Inertia")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, "-o", label="Silhouette Score")
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# Final K-means clustering
# ============================================================

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=30,
    n_init="auto"
)

labels = kmeans.fit_predict(X_transformed)
labels = np.asarray(labels)

print("Total cells:", Dataset.shape[0])
print("Total labels:", len(labels))

assert len(labels) == Dataset.shape[0]


# ============================================================
# Save cluster labels
# ============================================================

cluster_df = save_cluster_labels(
    labels=labels,
    groups=groups,
    group_names=GROUP_NAMES,
    rep_dir=rep_dir
)


# ============================================================
# Plot clustering
# ============================================================

plot_3d_clustering(X_transformed, labels, N_CLUSTERS)


# ============================================================
# Cluster representation per condition
# ============================================================

count_df, percent_df = summarize_clusters(
    labels=labels,
    groups=groups,
    group_names=GROUP_NAMES
)

print("Counts:")
print(count_df)

print("\nPercent of cells per condition:")
print(percent_df)


