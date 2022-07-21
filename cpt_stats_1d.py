# Imports
import numpy as np
import scipy.io
import scipy.stats
import skimage.measure

# Load test data for statistical analysis as dubects x time
data1 = scipy.io.loadmat("data1.mat")["data1"]
data2 = scipy.io.loadmat("data2.mat")["data2"]

# Get dims
n_obs, n_times = data1.shape

# Set params
n_perms = 1000
pval_voxel = 0.05
pval_cluster = 0.05

# Determine threshold
thresh_t = scipy.stats.t.ppf(1 - pval_voxel, n_obs - 1)

# Init matrices
permuted_t = np.zeros((n_perms, n_times))
perm_dist = np.zeros((n_perms, 3))
max_nvox = np.zeros((n_perms, 1))

# Iterate permutations
for perm in range(n_perms):

    # Permute data
    toflip = np.random.choice(n_obs, int(np.floor(n_obs / 2)))
    d1_perm = data1.copy()
    d1_perm[toflip, :] = data2[toflip, :].copy()
    d2_perm = data2.copy()
    d2_perm[toflip, :] = data1[toflip, :].copy()

    # Calculate and save t values
    tnum = np.squeeze(np.mean(d1_perm - d2_perm, axis=0))
    tdenum = np.squeeze(np.std(d1_perm - d2_perm, axis=0)) / np.sqrt(n_obs)
    fake_t = np.divide(tnum, tdenum)
    # permuted_t[perm, :] = fake_t.copy()

    # Identify positive clusters
    t_binary = (fake_t > thresh_t).astype("int")
    clust_labels_pos, n_clusts_pos = skimage.measure.label(
        t_binary, return_num=True, connectivity=None
    )

    # Identify negative clusters
    t_binary = (fake_t < -thresh_t).astype("int")
    clust_labels_neg, n_clusts_neg = skimage.measure.label(
        t_binary, return_num=True, connectivity=None
    )

    # Determine min and mux sum of t in clusters
    sum_t = [0]
    n_vox = [0]
    for cluster_idx in range(n_clusts_pos):
        sum_t.append(np.sum(fake_t[clust_labels_pos == cluster_idx + 1]))
        n_vox.append(sum(clust_labels_pos == cluster_idx + 1))
    for cluster_idx in range(n_clusts_neg):
        sum_t.append(np.sum(fake_t[clust_labels_neg == cluster_idx + 1]))
        n_vox.append(sum(clust_labels_neg == cluster_idx + 1))
    perm_dist[perm, :] = np.min(sum_t), np.max(sum_t), np.max(n_vox)

# Calculate cluster thresholds
clust_thresh_upper = np.percentile(perm_dist[:, 1], 100 - pval_cluster * 100, axis=0)
clust_thresh_lower = np.percentile(perm_dist[:, 0], pval_cluster * 100, axis=0)
clust_thresh_nvox = np.percentile(perm_dist[:, 2], 100 - pval_cluster * 100, axis=0)

# Calculate t values for real data
tnum = np.squeeze(np.mean(data1 - data2, axis=0))
tdenum = np.squeeze(np.std(data1 - data2, axis=0)) / np.sqrt(n_obs)
tmat = np.divide(tnum, tdenum)

# Identify positive clusters
t_binary = (tmat > thresh_t).astype("int")
clust_labels_pos, n_clusts_pos = skimage.measure.label(
    t_binary, return_num=True, connectivity=None
)

# Identify negative clusters
t_binary = (tmat < -thresh_t).astype("int")
clust_labels_neg, n_clusts_neg = skimage.measure.label(
    t_binary, return_num=True, connectivity=None
)

# Compile a cluster collection
sum_t = []
n_vox = []
polarity = []
cluster_idx = []
for cluster_label in range(n_clusts_pos):
    sum_t.append(np.sum(tmat[clust_labels_pos == cluster_label + 1]))
    n_vox.append(sum(clust_labels_pos == cluster_label + 1))
    polarity.append(1)
    cluster_idx.append(clust_labels_pos == cluster_label + 1)
for cluster_label in range(n_clusts_neg):
    sum_t.append(np.sum(tmat[clust_labels_neg == cluster_label + 1]))
    n_vox.append(sum(clust_labels_neg == cluster_label + 1))
    polarity.append(-1)
    cluster_idx.append(clust_labels_neg == cluster_label + 1)



# max_tsum[perm, :] = np.min(sum_t), np.max(sum_t)

# % Determine min and mux sum of t in clusters
# sum_t = [];
# sum_vox = [];
# for clu = 1 : n_clusts
#     sum_t(end + 1) = sum(fake_t(clust_labels == clu));
#     sum_vox(end + 1) = sum(clust_labels == clu);
# end

# % Determine upper and lower thresholds
# clust_thresh_lower = prctile(max_tsum(:, 1), pval_cluster * 100);
# clust_thresh_upper = prctile(max_tsum(:, 2), 100 - pval_cluster * 100);
# clust_thresh_nvox  = prctile(max_nvox, 100 - pval_cluster * 100);

# % Determine cluster to keep
# clust2keep = find(sum_t <= clust_thresh_lower | sum_t >= clust_thresh_upper);

# % Build cluster vector
# clust_vector = zeros(size(tmat));
# for clu = 1 : length(clust2keep)
#     clust_vector(clust_labels == clust2keep(clu)) = 1;
# end

# % Set the flag of significance
# sig_flag = logical(sum(clust_vector(:)));

# % Calculate effect sizes
# x = tvals.^2 ./ (tvals.^2 + (n_subjects - 1));
# apes = x - (1 - x) .* (1 / (n_subjects - 1));

# % Calculate averages
# mean_data1 = squeeze(mean(data1, 1));
# mean_data2 = squeeze(mean(data2, 1))
