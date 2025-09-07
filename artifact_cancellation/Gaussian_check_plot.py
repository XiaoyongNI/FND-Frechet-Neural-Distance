import matplotlib.pyplot as plt
import numpy as np


### load results list in npy file ##############################
cond = "seizure" # or "non_seizure"
file_names = [
    f"Gaussian_check_results_{cond}_N50_5k.npy",
    f"Gaussian_check_results_{cond}_N10k_100k.npy",
]

# Initialize empty lists for concatenation
all_N_vals = []
all_ks_means = []
all_ks_stds = []
all_mu_means = []
all_mu_stds = []
all_var_means = []
all_var_stds = []

# Load and merge
for file_name in file_names:
    results = np.load(file_name, allow_pickle=True).item()
    N_vals = results[cond]['N']
    
    # Unpack (mean, stderr) tuples
    ks_means, ks_stds = zip(*results[cond]['ks'])
    mu_means, mu_stds = zip(*results[cond]['mu'])
    var_means, var_stds = zip(*results[cond]['var'])

    all_N_vals.extend(N_vals)
    all_ks_means.extend(ks_means)
    all_ks_stds.extend(ks_stds)
    all_mu_means.extend(mu_means)
    all_mu_stds.extend(mu_stds)
    all_var_means.extend(var_means)
    all_var_stds.extend(var_stds)

# Optionally, sort the combined results by N
sorted_indices = np.argsort(all_N_vals)
all_N_vals = np.array(all_N_vals)[sorted_indices]
all_ks_means = np.array(all_ks_means)[sorted_indices]
all_ks_stds = np.array(all_ks_stds)[sorted_indices]
all_mu_means = np.array(all_mu_means)[sorted_indices]
all_mu_stds = np.array(all_mu_stds)[sorted_indices]
all_var_means = np.array(all_var_means)[sorted_indices]
all_var_stds = np.array(all_var_stds)[sorted_indices]

### Plot #################################################################
fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
axs[0].errorbar(all_N_vals, all_ks_means, yerr=all_ks_stds, label=cond, marker='o')
axs[1].errorbar(all_N_vals, all_mu_means, yerr=all_mu_stds, label=cond, marker='o')
axs[2].errorbar(all_N_vals, all_var_means, yerr=all_var_stds, label=cond, marker='o')

# Titles and labels
axs[0].set_title('Average KS Statistic vs N')
axs[0].set_ylabel('KS Avg')

axs[1].set_title('Mean Norm of μ vs N')
axs[1].set_ylabel('||μ||')

axs[2].set_title('Mean Norm of Variance vs N')
axs[2].set_ylabel('||σ²||')
axs[2].set_xlabel('N (Number of Samples)')

for ax in axs:
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(f'Gaussian_check_plot_{cond}.png', dpi=300)
