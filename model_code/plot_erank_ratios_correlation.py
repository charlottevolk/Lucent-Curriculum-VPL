import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

from scipy.stats import sem
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from skip_alexnet import AlexNet
import torch

SMALLEST_SIZE = 18
SMALL_SIZE = 22
TITLE_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 12

deg = u'\N{DEGREE SIGN}'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

sns.set_palette("colorblind")
colors = sns.color_palette("colorblind")[:3]

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data.append(row[0])
    data = np.array(data)
    data = data.astype(np.float64)
    return data

# Define Paths
base_dir = 'saved_outputs/'
save_dir = 'saved_outputs/plots/'
activation_dir = base_dir+'skip_activations_two_step_models_imagenet_100/'
models = ['1:10', '1:9', '1:8', '1:7', '1:6', '1:5', '1:4', '1:3', '1:2', '1:1', '2:1','3:1', '4:1', '5:1', '6:1', '7:1', '8:1', '9:1', '10:1']
trials = range(1, 21)

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3
num_steps = 25

num_neurons = 150

shuffled_dir = base_dir+'skip_shuff_5.0_1.0_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update/'
seq_dirs = [base_dir+'skip_shuff_5.0_1.0_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update_forced_sequential_with_weighted_samples_beginning_'+str(num_steps)+'_steps_factor_'+str(factor)+'/' for factor in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
nonseq_dirs = [base_dir+'skip_shuff_5.0_1.0_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update_forced_antisequential_with_weighted_samples_beginning_'+str(num_steps)+'_steps_factor_'+str(factor)+'/' for factor in [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]]

dirs = nonseq_dirs + shuffled_dir + seq_dirs

# Store results
effective_ranks = {model: [] for model in models}
test_accuracies = {model: [] for model in models}

epsilon = 1e-8

# Compute Effective Rank and Accuracy for Each Model
for model_dir, model in zip(dirs, models):
    print(f"Processing model: {model}, Path: {model_dir}")

    for trial in trials:
        # Load activations
        activations_all = np.load(activation_dir+'activations/SF_doubled_lr_0.0001/100_imgs_all_activations_no_noise_1000_imagenet_0_sf_0.05_sep_0.5_lr_0.0001_model_trial_'+str(trial)+'_batch_size_1.npy')
        activations_all = np.float64(activations_all)
        print("Loaded activations...")

        path = model_dir+'models/original_model_0_sf_0.05_sep_1.0_trial_'+str(trial)+'.pth'
        alexnet = AlexNet()
        alexnet.load_state_dict(torch.load(path))
        readout_weights = alexnet.fc1.weight.data[0]

        # Load important neuron indices
        source_path = model_dir + 'data/skip/SF_doubled_lr_0.0001/max_abs_neurons_'+str(num_neurons)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv'
        neuron_indices = np.loadtxt(source_path, delimiter=",", dtype=int)[1]
        sorted_indices = np.argsort(neuron_indices)
        important_neurons = neuron_indices[sorted_indices]
        imp_activations = activations_all[:, important_neurons]

        # Normalize activations (zero mean, unit variance across neurons)
        imp_activations = (imp_activations - np.mean(imp_activations, axis=0)) / (np.std(imp_activations, axis=0) + epsilon)

        # Handle any NaN values (happens if std=0 for some neurons)
        imp_activations = np.nan_to_num(imp_activations)

        # Compute SVD
        U, S, Vh = np.linalg.svd(imp_activations, full_matrices=False)

        # Compute Effective Rank
        p = S**2 / np.sum(S**2)
        effective_rank = np.exp(-np.sum(p * np.log(p)))
        effective_ranks[model].append(effective_rank)

        # Load test accuracy
        transfer_data =  read_data(model_dir, 'data/transfer_accuracy_ref_0_sf_0.1_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv')
        test_acc = np.mean(transfer_data)
        test_accuracies[model].append(np.mean(test_acc))

# Convert lists to arrays
effective_ranks = {model: np.array(ranks) for model, ranks in effective_ranks.items()}
test_accuracies = {model: np.array(accs) for model, accs in test_accuracies.items()}

# Compute mean and std across trials
mean_erank = {model: np.mean(effective_ranks[model]) for model in models}
std_erank = {model: sem(effective_ranks[model]) for model in models}

mean_acc = {model: np.mean(test_accuracies[model]) for model in models}
std_acc = {model: sem(test_accuracies[model]) for model in models}

# --- Colorbar plotting by model order ---
cmap = plt.cm.viridis
model_indices = np.arange(len(models))
norm = plt.Normalize(vmin=model_indices.min(), vmax=model_indices.max())

plt.figure(figsize=(6,6), dpi=300)
scatters = []
for (i, model), color in zip(enumerate(models), colors):
    # Errorbar for mean
    plt.errorbar(mean_erank[model], mean_acc[model], 
                 xerr=std_erank[model], yerr=std_acc[model],
                 capsize=5, lw=5, capthick=5,
                 color=color
                )
    # Mean point
    scat = plt.scatter(mean_erank[model], mean_acc[model], s=150, color=color, label=model)
    scatters.append(scat)

ax = plt.gca()

# Add colorbar for model order
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_ticks([model_indices[0], model_indices[10], model_indices[-1]])
cbar.set_ticklabels(['1:10', '1:1', '10:1'])
cbar.set_label('Easy:hard ratio')

# Fit and plot regression using seaborn regplot
import seaborn as sns
from scipy.stats import pearsonr

# Prepare data: x = mean effective rank, y = mean accuracy
x = np.array([mean_erank[model] for model in models])
y = np.array([mean_acc[model] for model in models])

# Pearson correlation and p-value
r_val, p_val = pearsonr(x, y)
print(f"Pearson r of Effective Rank vs. Test Accuracy (means): {r_val:.4f}, p = {p_val:.4g}")

# Plot regression with seaborn
sns.regplot(x=x, y=y, scatter=False, ax=ax, color='black', line_kws={'linestyle':'--','linewidth':2}, label=f'r={r_val:.2f}, p={p_val:.2g}')
plt.scatter(x, y)

plt.ylabel("Transfer Accuracy", labelpad=12)
plt.xlabel("Effective Rank", labelpad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "plots/effective_rank_vs_transfer_accuracy_single_sample_updates_"+str(num_neurons)+"_neurons.svg"))#no_noise_with_forced_seq_antiseq_with_weighted_samples_factor_"+str(factor)+"_neurons_single_sample_activations_distribution_normalized_different_palette_smaller_img_"+str(num_neurons)+"_neurons.svg"))
plt.show()
