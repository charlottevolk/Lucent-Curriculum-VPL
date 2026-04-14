import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

from scipy.stats import sem
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from models.skip_alexnet import AlexNet
from models.noskip_alexnet import AlexNet as AlexNet_noskip
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

base_dir = 'saved_outputs/'
seq_dir = base_dir + 'sequential_doubled_SF_AlexNet/'
shuff_dir = base_dir + 'shuffled_doubled_SF_AlexNet/'
nonseq_dir = base_dir + 'nonsequential_doubled_SF_AlexNet/'

random_readout = False
by_contribution = False
transfer_condition = 'SF' # or 'ref_ori'

activation_dir = base_dir+f'collecting_activations_imagenet_100_AlexNet/'
save_dir = base_dir + 'plots/'

dirs = [nonseq_dir, shuffled_dir, seq_dir]
models = ['Non-sequential', 'Shuffled', 'Sequential']
trials = range(1, 21)

if transfer_condition == 'ref_ori':
    root_dir = 'StimulusImages/SG_train_double_sf/'
    test_root_dir = 'StimulusImages/SG_test_changed_ref_15/sep_1.0/'
    spatial_freq_train = 0.05
    spatial_freq_test = 0.05
    ref_angle_train = 0
    ref_angle_test = 15
elif transfer_condition == 'SF':
    root_dir = 'StimulusImages/SG_train_double_sf/'
    test_root_dir = 'StimulusImages/SG_test_double_sf/sep_1.0/'
    spatial_freq_train = 0.05
    spatial_freq_test = 0.1
    ref_angle_train = 0
    ref_angle_test = 0

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3
num_steps = 25
num_neurons = 150

# Store results
effective_ranks = {model: [] for model in models}
test_accuracies = {model: [] for model in models}

epsilon = 1e-8

# Compute Effective Rank and Accuracy for Each Model
for model_dir, model in zip(dirs, models):
    print(f"Processing model: {model}, Path: {model_dir}")

    for trial in trials:
        # Load activations
        activations_all = np.load(activation_dir+'activations/100_imgs_all_activations_no_noise_1000_imagenet_0_sf_0.05_sep_0.5_lr_0.0001_model_trial_'+str(trial)+'_batch_size_1.npy')
        activations_all = np.float64(activations_all)
        print("Loaded activations...")

        path = model_dir+'models/original_model_0_sf_0.05_sep_1.0_trial_'+str(trial)+'.pth'
        trained_model = AlexNet()
        trained_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        readout_weights = trained_model.fc1.weight.data[0]
        
        if random_readout:
            neuron_indices = np.random.randint(0, activations_all.shape[1], size=num_neurons)
        else:
            if by_contribution:
                source_file = np.loadtxt(model_dir + 'data/lesioning_by_contribution_eps_0.001_200_imgs_all_neurons_dictionary_ref_'+str(ref_angle_test)+'_sf_'+str(spatial_freq_test)+'_sep_1.0_lr_0.0001_trial_'+str(trial)+'.csv', delimiter=',', dtype='float')
                # Sort by loss (column 1) in ascending order to get neurons with lowest loss
                source_file = source_file[source_file[:, 1].argsort()]
                neuron_indices = []
                accs = []
                for entry in source_file:
                    neuron_index = int(entry[0])
                    neuron_indices.append(neuron_index)
                    neuron_accuracy = float(entry[1])
                    accs.append(neuron_accuracy)

            else:
                source_path = model_dir + 'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv'
                neuron_indices = np.loadtxt(source_path, delimiter=",", dtype=int)[1]

        sorted_indices = np.argsort(neuron_indices)
        # Get activations from important neurons
        imp_activations = activations_all[:, neuron_indices[:num_neurons]]
        
        # Weight activations by readout weights to get readout subspace representation
        imp_readout_weights = readout_weights[neuron_indices[:num_neurons]].cpu().numpy()
        imp_activations = imp_activations * imp_readout_weights  # Weight each neuron's activations

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
        transfer_data =  read_data(model_dir, f'data/transfer_accuracy_ref_{ref_angle_test}_sf_{spatial_freq_test}_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv')
        test_acc = np.mean(transfer_data)
        test_accuracies[model].append(np.mean(test_acc))
        # print(f"Trial {trial}: Effective Rank = {effective_rank}, Test Accuracy = {test_acc}")

# Convert lists to arrays
effective_ranks = {model: np.array(ranks) for model, ranks in effective_ranks.items()}
test_accuracies = {model: np.array(accs) for model, accs in test_accuracies.items()}

# Compute mean and std across trials
mean_erank = {model: np.mean(effective_ranks[model]) for model in models}
std_erank = {model: sem(effective_ranks[model]) for model in models}

mean_acc = {model: np.mean(test_accuracies[model]) for model in models}
std_acc = {model: sem(test_accuracies[model]) for model in models}

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
    # All points for this model
    plt.scatter(effective_ranks[model], test_accuracies[model], s=150, alpha=0.3, color=color)

ax = plt.gca()

# Fit and plot regression using seaborn regplot
import seaborn as sns
from scipy.stats import pearsonr

# Prepare data: x = all trial effective ranks, y = all trial accuracies
x = np.concatenate([effective_ranks[model] for model in models])
y = np.concatenate([test_accuracies[model] for model in models])

# Pearson correlation and p-value
r_val, p_val = pearsonr(x, y)
print(f"Pearson r of Effective Rank vs. Test Accuracy (all datapoints): {r_val:.4f}, p = {p_val:.4g}")

# Plot regression with seaborn
sns.regplot(x=x, y=y, scatter=False, ax=ax, color='black', line_kws={'linestyle':'--','linewidth':2}, label=f'r={r_val:.2f}, p={p_val:.2g}')

plt.ylabel("Transfer Accuracy", labelpad=12)
plt.xlabel("Effective Rank", labelpad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.yticks([0.6, 0.65, 0.7, 0.75])
plt.legend()
plt.tight_layout()
if random_readout:
    plt.savefig(os.path.join(save_dir, f"effective_rank_random_readout_{num_neurons}_neurons_{model_name}.svg"))
else:
    if by_contribution:
        plt.savefig(os.path.join(save_dir, f"effective_rank_by_contribution_{num_neurons}_neurons_{model_name}.svg"))
    else:
        plt.savefig(os.path.join(save_dir, f"effective_rank_by_weight_{num_neurons}_neurons_{model_name}.svg"))
# plt.show()
