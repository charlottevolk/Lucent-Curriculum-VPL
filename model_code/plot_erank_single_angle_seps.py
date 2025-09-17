import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import sem

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

sns.set_palette("GnBu_r",5)
colors = sns.color_palette("GnBu_r",5)
# colors = [colors[0], colors[1]]
# sns.set_palette("colorblind")

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data.append(row[0])
    data = np.array(data)
    data = data.astype(np.float64)
    return data

# Set color palette
# sns.set_palette("colorblind", 3)

# Define Paths
base_dir = 'saved_outputs/'
save_dir = base_dir
activation_dir = base_dir+'skip_activations_two_step_models_imagenet_100/'
models = ['1.0', '5.0']
trials = range(1, 21)

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3

# Store results
effective_ranks = {model: [] for model in models}
test_accuracies = {model: [] for model in models}

epsilon = 1e-8

# Compute Effective Rank and Accuracy for Each Model
for model in models:
    model_dir = base_dir+'skip_model_full_training_non_sequential_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update/'
    print(f"Processing model: {model}, Path: {model_dir}")

    for trial in trials:
        # Load activations
        activations_all = np.load(activation_dir+'activations/100_imgs_all_activations_no_noise_1000_imagenet_0_sf_0.05_sep_0.5_lr_0.0001_model_trial_'+str(trial)+'_batch_size_1.npy')
        activations_all = np.float64(activations_all)
        print("Loaded activations...")

        path = model_dir+'models/original_model_0_sf_0.05_sep_'+model+'_trial_'+str(trial)+'.pth'
        #path = save_dir + 'models/SF_doubled_lr_0.0001/original_model_condition_hard_granite_marble_trial_1.pth'
        alexnet = AlexNet()
        alexnet.load_state_dict(torch.load(path))
        readout_weights = alexnet.fc1.weight.data[0]

        # Load important neuron indices
        num_neurons = 150
        source_path = model_dir + 'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+model+'_lr_0.0001_trial_'+str(trial)+'.csv'
        neuron_indices = np.loadtxt(source_path, delimiter=",", dtype=int)[1]
        sorted_indices = np.argsort(neuron_indices)
        important_neurons = neuron_indices[sorted_indices]

        # Extract activations of important neurons
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
        transfer_data =  read_data(model_dir, 'data/transfer_accuracy_ref_0_sf_0.1_sep_'+model+'_lr_0.0001_trial_'+str(trial)+'.csv')
        test_acc = np.mean(transfer_data)
        test_accuracies[model].append(np.mean(test_acc))

# Convert lists to arrays
effective_ranks = {model: np.array(ranks) for model, ranks in effective_ranks.items()}
test_accuracies = {model: np.array(accs) for model, accs in test_accuracies.items()}

# Compute mean and sem across trials
mean_erank = {model: np.mean(effective_ranks[model]) for model in models}
sem_erank = {model: sem(effective_ranks[model]) for model in models}

mean_acc = {model: np.mean(test_accuracies[model]) for model in models}
sem_acc = {model: sem(test_accuracies[model]) for model in models}

# Plot Results
plt.figure(figsize=(7,5), dpi=300)
for i, model in enumerate(models):
    plt.errorbar(mean_erank[model], mean_acc[model], 
                 xerr=sem_erank[model], yerr=sem_acc[model],
                 capsize=5, lw=5, capthick=5, color=colors[i])
    plt.scatter(mean_erank[model], mean_acc[model], s=150, label=model, color=colors[i])
    plt.scatter(effective_ranks[model], test_accuracies[model], s=150, alpha=0.3, color=colors[i])

plt.xlabel("Effective Rank", labelpad=12)
plt.ylabel("Transfer Accuracy", labelpad=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "plots/effective_rank_single_angle_seps.svg"))
plt.show()
