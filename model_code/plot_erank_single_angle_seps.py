import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import sem

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

sns.set_palette("GnBu_r",5)
colors = sns.color_palette("GnBu_r",5)
colors = [colors[1], colors[4]]

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
single_angle_sep_dir = base_dir + 'single_angle_seps/'

activation_dir = base_dir+f'collecting_activations_imagenet_100_AlexNet/'
save_dir = base_dir + 'plots/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

models = ['0.5', '1.0', '2.0', '5.0', '10.0']
trials = range(1, 21)

transfer_condition = 'SF' # or 'ref_ori'

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

# Store results
effective_ranks = {model: [] for model in models}
test_accuracies = {model: [] for model in models}

epsilon = 1e-8

# Compute Effective Rank and Accuracy for Each Model
for model in models:
    # if model_name == 'AlexNet_skip':
    #     model_dir = base_dir+f'single_angle_sep_doubled_SF_{model_name}_bs20/{model}_angle_sep/'
    # elif model_name == 'VGG11' or model_name == 'VGG11_skip':
    # else:
    #     model_dir = base_dir+f'single_angle_sep_doubled_SF_{model_name}_bs20_zeroed_readout_init/{model}_angle_sep/'
    # else:
    model_dir = f{single_angle_sep_dir}/{model}_angle_sep/'
    print(f"Processing model: {model}, Path: {model_dir}")

    for trial in trials:
        # Load activations
        activations_all = np.load(activation_dir+'activations/100_imgs_all_activations_no_noise_1000_imagenet_0_sf_0.05_sep_0.5_lr_0.0001_model_trial_'+str(trial)+'_batch_size_1.npy')
        activations_all = np.float64(activations_all)
        print("Loaded activations...")

        path = model_dir+'models/original_model_0_sf_0.05_sep_'+model+'_trial_'+str(trial)+'.pth'
        trained_model = AlexNet()
        trained_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        readout_weights = trained_model.fc1.weight.data[0]

        # Load important neuron indices
        num_neurons = 150
        source_path = model_dir + 'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+model+'_lr_0.0001_trial_'+str(trial)+'.csv'
        neuron_indices = np.loadtxt(source_path, delimiter=",", dtype=float)[1]
        neuron_indices = np.rint(neuron_indices).astype(np.int64)
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
        s2 = S ** 2
        s2_sum = np.sum(s2)
        if s2_sum <= 0:
            effective_rank = 0.0
        else:
            p = s2 / s2_sum
            p = p[p > 0]
            effective_rank = np.exp(-np.sum(p * np.log(p)))
        effective_ranks[model].append(effective_rank)

        # Load test accuracy
        transfer_data =  read_data(model_dir, f'data/transfer_accuracy_ref_{ref_angle_test}_sf_{spatial_freq_test}_sep_'+model+'_lr_0.0001_trial_'+str(trial)+'.csv')
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
plt.savefig(save_dir + f"effective_rank_single_angle_seps_{num_neurons}_neurons.svg")
# plt.show()
