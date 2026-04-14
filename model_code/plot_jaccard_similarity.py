import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import seaborn as sns
from scipy.stats import sem

SMALLEST_SIZE = 12
SMALL_SIZE = 20
TITLE_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

deg = u'\N{DEGREE SIGN}'

sns.set_palette("colorblind")

base_dir = 'saved_outputs/'
seq_dir = base_dir+'sequential_doubled_SF_AlexNet/'
shuff_dir = base_dir+'shuffled_doubled_SF_AlexNet/'
nonseq_dir = base_dir+'nonsequential_doubled_SF_AlexNet/'

plotting_confidence = True

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3
num_trials = 20

save_dir = base_dir + 'plots/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

base_dir_list = [nonseq_dir, shuff_dir, seq_dir]

model_names = ['Non-seq', 'Shuff', 'Seq']

num_indices = 150

def compute_jaccard_similarity(subspace_history):
    """
    subspace_history: list of sets, each containing the top-100 indices at a given time
    returns: list of Jaccard similarities between consecutive time points
    """
    jaccard_similarities = []
    for i in range(1, len(subspace_history)):
        a = subspace_history[i - 1]
        b = subspace_history[i]
        intersection = len(a & b)
        union = len(a | b)
        jaccard = intersection / union if union > 0 else 0
        jaccard_similarities.append(jaccard)
    return jaccard_similarities

def compute_jaccard_similarity_with_reference(subspace_history, reference_index=10):
    """
    subspace_history: list of sets
    reference_index: index in the history to which all other subspaces are compared
    returns: list of Jaccard similarities between each subspace and the reference
    """
    reference = subspace_history[reference_index]
    jaccard_similarities = []
    for i in range(len(subspace_history)):
        a = subspace_history[i]
        intersection = len(a & reference)
        union = len(a | reference)
        jaccard = intersection / union if union > 0 else 0
        jaccard_similarities.append(jaccard)
    return jaccard_similarities


plt.figure(figsize=(7, 5), dpi=300)
for model_name, model_dir in zip(model_names, base_dir_list):
    
    all_similarities = []
    for trial in range(1, num_trials+1):
        first_step_weights = np.load(model_dir+'weights/all_fc_weights_ref_0_sf_0.05_sep_5.0_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)
        second_step_weights = np.load(model_dir+'weights/all_fc_weights_ref_0_sf_0.05_sep_1.0_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)
        all_weights = np.concatenate((first_step_weights, second_step_weights), axis=0)

        iteration_steps = range(0,50)
        subspace_history = []
        for i in iteration_steps:
            weights = all_weights[i].squeeze()
            top_100 = np.argpartition(np.abs(weights), -num_indices)[-num_indices:]
            subspace_history.append(set(top_100))
        
        jaccard_similarities = compute_jaccard_similarity(subspace_history)
        all_similarities.append(jaccard_similarities)

    all_similarities = np.array(all_similarities)
    mean_sims = np.mean(all_similarities, axis=0)
    sem_sims = sem(all_similarities, axis=0)

    plot_steps = range(20,1000,20)
    # Plot
    plt.plot(plot_steps, mean_sims, lw=5, solid_capstyle='round', label=model_name)
    plt.fill_between(plot_steps, mean_sims-sem_sims, mean_sims+sem_sims, alpha=0.3, interpolate=True)

plt.xlabel('Steps')
plt.ylabel('J')
# Add vertical & horizontal dashed lines
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
plt.axvline(x=200, color='black', linestyle='--', linewidth=1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir+'jaccard_similarity.svg')