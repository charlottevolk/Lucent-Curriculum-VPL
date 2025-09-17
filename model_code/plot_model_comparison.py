import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import starbars
import os

deg = u'\N{DEGREE SIGN}'

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data.append(row[0])
    data = np.array(data)
    data = data.astype(np.float64)
    return data

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

colours = sns.color_palette("colorblind", 6)
cols = [colours[1], colours[3], colours[4]]

lambda_reg = 0.0001

base_dir = '' # specify if model data stored somewhere else
save_dir = 'saved_outputs/plots/'

plot_frozen_readout = False

if plot_frozen_readout:
    x_labels = ['Non-sequential', 'Sequential', 'Shuffled 6:1', 'Frozen Readout']
else:
    x_labels = ['Non-sequential', 'Shuffled', 'Sequential']

labels = x_labels

noise_sd = 0.02
confidence_sd = 0.3

fig, ax = plt.subplots(figsize=(7,5), dpi=300)
ax.xaxis.set_ticks(range(len(x_labels)))
ax.xaxis.set_ticklabels(x_labels, rotation=0)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) 
plt.xlabel("Model", labelpad=12)
plt.ylabel("Transfer Accuracy", labelpad=12)

trials = range(1,21)

means = np.zeros(len(labels))
all_vals = [[] for _ in range(len(labels))]

factor = 6.0 # change if desired
forced_steps = 25

for trial_num in trials:
    seq_dir = base_dir + 'skip_seq_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update/'
    shuff_dir = base_dir + 'skip_shuff_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update/'
    nonseq_dir = base_dir + 'skip_nonseq_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update/'

    shuff_forced_seq = base_dir + 'skip_shuff_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update_forced_sequential_with_weighted_samples_beginning_'+str(forced_steps)+'_steps_factor_'+str(factor)+'/'
    frozen_readout = base_dir + 'FROZEN_READOUT_skip_nonseq_from_forced_seq_w_factor_6.0_num_steps_25_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update/'

    if plot_frozen_readout:
        dirs = [nonseq_dir_end, seq_dir, shuff_forced_seq, frozen_readout]
    else:
        dirs = [nonseq_dir_end, shuff_dir, seq_dir]

    for i, (dir, label) in enumerate(zip(dirs, labels)):
        try:
            transfer_accuracy_data = read_data(dir, 'data/skip/SF_doubled_lr_0.0001/transfer_accuracy_ref_0_sf_0.1_sep_1.0_lr_0.0001_trial_'+str(trial_num)+'.csv')
        except:
            print('not found', dir)
            continue
        specs = np.mean(transfer_accuracy_data)
        means[i] += specs
        all_vals[i].append(specs)

# Functions for statistical tests         
def check_normality(data):
    """Perform Shapiro-Wilk test for normality."""
    _, p_value = stats.shapiro(data)
    return p_value > 0.05  # If p > 0.05, assume normal distribution

def compute_statistical_significance(group1, group2):
    """Computes statistical significance between two groups."""
    normal1 = check_normality(group1)
    normal2 = check_normality(group2)
    normal1=True
    normal2=True

    if normal1 and normal2:
        stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test
        test_used = "t-test"
    else:
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')  # Mann-Whitney U test
        test_used = "Mann-Whitney U test"

    return {"p_value": p_value, "test": test_used}

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

if plot_frozen_readout:
    groups = {
        'Non-seq': all_vals[0],
        'Sequential': all_vals[1],
        'Shuffled': all_vals[2],
        'Frozen-readout': all_vals[3]
    }

    comparisons = [
            ('Non-seq', 'Sequential'),
            ('Sequential', 'Shuffled'),
            ('Sequential', 'Frozen-readout'),
            ('Shuffled', 'Frozen-readout'),
            # ('Non-seq', 'Frozen-readout')
        ]

else:
    groups = {
        'Non-seq': all_vals[0],
        'Shuff': all_vals[1],
        'Seq': all_vals[2],
    }

    comparisons = [
        ('Non-seq', 'Shuff'),
        ('Shuff', 'Seq'),
        ('Non-seq', 'Seq')
    ]

# Compute significance for each comparison
results = {pair: compute_statistical_significance(groups[pair[0]], groups[pair[1]]) for pair in comparisons}

# Plotting
plt.figure(figsize=(8,5), dpi=300)
ax=plt.gca()
if plot_frozen_readout:
    colors = {0: cols[0], 1: cols[2], 2: cols[3], 3: cols[4]}
else:
    colors = {0: cols[0], 1: cols[1], 2: cols[2]}
ax = sns.boxplot(data=all_vals, palette=colors)
for patch in ax.artists:
    fc = patch.get_facecolor()
    patch.set_facecolor(plt.colors.to_rgba(fc, 0.3))
sns.swarmplot(data=all_vals, ax=ax, color='k')
if plot_frozen_readout:
    plt.xticks([0,1,2,3], ['Non-seq', 'Sequential', 'Shuffled', 'Frozen-readout'])
else:
    plt.xticks([0,1,2], ["Non-seq", "Shuff", "Seq"])

annotations = []
for pair in results.keys():
    annotation = (pair[0], pair[1], results[pair]['p_value'])
    annotations.append(annotation)
# Add statistical significance bars using starbars
starbars.draw_annotation(annotations)

if plot_frozen_readout:
    plt.xticks([0,1,2,3], ['Non-seq', 'Seq', 'Shuff 6:1', 'Frozen'])
else:
    plt.xticks([0,1,2], ["Nonseq", "Shuff", "Seq"])

# Add significance annotations
x_positions = {name: i for i, name in zip(groups.keys(), range(len(groups) - 1))}

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('Transfer Accuracy', labelpad=12)
plt.xlabel('Condition', labelpad=12)
plt.tight_layout()
plt.savefig(save_dir+'model_comparison_transfer_accuracy.svg')
# plt.show()