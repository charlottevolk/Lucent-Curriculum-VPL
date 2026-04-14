import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as stats
from scipy.stats import sem
from itertools import combinations
import starbars

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

SMALLEST_SIZE = 12
SMALL_SIZE = 20
TITLE_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 12

colours = sns.color_palette("GnBu_r", 3)
plt.rcParams['svg.fonttype'] = 'none'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

deg = u'\N{DEGREE SIGN}'

base_dir = 'saved_outputs/'
seq_dir = base_dir+'sequential_doubled_SF_AlexNet/'
shuffled_dir = base_dir+'shuffled_doubled_SF_AlexNet/'
nonseq_dir = base_dir+'nonsequential_doubled_SF_AlexNet/'
normalized = True

plotting_confidence = True

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3
num_trials = 20

save_dir = base_dir + 'plots/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

base_dir_list = [shuffled_dir, seq_dir, nonseq_dir]

model_names = [
    'Shuff',
    'Seq',
    'Non-seq',
]

num_indices = 150

subspaces = {}

for model_name, model_dir in zip(model_names, base_dir_list):
    all_trial_subspaces = []
    for trial in range(1, num_trials+1):
        csv_path = (
            model_dir+'weights/top150_final_fc_indices_ref_0_sf_0.05_sep_1.0_lr_0.0001_trial_'+str(trial)+'.csv')
        all_indices = np.loadtxt(csv_path, delimiter=',').squeeze()
        top_neurons = np.asarray(all_indices, dtype=np.int64)
        all_trial_subspaces.append(set(top_neurons.tolist()))
    subspaces[model_name] = all_trial_subspaces

# Per-model: mean Jaccard of each trial against all 19 other trials (used as baseline)
within_mean_scores = [
    [
        np.mean([jaccard_similarity(subspaces[m][t], subspaces[m][t2]) for t2 in range(num_trials) if t2 != t])
        for t in range(num_trials)
    ]
    for m in model_names
]
within_scores_by_model = {m: np.array(within_mean_scores[i]) for i, m in enumerate(model_names)}

pairwise_labels = []
pairwise_scores = []

for model_a, model_b in combinations(model_names, 2):
    sims = []
    for trial_idx in range(num_trials):
        raw = jaccard_similarity(subspaces[model_a][trial_idx], subspaces[model_b][trial_idx])
        baseline = (within_scores_by_model[model_a][trial_idx] + within_scores_by_model[model_b][trial_idx]) / 2
        if normalized:
            sims.append(raw / baseline if baseline > 0 else 0.0)
        else:
            sims.append(raw)
    pairwise_labels.append(f'{model_a}/{model_b}')
    pairwise_scores.append(np.array(sims))

def check_normality(data):
    _, p_value = stats.shapiro(data)
    return p_value > 0.05

def compute_statistical_significance(group1, group2):
    normal1 = check_normality(group1)
    normal2 = check_normality(group2)
    normal1 = True
    normal2 = True
    if normal1 and normal2:
        stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        test_used = "t-test"
    else:
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_used = "Mann-Whitney U test"
    return {"p_value": p_value, "test": test_used}

groups = {label: scores.tolist() for label, scores in zip(pairwise_labels, pairwise_scores)}
comparisons = list(combinations(pairwise_labels, 2))
results = {pair: compute_statistical_significance(groups[pair[0]], groups[pair[1]]) for pair in comparisons}

colors = {i: colours[i] for i in range(len(pairwise_labels))}

plt.figure(figsize=(8, 5), dpi=300)
ax = plt.gca()
ax = sns.boxplot(data=[s.tolist() for s in pairwise_scores], palette=colors)
for patch in ax.artists:
    fc = patch.get_facecolor()
    patch.set_facecolor(plt.colors.to_rgba(fc, 0.3))
sns.swarmplot(data=[s.tolist() for s in pairwise_scores], ax=ax, color='k')
plt.xticks(range(len(pairwise_labels)), pairwise_labels)

annotations = [(pair[0], pair[1], results[pair]['p_value']) for pair in results.keys()]
starbars.draw_annotation(annotations)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
if normalized:
    plt.ylabel('Jnorm', labelpad=12)
else:
    plt.ylabel('J', labelpad=12)
plt.xlabel('Model Pair', labelpad=12)
plt.tight_layout()
if normalized:
    plt.savefig(save_dir + 'subspace_similarity_pairwise_jaccard_comparison_normalized.svg')
else:
    plt.savefig(save_dir + 'subspace_similarity_pairwise_jaccard_comparison.svg')