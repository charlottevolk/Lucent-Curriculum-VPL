import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import sem, linregress, t

deg = u'\N{DEGREE SIGN}'

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
plt.rc('legend', fontsize=SMALLEST_SIZE) # legend fontsize

sns.set_palette("colorblind")

# Load JSON files
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

rt_easy = load_json('rt_and_corr_easy_trials_with_nans_to_equalize_length.json')
rt_hard = load_json('rt_and_corr_hard_trials_with_nans_to_equalize_length.json')

easy_all_rt = []
hard_all_rt = []

easy_all_acc = []
hard_all_acc = []

transfer_accs_all = []
names_all = []

transfer_accs = load_json('transfer_accs.json')
# Get all participant names from json file
all_participants = list(transfer_accs.keys())

for participant in all_participants:
    means_dict = {participant}
    transfer_acc = transfer_accs[participant]
    transfer_accs_all.append(transfer_acc)
    names_all.append(participant)

    def get_accs(names):
        data_arr_easy = []
        data_arr_hard = []
        rt_arr_easy = []
        rt_arr_hard = []
        for name in names:
            easy_arr = []
            hard_arr = []
            easy_arr_rt = []
            hard_arr_rt = []
            for rt, acc in rt_easy[name]:
                # Remove non-responses
                if acc == 0 and rt > 10:
                    easy_arr.append(np.nan)
                    easy_arr_rt.append(np.nan)
                else:
                    easy_arr.append(acc)
                    easy_arr_rt.append(rt)
            for rt, acc in rt_hard[name]:
                if acc == 0 and rt > 10:
                    hard_arr.append(np.nan)
                    hard_arr_rt.append(np.nan)
                else:
                    hard_arr.append(acc)
                    hard_arr_rt.append(rt)
            data_arr_easy.append(easy_arr)
            data_arr_hard.append(hard_arr)
            rt_arr_easy.append(easy_arr_rt)
            rt_arr_hard.append(hard_arr_rt)
        return data_arr_easy, data_arr_hard, rt_arr_easy, rt_arr_hard

    easy_acc, hard_acc, easy_rt, hard_rt = get_accs(means_dict)

    easy_all_rt.append(easy_rt[0])
    hard_all_rt.append(hard_rt[0])
    easy_all_acc.append(easy_acc[0])
    hard_all_acc.append(hard_acc[0])

# Collect into groups of 80 trials (1 block)
def blockwise_mean_and_ci(data, block_size=80, ci=0.95):
    n_participants, n_trials = data.shape
    n_blocks = n_trials // block_size

    reshaped = data[:, :n_blocks * block_size].reshape(n_participants, n_blocks, block_size)
    block_means = np.nanmean(reshaped, axis=2)
    block_mean = np.nanmean(block_means, axis=0)
    block_sem = sem(block_means, axis=0, nan_policy='omit')
    n = np.sum(~np.isnan(block_means), axis=0)
    h = block_sem * t.ppf((1 + ci) / 2., n-1)  # margin of error for CI

    ci_lower = block_mean - h
    ci_upper = block_mean + h
    return block_mean, ci_lower, ci_upper

easy_acc, easy_acc_ci_lower, easy_acc_ci_upper = blockwise_mean_and_ci(np.array(easy_all_acc))
hard_acc, hard_acc_ci_lower, hard_acc_ci_upper = blockwise_mean_and_ci(np.array(hard_all_acc))

x_vals = np.arange(1, len(easy_acc)+1)

# Plot accuracy split by condition
plt.figure(figsize=(5,5), dpi=300)
plt.plot(x_vals[:16], easy_acc[:16], label='5.0' + deg, lw=5, solid_capstyle='round', color='tab:blue')
plt.plot(x_vals[:16], hard_acc[:16], label='1.0' + deg, lw=5, solid_capstyle='round', color='tab:orange')

plt.fill_between(x_vals[:16], easy_acc_ci_lower[:16], easy_acc_ci_upper[:16], alpha=0.2, color='tab:blue')
plt.fill_between(x_vals[:16], hard_acc_ci_lower[:16], hard_acc_ci_upper[:16], alpha=0.2, color='tab:orange')

plt.axhline(0.9, color='black', linestyle='--', linewidth=1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Block', labelpad=12)
plt.ylabel('Accuracy', labelpad=12)
plt.tight_layout()
plt.legend()
plt.savefig('observer_accuracy_split_by_condition.svg')

# Plot correlations
nums = np.int32(np.unique((np.rint(np.logspace(base=2, start=2, stop=7))))) # sample n logarithmically
mask = (nums > 4) & (nums <= 100) # Make sure no repeated values & n not too small
nums = nums[mask]

r_vals = []

for num in nums:

    def implicit_curriculum_metric_non_nan(data, block_size=num, num_trials=1280):
        n_participants, _ = data.shape
        n_blocks = num_trials // block_size
        block_means = np.full((n_participants, n_blocks), np.nan)

        for p in range(n_participants):
            valid_trials = data[p][~np.isnan(data[p])]
            selected = valid_trials[:n_blocks * block_size]

            if len(selected) < n_blocks * block_size:
                # pad with NaNs if not enough trials
                padded = np.full(n_blocks * block_size, np.nan)
                padded[:len(selected)] = selected
            else:
                padded = selected

            reshaped = padded.reshape(n_blocks, block_size)
            block_means[p] = np.nanmean(reshaped, axis=1)

        return block_means

    easy_all_rt = np.array(easy_all_acc)
    hard_all_rt = np.array(hard_all_acc)

    easy_curriculum_metric = implicit_curriculum_metric_non_nan(easy_all_rt)
    hard_curriculum_metric = implicit_curriculum_metric_non_nan(hard_all_rt)

    curriculums = []
    transfer_accuracies = []

    # Plot correlation for n with highest r-value
    if num == 13:
        plt.figure(figsize=(5,5), dpi=300)
    for easy_metric, hard_metric, transfer_acc, name in zip(easy_curriculum_metric, hard_curriculum_metric, transfer_accs_all, names_all):
        curriculum = easy_metric[0] - hard_metric[0]
        curriculums.append(curriculum)
        transfer_accuracies.append(transfer_acc)

    curriculums = np.array(curriculums)
    transfer_accuracies = np.array(transfer_accuracies)

    if num == 13:
        sns.regplot(x=curriculums, y=transfer_accuracies, scatter=False, ci=95, color='black', line_kws={'linestyle': '--'})
    slope, intercept, r_value, p_value, _ = linregress(curriculums, transfer_accuracies)
    if num == 13:
        plt.plot([], [], color='black', linestyle='--', label=f'$r$ = {r_value:.2f}, p = {p_value:.3f}')
        plt.scatter(curriculums, transfer_accuracies, s=150, color='tab:blue')
    r_vals.append(r_value)

    if num == 13:
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Curriculum metric', labelpad=12)
        plt.ylabel('Transfer Accuracy', labelpad=12)
        plt.yticks([0.5, 0.6, 0.7])
        plt.tight_layout()
        plt.legend()
        plt.savefig('accuracy_curriculum_metric_correlation_n=13.svg')

plt.figure(figsize=(5,5), dpi=300)
plt.scatter(nums, r_vals, s=150)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('$n$', labelpad=12)
plt.ylabel('r', labelpad=12)
plt.tight_layout()
plt.savefig('all_r_values_from_correlations.svg')
