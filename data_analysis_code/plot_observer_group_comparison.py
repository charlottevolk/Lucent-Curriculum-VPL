import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import glob
from scipy.stats import bootstrap, sem
import seaborn as sns
import scipy.stats as stats
import starbars
from natsort import natsorted

SMALLEST_SIZE = 18
SMALL_SIZE = 20
TITLE_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

sns.set_palette("colorblind")

base_dir = 'Data_Final_for_Repository/' # path to data folder - must be inside data_analysis_code folder
NS_folder = base_dir + 'Shuffled_subjects/Original_10/'
S_folder = base_dir + 'Sequential_subjects/'
Non_folder = base_dir + 'Non-sequential_subjects/'

NS_dirs = [dir for dir in os.listdir(NS_folder) if '.' not in dir]
S_dirs = [dir for dir in os.listdir(S_folder) if '.' not in dir]
Non_dirs = [dir for dir in os.listdir(Non_folder) if '.' not in dir]
        
NS_SIs = []
S_SIs = []
Non_SIs = []

NS_tests = []
S_tests = []
Non_tests = []

NS_avgs = []
S_avgs = []
Non_avgs = []

# Shuffled
for d in NS_dirs:

    dir = NS_folder + d + '/'

    usecols = ['Correct', 'Response time (s)', 'Angle Separation (deg)']

    # Get all files in the folder that end with .csv
    data_files = natsorted([dir+file for file in os.listdir(dir) if file.endswith('.csv')])

    data_arr = [pd.read_csv(data_files[0], usecols=usecols), # Day 1 - shuffled
                # pd.read_csv(data_files[1], usecols=usecols),
                # pd.read_csv(data_files[2], usecols=usecols),
                pd.read_csv(data_files[3], usecols=usecols), # Day 4 - shuffled
                pd.read_csv(data_files[4], usecols=usecols), # Day 5 - test hard
                ]

    corr_all = []
    hard_all = []
    easy_all = []

    corr_stes_all = []
    hard_stes_all = []
    easy_stes_all = []

    all_avgs = []
    hard_avgs = []
    easy_avgs = []

    plt.figure(figsize=(25,8))
    for data, in zip(data_arr):
        trials = []

        for i in range(0, len(data)-1, 80):
            trials.append(data[i:i+80])

        seps = [np.array(np.abs(trial['Angle Separation (deg)'])) for trial in trials]
        trials = [np.array(trial['Correct']) for trial in trials]
        trials = [(trial,) for trial in trials]

        easy_trials = []
        hard_trials = []
        for t, s in zip(trials, seps):
            h_trial = []
            e_trial = []
            for trial, sep in zip(t[0], s):
                if sep == 1:
                    h_trial.append(trial)
                else:
                    e_trial.append(trial)
            hard_trials.append(h_trial)
            easy_trials.append(e_trial)

        hard_trials = [(trial,) for trial in hard_trials]
        easy_trials = [(trial,) for trial in easy_trials]

        corr_arr = [np.mean(trial) for trial in trials]
        easy_arr = [np.mean(easy_trial) for easy_trial in easy_trials]
        hard_arr = [np.mean(hard_trial) for hard_trial in hard_trials]

        all_avgs.append(np.mean(corr_arr))
        hard_avgs.append(np.mean(hard_arr))
        easy_avgs.append(np.mean(easy_arr))

    NS_tests.append(all_avgs[-1])

# Sequential
for d in S_dirs:

    dir = S_folder + d + '/'

    usecols = ['Correct', 'Response time (s)', 'Angle Separation (deg)']

    # Get all files in the folder that end with .csv
    data_files = natsorted([dir+file for file in os.listdir(dir) if file.endswith('.csv')])

    data_arr = [pd.read_csv(data_files[0], usecols=usecols), # Day 1 - shuffled
                # pd.read_csv(data_files[1], usecols=usecols),
                # pd.read_csv(data_files[2], usecols=usecols),
                pd.read_csv(data_files[3], usecols=usecols), # Day 4 - shuffled
                pd.read_csv(data_files[4], usecols=usecols), # Day 5 - test hard
                ]

    corr_all = []
    hard_all = []
    easy_all = []

    corr_stes_all = []
    hard_stes_all = []
    easy_stes_all = []

    all_avgs = []
    hard_avgs = []
    easy_avgs = []

    #plt.figure(figsize=(25,8))
    for data, in zip(data_arr):
        trials = []

        for i in range(0, len(data)-1, 80):
            trials.append(data[i:i+80])

        seps = [np.array(np.abs(trial['Angle Separation (deg)'])) for trial in trials]
        trials = [np.array(trial['Correct']) for trial in trials]
        trials = [(trial,) for trial in trials]

        corr_arr = [np.mean(trial) for trial in trials]

        all_avgs.append(np.mean(corr_arr))

    S_tests.append(all_avgs[-1])

# Non-sequential
for d in Non_dirs:

    dir = Non_folder + d + '/'

    usecols = ['Correct', 'Response time (s)', 'Angle Separation (deg)']

    # Get all files in the folder that end with .csv
    data_files = natsorted([dir+file for file in os.listdir(dir) if file.endswith('.csv')])

    data_arr = [pd.read_csv(data_files[0], usecols=usecols), # Day 1 - shuffled
                # pd.read_csv(data_files[1], usecols=usecols),
                # pd.read_csv(data_files[2], usecols=usecols),
                pd.read_csv(data_files[3], usecols=usecols), # Day 4 - shuffled
                pd.read_csv(data_files[4], usecols=usecols), # Day 5 - test hard
                ]

    corr_all = []
    hard_all = []
    easy_all = []

    corr_stes_all = []
    hard_stes_all = []
    easy_stes_all = []

    all_avgs = []
    hard_avgs = []
    easy_avgs = []

    plt.figure(figsize=(25,8))
    for data, in zip(data_arr):
        trials = []

        for i in range(0, len(data)-1, 80):
            trials.append(data[i:i+80])

        seps = [np.array(np.abs(trial['Angle Separation (deg)'])) for trial in trials]
        trials = [np.array(trial['Correct']) for trial in trials]
        trials = [(trial,) for trial in trials]

        corr_arr = [np.mean(trial) for trial in trials]

        all_avgs.append(np.mean(corr_arr))

    Non_tests.append(all_avgs[-1])

# Plot observer groups comparison with statistical testing

def check_normality(data):
    """Perform Shapiro-Wilk test for normality."""
    _, p_value = stats.shapiro(data)
    return p_value > 0.05  # If p > 0.05, assume normal distribution

def compute_statistical_significance(group1, group2):
    """Computes statistical significance between two groups."""
    normal1 = check_normality(group1)
    normal2 = check_normality(group2)

    if normal1 and normal2:
        stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test
        test_used = "t-test"
    else:
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')  # Mann-Whitney U test
        test_used = "Mann-Whitney U test"

    return {"p_value": p_value, "test": test_used}

# Define pairs to compare
groups = {
    "Non_tests": Non_tests,
    "NS_tests": NS_tests,
    "S_tests": S_tests,
}
comparisons = [
    ("Non_tests", "NS_tests"),
    ("Non_tests", "S_tests"),
    ("NS_tests", "S_tests"),
]

# Compute significance for each comparison
results = {pair: compute_statistical_significance(groups[pair[0]], groups[pair[1]]) for pair in comparisons}
print(results)

# Plotting
plt.figure(figsize=(8,5), dpi=300)
ax=plt.gca()
ax = sns.boxplot(data=[Non_tests, NS_tests, S_tests], palette="colorblind")
for patch in ax.artists:
    fc = patch.get_facecolor()
    patch.set_facecolor(plt.colors.to_rgba(fc, 0.3))
sns.swarmplot(data=[Non_tests, NS_tests, S_tests], ax=ax, color='k')
plt.xticks([0, 1, 2], ["Non_tests", "NS_tests", "S_tests"])

annotations = []
for pair in results.keys():
    annotation = (pair[0], pair[1], results[pair]['p_value'])
    print(type(annotation[-1]))
    annotations.append(annotation)
# Add statistical significance bars using starbars
starbars.draw_annotation(annotations)

plt.xticks([0, 1, 2], ["Non-sequential", "Shuffled", "Sequential"])

# Add significance annotations
x_positions = {name: i for i, name in zip(groups.keys(), range(len(groups) - 1))}

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylabel('Transfer Accuracy', labelpad=12)
plt.xlabel('Condition', labelpad=12)
plt.tight_layout()
plt.savefig('human_observer_group_comparison.png')
# plt.show()