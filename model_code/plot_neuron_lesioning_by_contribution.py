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

sns.set_palette("colorblind", 3)

base_dir = 'saved_outputs/'
nonseq_dir = base_dir+'nonsequential_doubled_SF_AlexNet/'
seq_dir = base_dir+'sequential_doubled_SF_AlexNet/'
shuffled_dir = base_dir+'shuffled_doubled_SF_AlexNet/'

save_dir = base_dir + 'plots/'

trials = range(1,21)
noise_sd = 0.02
confidence_sd = 0.3

model_labels = ['Non-sequential', 'Shuffled', 'Sequential']
model_dirs = [nonseq_dir, shuffled_dir, seq_dir]

transfer_condition = 'SF' # or 'ref_ori'

if transfer_condition == 'ref_ori':
        test_sf = 0.05
        test_ref_ori = 15
    elif transfer_condition == 'SF':
        test_sf = 0.1
        test_ref_ori = 0

fig, ax = plt.subplots(figsize=(7,5), dpi=300)
plt.xlabel("N", labelpad=12)
plt.ylabel(r"$\Delta$Acc./N", labelpad=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

num_neurons = 250
neurons_lesioned = list(range(10, num_neurons+1, 10))

for model_dir, model_name in zip(model_dirs, model_labels):
    transfer_accuracies = []
    for trial_num in trials:
        base_acc = np.mean(read_data(model_dir, 'data/lesioning_0_neurons_transfer_accuracy_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_1.0_lr_0.0001_trial_'+str(trial_num)+'.csv'))
        print(base_acc)
        t_acc_trial = []
        base_file_lesioning = np.loadtxt(model_dir + 'data/lesioning_by_contribution_all_neurons_dictionary_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_1.0_lr_0.0001_trial_'+str(trial_num)+'.csv', delimiter=',', dtype='float')
        indices = []
        losses = []
        accs = []
        for entry in base_file_lesioning:
            indices.append(int(entry[0]))
            losses.append(float(entry[1]))
            accs.append(float(entry[2]))
        # Array is sorted by accuracy (ascending), so lowest accuracy (most important) neurons are at the start
        for neurons in neurons_lesioned:
            t_acc = accs[:neurons]
            t_acc_trial.append((base_acc - np.mean(t_acc))/neurons)
        transfer_accuracies.append(t_acc_trial)
    acc_mean = np.mean(transfer_accuracies, axis=0)
    acc_std = np.std(transfer_accuracies, axis=0)

    plt.plot(neurons_lesioned, acc_mean, lw=5, label=model_name, color=sns.color_palette("colorblind")[model_labels.index(model_name)])
    plt.scatter(neurons_lesioned, acc_mean, s=150, color=sns.color_palette("colorblind")[model_labels.index(model_name)])
    plt.fill_between(neurons_lesioned, acc_mean-acc_std, acc_mean+acc_std, alpha=0.3, color=sns.color_palette("colorblind")[model_labels.index(model_name)])

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir+'neuron_lesioning_by_contribution_plot.svg', dpi=300)