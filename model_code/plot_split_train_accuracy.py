import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib as mpl
from cycler import cycler
from matplotlib.cm import get_cmap
import seaborn as sns
import os
import torch
from scipy.stats import sem

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

colours = sns.color_palette("colorblind")

base_dir = 'saved_outputs/shuffled/'
save_dir = base_dir + 'plots/'  

deg = u'\N{DEGREE SIGN}'

plot_SEM = True

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data.append(row[0])
    data = np.array(data)
    data = data.astype(np.float64)
    return data

iterations = np.arange(0,1000)#, 50)

# Training learning curve plot
spatial_freq_train = 0.05
spatial_freq_test = 0.1
ref_angle_train = 0
ref_angle_test = 0

trials = range(1,21)

to_plot = ['train', 'transfer']

train_acc_all_5 = []
train_acc_all_1 = []

for trial in trials:
    print('Trial: ', trial)

    train_acc_5 = []
    train_acc_1 = []

    # Load training accuracy data
    train_data_first = read_data(base_dir, 'data/train_accuracy_ref_'+str(ref_angle_train)+'_sf_'+str(spatial_freq_train)+'_sep_'+str(5.0)+'_lr_0.0001_trial_'+str(trial)+'.csv')
    train_data_second = read_data(base_dir, 'data/train_accuracy_ref_'+str(ref_angle_train)+'_sf_'+str(spatial_freq_train)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv')
    
    # Concatenate both halves
    train_data = np.concatenate((train_data_first, train_data_second), axis=0)

    for iter in iterations:
        # Load both gradient files at every step (original behavior)
        sep = torch.load(base_dir + f'data/seps_iteration_{iter}_trial_{trial}.pt', weights_only=False)
        if float(sep) == 5.0:
            train_acc_5.append(train_data[iter])
            if len(train_acc_1) > 0:
                train_acc_1.append(train_acc_1[-1])
            else:
                train_acc_1.append(0.0)
        else:
            train_acc_1.append(train_data[iter])
            if len(train_acc_5) > 0:
                train_acc_5.append(train_acc_5[-1])
            else:
                train_acc_5.append(0.0)

    train_acc_all_5.append(train_acc_5)
    train_acc_all_1.append(train_acc_1)

train_acc_5 = np.array(train_acc_all_5)
train_acc_1 = np.array(train_acc_all_1)

print(train_acc_5.shape, train_acc_1.shape)

mean_5 = np.mean(train_acc_5, axis=0)
mean_1 = np.mean(train_acc_1, axis=0)

# Split into blocks of 10 steps each for smoothing out the curves
mean_split_5 = []
mean_split_1 = []
block_size = 1
for i in range(0, len(mean_5), block_size):
    block_5 = mean_5[i:i+block_size]
    block_1 = mean_1[i:i+block_size]
    mean_split_5.append(np.mean(block_5))
    mean_split_1.append(np.mean(block_1))

mean_5 = np.array(mean_split_5)
mean_1 = np.array(mean_split_1)

# Apply same block averaging to all trials for SEM computation
train_acc_split_5 = []
train_acc_split_1 = []
for trial_idx in range(train_acc_5.shape[0]):
    trial_split_5 = []
    trial_split_1 = []
    for i in range(0, train_acc_5.shape[1], block_size):
        block_5 = train_acc_5[trial_idx, i:i+block_size]
        block_1 = train_acc_1[trial_idx, i:i+block_size]
        trial_split_5.append(np.mean(block_5))
        trial_split_1.append(np.mean(block_1))
    train_acc_split_5.append(trial_split_5)
    train_acc_split_1.append(trial_split_1)

train_acc_split_5 = np.array(train_acc_split_5)
train_acc_split_1 = np.array(train_acc_split_1)

# compute confidence intervals
# 95% confidence intervals for grads_5 and grads_1 on axis 0
n_5 = train_acc_split_5.shape[0]
n_1 = train_acc_split_1.shape[0]
sem_5 = sem(train_acc_split_5, axis=0)
sem_1 = sem(train_acc_split_1, axis=0)
ci_5 = mean_5 + 1.96 * sem_5, mean_5 - 1.96 * sem_5
ci_1 = mean_1 + 1.96 * sem_1, mean_1 - 1.96 * sem_1

# Plot

x = range(block_size, 1001, block_size)

end_step = 20

plt.plot(x[:end_step], mean_5[:end_step], label='5.0'+deg, lw=5, solid_capstyle='round', color=colours[0])
plt.fill_between(x[:end_step], ci_5[0][:end_step], ci_5[1][:end_step], alpha=0.3, interpolate=True, color=colours[0])
plt.plot(x[:end_step], mean_1[:end_step], label='1.0'+deg, lw=5, solid_capstyle='round', color=colours[1])
plt.fill_between(x[:end_step], ci_1[0][:end_step], ci_1[1][:end_step], alpha=0.3, interpolate=True, color=colours[1])

# plt.axhline(y=0.001, color='black', linestyle='--', linewidth=1)
# plt.axvline(x=25, color='black', linestyle='--', linewidth=1)
plt.xlabel('Steps', labelpad=12)
# plt.xticks(ticks=[0,25,99], labels=[0, 25, num_steps])  
plt.ylabel('Accuracy', color='black', labelpad=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right')
plt.tight_layout()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.legend()

plt.savefig(save_dir+f'train_accuracy_split_learning_curve_{end_step}_steps.svg')
# plt.show()