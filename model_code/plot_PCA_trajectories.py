from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch
import math
import seaborn as sns

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
colours = sns.color_palette("colorblind", 5)

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data = row
    data = np.array(data, dtype=np.float64)
    return data

base_dir = 'saved_outputs/'
nonseq_dir = base_dir+'nonsequential/'
seq_dir = base_dir+'sequential/'
shuffled_dir = base_dir+'shuffled/'

save_dir = base_dir+'plots/'

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3

dirs = [nonseq_dir+'weights/', shuffled_dir+'weights/', seq_dir+'weights/']
labels = ['Non-sequential', 'Shuffled', 'Sequential']

iterations = np.arange(1,501)

# Train plot

#[ref ori, sf, sep]
train_params = [
              [0,0.05,5.0],
              [0,0.05,1.0]
              ]

num_models = len(dirs)
num_steps = 3 # initial, after 1st step of training, after 2nd step of training
num_trials = 30

steps = range(num_steps)
trials = range(1,num_trials+1)

final_weights_together = []
for dir in dirs:
    for step in steps:
        for trial in trials:
            if step == 0:
                data = read_data(dir,'init_fc_weights_ref_'+str(0)+'_sf_'+str(0.05)+'_sep_'+str(5.0)+'_lr_0.0001_trial_'+str(trial)+'.csv')
            elif step == 1:
                data = read_data(dir,'final_fc_weights_ref_'+str(0)+'_sf_'+str(0.05)+'_sep_'+str(5.0)+'_lr_0.0001_trial_'+str(trial)+'.csv')
            else:
                data = data = read_data(dir,'final_fc_weights_ref_'+str(0)+'_sf_'+str(0.05)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv')
            final_weights_together.append(data)

final_weights_together = np.array(final_weights_together)

print("Mean of each feature:", np.mean(np.mean(final_weights_together, axis=0)))
print("Std of each feature:", np.mean(np.std(final_weights_together, axis=0)))

# PCA
averaged_weights = torch.from_numpy(final_weights_together)

plt.figure(figsize=(8,6), dpi=300)

pca = PCA(n_components=2)
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
Xt = pipe.fit_transform(averaged_weights)
Xt = Xt.reshape(num_models, num_steps, num_trials, 2)

mean_pca = np.mean(Xt, axis=2)  # Shape: (num_models, num_steps, 2)
std_pca = np.std(Xt, axis=2)  # Shape: (num_models, num_steps, 2)

for i, (mean, std, model) in enumerate(zip(mean_pca, std_pca, labels)):
    # Plot trajectory
    plt.plot(mean[:, 0], mean[:, 1], lw=5, solid_capstyle='round', label=model, color=colours[i])
    plt.scatter(mean[:, 0], mean[:, 1], s=150, color=colours[i])
    
    # Add arrows between first and second point, and second and third point
    for j in range(2):  # 0->1 and 1->2
        # Calculate midpoint
        mid_x = (mean[j, 0] + mean[j+1, 0]) / 2
        mid_y = (mean[j, 1] + mean[j+1, 1]) / 2
        # Calculate direction from j to j+1
        dx = mean[j+1, 0] - mean[j, 0]
        dy = mean[j+1, 1] - mean[j, 1]
        # Place arrow at midpoint pointing in the direction of travel
        plt.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                    arrowprops=dict(arrowstyle='->', color=colours[i], lw=3))
    
    # Add black star at initial point
    plt.scatter(mean[0, 0], mean[0, 1], s=600, marker='*', color='black', zorder=10)
    
    # Add black circle at end point
    plt.scatter(mean[-1, 0], mean[-1, 1], s=150, marker='o', edgecolors='black', facecolors='black', linewidths=2, zorder=10)

plt.xlabel(r"Readout $\mathbf{w}$ PC 1", labelpad=12)
plt.ylabel(r"Readout $\mathbf{w}$ PC 2", labelpad=12)

plt.legend()

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(save_dir+'PCA_trajectories.svg')

# plt.show()





