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
colours = sns.color_palette("colorblind")

use_active_gradients_only = True # If True, only plot gradients when each angle was active; if False, plot all saved gradients

base_dir = 'saved_outputs/'
grad_dir = base_dir + 'shuffled_collecting_gradients/'
save_dir = base_dir + 'plots/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

num_steps = 100 # Number of gradient steps to plot

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3

num_trials = 20

grads_all_5 = []
grads_all_1 = []

plt.figure(figsize=(7, 5), dpi=300)

for trial in range(1, num_trials+1):
    print('Trial: ', trial)

    grads_5 = []
    grads_1 = []

    for j in range(num_steps):
        if use_active_gradients_only:
            # Load which separation was ACTIVE at this step
            active_sep = torch.load(grad_dir + f'data/seps_iteration_{j}_trial_{trial}.pt', weights_only=False)
            active_sep = float(np.asarray(active_sep).flatten()[0])
            
            # Only load and use the gradients from the ACTIVE separation
            if active_sep == 5.0:
                loaded_grads = torch.load(grad_dir + f'data/gradients_per_sample_iteration_{j}_sep_5.0_trial_{trial}.pt', map_location=torch.device('cpu'))
                mean_grads_by_layer = []
                for i in range(len(loaded_grads)):
                    mean_grads_by_layer.append(torch.mean(torch.flatten(torch.abs(loaded_grads[i]), 1)))
                mean_grads_by_layer = torch.stack(mean_grads_by_layer)
                mean_grads_by_layer = torch.mean(mean_grads_by_layer, dim=0)
                grads_5.append(mean_grads_by_layer.item())
                # Repeat last value for inactive angle
                if len(grads_1) > 0:
                    grads_1.append(grads_1[-1])
                else:
                    grads_1.append(0.0)
                    
            elif active_sep == 1.0:
                loaded_grads = torch.load(grad_dir + f'data/gradients_per_sample_iteration_{j}_sep_1.0_trial_{trial}.pt', map_location=torch.device('cpu'))
                mean_grads_by_layer = []
                for i in range(len(loaded_grads)):
                    mean_grads_by_layer.append(torch.mean(torch.flatten(torch.abs(loaded_grads[i]), 1)))
                mean_grads_by_layer = torch.stack(mean_grads_by_layer)
                mean_grads_by_layer = torch.mean(mean_grads_by_layer, dim=0)
                grads_1.append(mean_grads_by_layer.item())
                # Repeat last value for inactive angle
                if len(grads_5) > 0:
                    grads_5.append(grads_5[-1])
                else:
                    grads_5.append(0.0)
        
        else:
            # Load both gradient files at every step and use them regardless of which separation was active
            loaded_grads_5 = torch.load(grad_dir + f'data/gradients_per_sample_iteration_{j}_sep_5.0_trial_{trial}.pt', map_location=torch.device('cpu'))
            loaded_grads_1 = torch.load(grad_dir + f'data/gradients_per_sample_iteration_{j}_sep_1.0_trial_{trial}.pt', map_location=torch.device('cpu'))
            
            mean_grads_by_layer_5 = []
            for i in range(len(loaded_grads_5)):
                mean_grads_by_layer_5.append(torch.mean(torch.flatten(torch.abs(loaded_grads_5[i]), 1)))
            mean_grads_by_layer_5 = torch.stack(mean_grads_by_layer_5)
            mean_grads_by_layer_5 = torch.mean(mean_grads_by_layer_5, dim=0)
            grads_5.append(mean_grads_by_layer_5.item())

            mean_grads_by_layer_1 = []
            for i in range(len(loaded_grads_1)):
                mean_grads_by_layer_1.append(torch.mean(torch.flatten(torch.abs(loaded_grads_1[i]), 1)))
            mean_grads_by_layer_1 = torch.stack(mean_grads_by_layer_1)
            mean_grads_by_layer_1 = torch.mean(mean_grads_by_layer_1, dim=0)
            grads_1.append(mean_grads_by_layer_1.item())

    grads_all_5.append(grads_5)
    grads_all_1.append(grads_1)

grads_5 = np.array(grads_all_5)
grads_1 = np.array(grads_all_1)

mean_5 = np.mean(grads_5, axis=0)
mean_1 = np.mean(grads_1, axis=0)

# compute confidence intervals
# 95% confidence intervals for grads_5 and grads_1 on axis 0
n_5 = grads_5.shape[0]
n_1 = grads_1.shape[0]
sem_5 = sem(grads_5, axis=0)
sem_1 = sem(grads_1, axis=0)
ci_5 = mean_5 + 1.96 * sem_5, mean_5 - 1.96 * sem_5
ci_1 = mean_1 + 1.96 * sem_1, mean_1 - 1.96 * sem_1

# Plot

x = np.arange(num_steps)

plt.plot(x, mean_5, label='5.0'+deg, lw=5, solid_capstyle='round', color=colours[0])
plt.fill_between(x, ci_5[0], ci_5[1], alpha=0.3, interpolate=True, color=colours[0])
plt.plot(x, mean_1, label='1.0'+deg, lw=5, solid_capstyle='round', color=colours[1])
plt.fill_between(x, ci_1[0], ci_1[1], alpha=0.3, interpolate=True, color=colours[1])

# plt.axhline(y=0.001, color='black', linestyle='--', linewidth=1)
plt.axvline(x=25, color='black', linestyle='--', linewidth=1)
plt.xlabel('Steps', labelpad=12)
plt.xticks(ticks=[0,25,99], labels=[0, 25, num_steps])  
plt.ylabel(r'$|\nabla\mathbf{w}|$', color='black', labelpad=12)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right')
plt.tight_layout()

if use_active_gradients_only:
    plt.savefig(save_dir+'shuffled_active_gradients.svg')
else:
    plt.savefig(save_dir+'shuffled_gradients.svg')


