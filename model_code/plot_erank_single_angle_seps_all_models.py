import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from matplotlib.lines import Line2D

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import sem, pearsonr


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
colors = [colors[0], colors[1], colors[2], colors[3], colors[4]]

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data.append(row[0])
    data = np.array(data)
    data = data.astype(np.float64)
    return data

# All model names to process
model_names = ['AlexNet_skip', 'EfficientNet_skip', 'GoogLeNet_skip']
label_names = ['AlexNet', 'EfficientNet', 'GoogLeNet']

# Define Paths
base_dir = 'saved_outputs'
single_angle_sep_dirs = [base_dir + f'single_angle_seps_{model_name}/' for model_name in model_names]

save_dir = base_dir + 'plots/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

angle_seps = ['0.5', '1.0', '2.0', '5.0', '10.0']
trials = range(1, 21)

root_dir = 'StimulusImages/SG_train_double_sf/'
test_root_dir = 'StimulusImages/SG_test_double_sf/sep_1.0/'
spatial_freq_train = 0.05
spatial_freq_test = 0.1
ref_angle_train = 0
ref_angle_test = 0

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3
num_neurons = 150

epsilon = 1e-8

# Store results: angle_sep -> model_name -> [20 trials]
all_results = {
    angle_sep: {
        'effective_ranks': [],  # Will store [model_name][trial]
        'test_accuracies': []   # Will store [model_name][trial]
    } for angle_sep in angle_seps
}

# Process each model
for model_name, single_angle_sep_dir in zip(model_names, single_angle_sep_dirs):
    print(f"Processing model: {model_name}")
    activation_dir = base_dir+f'collecting_activations_imagenet_100_{model_name}/'
    
    # Store results for this model
    model_effective_ranks = {angle_sep: [] for angle_sep in angle_seps}
    model_test_accuracies = {angle_sep: [] for angle_sep in angle_seps}
    
    # Compute Effective Rank and Accuracy for Each Angle Separation
    # Activation file is the same for all angle_seps, so load once per trial
    for trial in trials:
        activation_path = activation_dir+'activations/100_imgs_all_activations_no_noise_1000_imagenet_0_sf_0.05_sep_0.5_lr_0.0001_model_trial_'+str(trial)+'_batch_size_1.npy'
        print(f"  Trial {trial}")
        try:
            # mmap_mode='r' avoids loading the full array into RAM; only sliced columns are materialized
            activations_mmap = np.load(activation_path, mmap_mode='r')
        except Exception as e:
            print(f"    Warning: Could not load activations for trial {trial}: {e}")
            continue

        for angle_sep in angle_seps:
            model_dir = single_angle_sep_dir + f'{angle_sep}_angle_sep/'
            try:
                # Load important neuron indices
                source_path = model_dir + 'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+angle_sep+'_lr_0.0001_trial_'+str(trial)+'.csv'
                neuron_indices = np.loadtxt(source_path, delimiter=",", dtype=int)[1]
                sorted_indices = np.argsort(neuron_indices)
                important_neurons = neuron_indices[sorted_indices]

                # Extract and copy only the needed columns (materializes the mmap slice)
                imp_activations = np.array(activations_mmap[:, important_neurons], dtype=np.float64)

                # Normalize activations (zero mean, unit variance across neurons)
                imp_activations = (imp_activations - np.mean(imp_activations, axis=0)) / (np.std(imp_activations, axis=0) + epsilon)

                # Handle any NaN values (happens if std=0 for some neurons)
                imp_activations = np.nan_to_num(imp_activations)

                # Compute SVD
                U, S, Vh = np.linalg.svd(imp_activations, full_matrices=False)

                # Compute Effective Rank
                p = S**2 / np.sum(S**2)
                effective_rank = np.exp(-np.sum(p * np.log(p)))
                model_effective_ranks[angle_sep].append(effective_rank)
                del imp_activations, U, S, Vh, p

                # Load test accuracy
                transfer_data = read_data(model_dir, f'data/transfer_accuracy_ref_{ref_angle_test}_sf_{spatial_freq_test}_sep_'+angle_sep+'_lr_0.0001_trial_'+str(trial)+'.csv')
                test_acc = np.mean(transfer_data)
                model_test_accuracies[angle_sep].append(test_acc)

            except Exception as e:
                print(f"    Warning: Error processing trial {trial} for {model_name} - {angle_sep}: {e}")
                continue

        del activations_mmap
    
    # Convert to arrays and add to all_results
    for angle_sep in angle_seps:
        if len(model_effective_ranks[angle_sep]) > 0:
            eranks_array = np.array(model_effective_ranks[angle_sep])
            accs_array = np.array(model_test_accuracies[angle_sep])
            
            all_results[angle_sep]['effective_ranks'].append(eranks_array)
            all_results[angle_sep]['test_accuracies'].append(accs_array)

# Convert to numpy arrays: shape = (num_models, num_trials)
for angle_sep in angle_seps:
    all_results[angle_sep]['effective_ranks'] = np.array(all_results[angle_sep]['effective_ranks'])
    all_results[angle_sep]['test_accuracies'] = np.array(all_results[angle_sep]['test_accuracies'])

# Compute statistics for plotting
plot_data = {
    angle_sep: {
        'model_means_erank': [],  # Mean across trials for each model
        'model_means_acc': [],
        'model_sems_erank': [],   # SEM across trials for each model
        'model_sems_acc': [],
        'overall_mean_erank': 0,  # Mean across models
        'overall_mean_acc': 0,
        'overall_sem_erank': 0,   # SEM across models
        'overall_sem_acc': 0
    } for angle_sep in angle_seps
}

for angle_sep in angle_seps:
    # Compute mean and SEM across trials for each model
    model_means_erank = np.mean(all_results[angle_sep]['effective_ranks'], axis=1)
    model_means_acc = np.mean(all_results[angle_sep]['test_accuracies'], axis=1)
    model_sems_erank = sem(all_results[angle_sep]['effective_ranks'], axis=1)
    model_sems_acc = sem(all_results[angle_sep]['test_accuracies'], axis=1)
    
    plot_data[angle_sep]['model_means_erank'] = model_means_erank
    plot_data[angle_sep]['model_means_acc'] = model_means_acc
    plot_data[angle_sep]['model_sems_erank'] = model_sems_erank
    plot_data[angle_sep]['model_sems_acc'] = model_sems_acc
    
    # Compute overall mean and SEM across models
    plot_data[angle_sep]['overall_mean_erank'] = np.mean(model_means_erank)
    plot_data[angle_sep]['overall_mean_acc'] = np.mean(model_means_acc)
    plot_data[angle_sep]['overall_sem_erank'] = sem(model_means_erank)
    plot_data[angle_sep]['overall_sem_acc'] = sem(model_means_acc)
    
    print(f"\n{angle_sep} Angle Separation Statistics:")
    print(f"  Overall mean effective rank: {plot_data[angle_sep]['overall_mean_erank']:.3f} ± {plot_data[angle_sep]['overall_sem_erank']:.3f}")
    print(f"  Overall mean test accuracy: {plot_data[angle_sep]['overall_mean_acc']:.3f} ± {plot_data[angle_sep]['overall_sem_acc']:.3f}")

# Create the plot
plt.figure(figsize=(10,8), dpi=300)

shapes = ['s', 'D', "*"]

for angle_sep, color in zip(angle_seps, colors):
    # Plot individual model points with error bars
    for i in range(len(plot_data[angle_sep]['model_means_erank'])):
        marker_shape = shapes[i % len(shapes)]
        plt.scatter(
            plot_data[angle_sep]['model_means_erank'][i],
            plot_data[angle_sep]['model_means_acc'][i],
            s=150, color=color, alpha=0.7, marker=marker_shape)
    
    # Plot overall mean with larger error bars
    plt.errorbar(
        plot_data[angle_sep]['overall_mean_erank'],
        plot_data[angle_sep]['overall_mean_acc'],
        xerr=plot_data[angle_sep]['overall_sem_erank'],
        yerr=plot_data[angle_sep]['overall_sem_acc'],
        capsize=5, lw=5, capthick=5, color=color
    )

    plt.scatter(
        plot_data[angle_sep]['overall_mean_erank'],
        plot_data[angle_sep]['overall_mean_acc'],
        s=150, color=color)#, label=f'{angle_sep}°'
    # )

# Add correlation line for all models and angle separations combined
all_eranks = []
all_accs = []
for angle_sep in angle_seps:
    all_eranks.extend(plot_data[angle_sep]['model_means_erank'])
    all_accs.extend(plot_data[angle_sep]['model_means_acc'])


# Plot a regression line using the overall means for each angle separation
overall_means_erank = [plot_data[angle_sep]['overall_mean_erank'] for angle_sep in angle_seps]
overall_means_acc = [plot_data[angle_sep]['overall_mean_acc'] for angle_sep in angle_seps]

r_val_overall, p_val_overall = pearsonr(overall_means_erank, overall_means_acc)

print(f"Pearson r of Effective Rank vs. Test Accuracy (overall means): {r_val_overall:.4f}, p = {p_val_overall:.4g}")
ax = plt.gca()
sns.regplot(x=overall_means_erank, y=overall_means_acc, scatter=False, ax=ax, color='black', line_kws={'linestyle':'--','linewidth':2}, label=f'Overall means r={r_val_overall:.2f}, p={p_val_overall:.2g}')

# Add colourbar legend for angle separations
norm = plt.Normalize(0.5, 10)
sm = plt.cm.ScalarMappable(cmap=sns.color_palette("GnBu_r", as_cmap=True), norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, ticks=[0.5, 1.0, 2.0, 5.0, 10.0])
cbar.set_label('Angle Separation (°)', rotation=270, labelpad=15)

ax = plt.gca()
plt.ylabel("Transfer Accuracy", labelpad=12)
plt.xlabel("Effective Rank", labelpad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim(0.6, 1.0)

# Model legend uses marker shape only (independent of angle-separation colors).
model_handles = [
    Line2D(
        [0], [0],
        marker=shapes[i % len(shapes)],
        color='black',
        linestyle='None',
        markerfacecolor='white',
        markeredgecolor='black',
        markersize=12,
        label=label_names[i]
    )
    for i in range(num_models)
]
plt.legend(handles=model_handles)
plt.tight_layout()

# Save figure
output_filename = f"effective_rank_all_models_all_angle_seps_{num_neurons}_neurons_per_model.svg"
plt.savefig(os.path.join(save_dir, output_filename))
print(f"\nPlot saved to: {os.path.join(save_dir, output_filename)}")

# plt.show()
