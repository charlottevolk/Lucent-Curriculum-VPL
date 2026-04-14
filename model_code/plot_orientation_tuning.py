import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as stats
from scipy.stats import sem

SMALLEST_SIZE = 12
SMALL_SIZE = 20
TITLE_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 12

plt.rcParams['svg.fonttype'] = 'none'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize

base_dir = 'saved_outputs/'

activations_dir = base_dir + 'orientation_tuning_alexnet/'
plot_save_dir = base_dir + 'plots/'
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

num_neurons_range = [150]  # number of top neurons to include in the plot
model_names = ['sequential', 'shuffled', 'nonsequential']
for col_index, num_neurons in enumerate(num_neurons_range):
    # print(f'Processing top {num_neurons} neurons for each model...')
    mean_curves_by_model = {}
    for model_name in model_names:
        print(f'Processing model: {model_name} with top {num_neurons} neurons...')

        # Reference orientation used during collection
        ref_ori = 0

        # Separations used during collection
        seps = [0.5] + [float(x) for x in range(1, 181)]

        # Spatial frequency
        sf = 0.05

        # Trials
        trials = range(1, 21)

        # CW stimulus at separation S is at orientation: ref_ori + S
        # CCW stimulus at separation S is at orientation: ref_ori - S
        orientations_CW = [ref_ori + s for s in seps]   # positive side
        orientations_CCW = [ref_ori - s for s in seps]   # negative side

        # Combined and sorted
        all_orientations = sorted(set(orientations_CCW + orientations_CW))

        neuron_responses = {ori: [] for ori in all_orientations}  # ori -> list of activation values per neuron

        # Top-neuron files are trial-dependent but not sep-dependent, so load them once.
        neuron_indices_by_trial = {}
        for trial in trials:
            neuron_file = base_dir + f'{model_name}_doubled_SF_AlexNet/data/max_abs_neurons_150_sep_1.0_lr_0.0001_trial_{trial}.csv'
            neuron_data = np.loadtxt(neuron_file, delimiter=",", dtype=float)
            neuron_indices = neuron_data[1].astype(int)  # row 1 = indices
            neuron_indices_by_trial[trial] = neuron_indices[:num_neurons]
            del neuron_data

        for sep in seps:
            ori_cw = ref_ori + sep
            ori_ccw = ref_ori - sep

            for trial in trials:
                neuron_indices = neuron_indices_by_trial[trial]

                # Load CW activations
                cw_path = activations_dir + f'activations_all_CW_sep_{sep}_sf_{sf}_trial_{trial}.npy'
                if os.path.exists(cw_path):
                    cw_data = np.load(cw_path, allow_pickle=True)
                    if len(cw_data) > 0:
                        # cw_data is a list of concatenated activation vectors
                        cw_matrix = np.array(cw_data)  # [n_samples, n_total_neurons]
                        del cw_data
                        # Extract the top 150 neurons and average across samples
                        cw_neuron_acts = cw_matrix[:, neuron_indices][:, :num_neurons]  # [n_samples, num_neurons]
                        del cw_matrix
                        neuron_responses[ori_cw].append(cw_neuron_acts)

                # Load CCW activations
                ccw_path = activations_dir + f'activations_all_CCW_sep_{sep}_sf_{sf}_trial_{trial}.npy'
                if os.path.exists(ccw_path):
                    ccw_data = np.load(ccw_path, allow_pickle=True)
                    if len(ccw_data) > 0:
                        ccw_matrix = np.array(ccw_data)  # [n_samples, n_total_neurons]
                        del ccw_data
                        ccw_neuron_acts = ccw_matrix[:, neuron_indices][:, :num_neurons]  # [n_samples, num_neurons]
                        del ccw_matrix
                        neuron_responses[ori_ccw].append(ccw_neuron_acts)

        # Average across all trials and samples for each orientation
        # Result: mean_responses[ori] = array of shape [num_neurons]
        mean_responses = {}
        for ori in all_orientations:
            if len(neuron_responses[ori]) > 0:
                # Concatenate all samples across all trials: [total_samples, num_neurons]
                all_samples = np.concatenate(neuron_responses[ori], axis=0)
                mean_responses[ori] = np.mean(all_samples, axis=0)  # [num_neurons]
                del all_samples
            else:
                mean_responses[ori] = np.zeros(num_neurons)

        del neuron_responses

        # Build tuning curve matrix: [num_neurons, num_orientations]
        tuning_curves = np.array([mean_responses[ori] for ori in all_orientations]).T  # [num_neurons, num_orientations]
        del mean_responses

        print(f'Tuning curve matrix shape: {tuning_curves.shape}')
        print(f'  {tuning_curves.shape[0]} neurons x {tuning_curves.shape[1]} orientations')

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot each neuron's tuning curve with transparency

        # Calculate OSI for each neuron
        OSI_list = []
        for i in range(num_neurons):
            # Find preferred orientation (max mean activation)
            pref_idx = np.argmax(tuning_curves[i, :])
            R_pref = tuning_curves[i, pref_idx]
            # Find orthogonal orientation (preferred + 90)
            pref_ori = all_orientations[pref_idx]
            # Wrap orientation to [-180, 180]
            orth_ori = pref_ori + 90
            if orth_ori > 180:
                orth_ori -= 360
            elif orth_ori < -180:
                orth_ori += 360
            # Find closest orientation index to orth_ori
            orth_idx = np.argmin([abs(o - orth_ori) for o in all_orientations])
            R_orth = tuning_curves[i, orth_idx]
            # Compute OSI
            OSI = (R_pref - R_orth) / (R_pref + R_orth) if (R_pref + R_orth) != 0 else 0
            OSI_list.append(OSI)
            ax.plot(all_orientations, tuning_curves[i, :], color='tab:blue', alpha=0.1, linewidth=1)

        # # # Plot the population average at full opacity
        mean_curve = np.mean(tuning_curves, axis=0)
        mean_curves_by_model[model_name] = mean_curve
        ax.plot(all_orientations, mean_curve, color='tab:blue', alpha=1.0, linewidth=5, label='Population mean')

        # Store OSI values for each model for later comparison
        if 'OSI_across_models' not in locals():
            OSI_across_models = {}
        OSI_across_models[model_name + f'_top{num_neurons}'] = OSI_list
        del tuning_curves, neuron_indices_by_trial

        print(f'OSI values for each neuron: {OSI_list}')

        ax.set_xlabel('Stimulus Orientation (°)')
        ax.set_ylabel('Mean Activation')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 5.5)
        plt.yticks([0,1,2,3,4,5])
        # ax.legend()
        ax.axvline(x=ref_ori, color='gray', linestyle='--', alpha=0.5, label='Reference')
        plt.tight_layout()

        plt.savefig(plot_save_dir + f'orientation_tuning_curves_top{num_neurons}_{model_name}_redone_colours_mean_curve_sem_only_same_ax_limits.svg', dpi=300)
        print(f'Saved plot to {plot_save_dir}')
        # plt.show()

    all_vals = [OSI_across_models[m + f'_top{num_neurons}'] for m in model_names]
    x_labels = ['Sequential', 'Shuffled', 'Non-sequential']

    mean_vals = [np.mean(vals) for vals in all_vals]
    sem_vals = [sem(vals) for vals in all_vals]

    plt.figure(figsize=(8, 5), dpi=300)
    plt.scatter(x_labels, mean_vals, color='tab:blue', s=150)
    plt.errorbar(x_labels, mean_vals, yerr=sem_vals, capsize=3, color='tab:blue')
    plt.xticks([0, 1, 2], x_labels)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylabel('Mean OSI', labelpad=12)
    plt.xlabel('Condition', labelpad=12)
    plt.tight_layout()

    plt.savefig(plot_save_dir + f'OSI_avg_comparison_across_models_{num_neurons}_neurons.svg', dpi=300)
    # plt.show()
