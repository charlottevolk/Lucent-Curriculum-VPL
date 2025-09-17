if __name__ == '__main__':

    import torch
    import numpy as np
    from skip_alexnet import AlexNet

    stimulus_noise_sd = 0.02
    confidence_noise_sd = 0.3
    num_steps = 25
    num_neurons_range = list(range(10, 1001, 10))
    for num_neurons in num_neurons_range:
        trials = range(1,21)
        base_dir = 'saved_outputs/'
        model_names = ['seq', 'shuff', 'nonseq']
        dirs = [base_dir+'skip_'+model+'_5.0_1.0_noise_sd_0.02_added_confidence_noise_sd_0.3_single_sample_update/' for model in model_names]
        seq_dirs = [base_dir+'skip_shuff_5.0_1.0_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update_forced_sequential_with_weighted_samples_beginning_'+str(num_steps)+'_steps_factor_'+str(factor)+'/' for factor in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
        nonseq_dirs = [base_dir+'skip_shuff_5.0_1.0_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update_forced_antisequential_with_weighted_samples_beginning_'+str(num_steps)+'_steps_factor_'+str(factor)+'/' for factor in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
        for trial_num in trials:
                for save_dir in dirs:
                    sep = 1.0

                    # For loading old models
                    path = save_dir+'models/original_model_0_sf_0.05_sep_'+str(sep)+'_trial_'+str(trial_num)+'.pth'
                    alexnet = AlexNet()
                    alexnet.load_state_dict(torch.load(path))

                    max_values = []
                    indices = []
                    values_with_signs = []

                    # Get all highest weight neurons
                    for i in range(0, num_neurons):
                        value, index = torch.max(torch.abs(alexnet.fc1.weight.data[0]), dim=0)
                        value_with_sign = alexnet.fc1.weight.data[0][index]
                        max_values.append(torch.Tensor.item(value))
                        indices.append(torch.Tensor.item(index))
                        values_with_signs.append(torch.Tensor.item(value_with_sign))
                        # print("Max val: ", value, " Index: ", index, "With sign: ", value_with_sign)
                        alexnet.fc1.weight.data[0][index] = 0.0
                    
                    # Save unshuffled indices
                    to_save = np.row_stack((max_values, indices, values_with_signs))
                    np.savetxt(save_dir+'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', to_save, delimiter=",")