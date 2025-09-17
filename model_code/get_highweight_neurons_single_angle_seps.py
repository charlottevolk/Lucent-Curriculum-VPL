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
        model_names = ['0.5', '1.0', '2.0', '5.0', '10.0']
        single_angle_sep_dir = base_dir+'skip_model_full_training_non_sequential_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update/'
        for trial_num in trials:
                for sep in model_names:
                    # For loading old models
                    path = single_angle_sep_dir+'models/original_model_0_sf_0.05_sep_'+str(sep)+'_trial_'+str(trial_num)+'.pth'
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