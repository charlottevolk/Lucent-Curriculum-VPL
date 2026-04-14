if __name__ == '__main__':

    import torch
    import numpy as np
    from models.skip_alexnet import AlexNet

    factors = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    base_dir = 'saved_outputs/'
    seq_dir = base_dir + 'sequential_doubled_SF_AlexNet/'
    shuff_dir = base_dir + 'shuffled_doubled_SF_AlexNet/'
    nonseq_dir = base_dir + 'nonsequential_doubled_SF_AlexNet/'
    forced_seq_dirs = [base_dir+'forced_sequential_doubled_SF_AlexNet_factor_'+str(factor)+'/' for factor in factors]
    forced_antiseq_dirs = [base_dir+'forced_antisequential_doubled_SF_AlexNet_factor_'+str(factor)+'/' for factor in factors]

    dirs = [seq_dir, shuff_dir, nonseq_dir]# + forced_seq_dirs + forced_antiseq_dirs # change this if you only want to run some conditions
    num_neurons_range = list(range(10, 251, 10))

    for num_neurons in num_neurons_range:
        trials = range(1,21)
        for trial_num in trials:
                print(f'Num neurons: {num_neurons}, Trial: {trial_num}')
                for save_dir in dirs:
                    sep = 1.0

                    # For loading old models
                    path = save_dir+'models/original_model_0_sf_0.05_sep_'+str(sep)+'_trial_'+str(trial_num)+'.pth'
                    model = AlexNet()
                    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

                    max_values = []
                    indices = []
                    values_with_signs = []

                    # Get all highest weight neurons
                    for i in range(0, num_neurons):
                            value, index = torch.max(torch.abs(model.fc1.weight.data[0]), dim=0)
                            value_with_sign = model.fc1.weight.data[0][index]
                            max_values.append(torch.Tensor.item(value))
                            indices.append(torch.Tensor.item(index))
                            values_with_signs.append(torch.Tensor.item(value_with_sign))
                            model.fc1.weight.data[0][index] = 0.0
                    
                    # Save unshuffled indices
                    to_save = np.row_stack((max_values, indices, values_with_signs))
                    np.savetxt(save_dir+'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', to_save, delimiter=",")