if __name__ == '__main__':

    import torch
    import numpy as np
    from models.skip_alexnet import AlexNet
    import os

    model_names = ['AlexNet']
    base_dir = 'saved_outputs/'
    single_angle_sep_dirs = [base_dir + f'single_angle_seps_{model_name}/' for model_name in model_names]

    sep_names = ['0.5', '1.0', '2.0', '5.0', '10.0']

    stimulus_noise_sd = 0.02
    confidence_noise_sd = 0.3
    num_steps = 25

    num_neurons_range = [150]

    for model_name, single_angle_sep_dir in zip(model_names, single_angle_sep_dirs):
        print("Model name: ", model_name)
        if model_name == 'AlexNet_skip' or model_name == 'AlexNet':
            from models.skip_alexnet import AlexNet
        elif model_name == 'alexnet_noskip':
            from models.noskip_alexnet import AlexNet
        elif model_name == 'ResNet18_skip':
            from models.ResNet18_skip import ResNet18
        elif model_name == 'VGG16_skip':
            from models.VGG16_skip import VGG16
        elif model_name == 'VGG11_skip':
            from models.VGG11_skip import VGG11
        elif model_name == 'DenseNet121_skip':
            from models.DenseNet121_skip import DenseNet121
        elif model_name == 'ConvNeXt_skip':
            from models.ConvNeXt_skip import ConvNeXt as ConvNeXt
        elif model_name == 'GoogLeNet_skip':
            from models.GoogLeNet_skip import GoogLeNet
        elif model_name == 'alexnet_noskip':
            from models.noskip_alexnet import AlexNet
        elif model_name == 'EfficientNet_skip':
            from models.EfficientNet_skip import EfficientNet
        elif model_name == 'SqueezeNet_skip':
            from models.SqueezeNet_skip import SqueezeNet
        elif model_name == 'MNASNet_skip':
            from models.MNASNet_skip import MNASNet
        elif model_name == 'MobileNetV2_skip':
            from models.MobileNetV2_skip import MobileNetV2
        elif model_name == 'MobileNetV3_skip':
            from models.MobileNetV3_skip import MobileNetV3
        else:
            raise ValueError('Invalid model name')

        for num_neurons in num_neurons_range:
            print("Num neurons: ", num_neurons)
            trials = range(1,21)
            for trial_num in trials:
                    print('Trial: ', trial_num)
                    for sep in sep_names:
                        # For loading old models
                        save_dir = single_angle_sep_dir + f'{sep}_angle_sep/'
                        path = save_dir+'models/original_model_0_sf_0.05_sep_'+str(sep)+'_trial_'+str(trial_num)+'.pth'
                        if model_name == 'AlexNet_skip' or model_name == 'alexnet_noskip':
                            model = AlexNet()
                        elif model_name == 'ResNet18_skip':
                            model = ResNet18()
                        elif model_name == 'VGG16_skip':
                            model = VGG16()
                        elif model_name == 'EfficientNet_skip':
                            model = EfficientNet()
                        elif model_name == 'DenseNet121_skip':
                            model = DenseNet121()
                        elif model_name == 'ConvNeXt_skip':
                            model = ConvNeXt()
                        elif model_name == 'GoogLeNet_skip':
                            model = GoogLeNet()
                        elif model_name == 'VGG11_skip':
                            model = VGG11()
                        elif model_name == 'SqueezeNet_skip':
                            model = SqueezeNet()
                        elif model_name == 'MNASNet_skip':
                            model = MNASNet()
                        elif model_name == 'MobileNetV2_skip':
                            model = MobileNetV2()
                        elif model_name == 'MobileNetV3_skip':
                            model = MobileNetV3()
                        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

                        max_values = []
                        indices = []
                        values_with_signs = []

                        # Get all highest weight neurons
                        for i in range(0, num_neurons):
                            if model_name == 'AlexNet_skip' or model_name == 'alexnet_noskip' or model_name == 'VGG11_skip' or model_name == 'VGG16_skip':
                                value, index = torch.max(torch.abs(model.fc1.weight.data[0]), dim=0)
                                value_with_sign = model.fc1.weight.data[0][index]
                                max_values.append(torch.Tensor.item(value))
                                indices.append(torch.Tensor.item(index))
                                values_with_signs.append(torch.Tensor.item(value_with_sign))
                                model.fc1.weight.data[0][index] = 0.0
                            else:
                                try:
                                    value, index = torch.max(torch.abs(model.fc.weight.data[0]), dim=0)
                                    value_with_sign = model.fc.weight.data[0][index]
                                    max_values.append(torch.Tensor.item(value))
                                    indices.append(torch.Tensor.item(index))
                                    values_with_signs.append(torch.Tensor.item(value_with_sign))
                                    # print("Max val: ", value, " Index: ", index, "With sign: ", value_with_sign)
                                    model.fc.weight.data[0][index] = 0.0
                                except:
                                    print('Model does not have attribute fc')
                        
                        # Save unshuffled indices
                        to_save = np.row_stack((max_values, indices, values_with_signs))
                        np.savetxt(save_dir+'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', to_save, delimiter=",")