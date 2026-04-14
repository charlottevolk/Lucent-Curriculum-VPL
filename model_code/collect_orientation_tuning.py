if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision import transforms, utils
    from torchvision import datasets, models, transforms
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import os
    import re
    import PIL
    from PIL import Image
    from dataset_single_example import GratingDataset
    from transforms import GaussianNoise
    from activation_models.activation_alexnet import AlexNet
    from copy_weights import copy_weights_alexnet

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_dir = 'saved_outputs/'
    
    save_dir = base_dir + 'orientation_tuning_alexnet/'
    if not os.path.exists(save_dir + 'activations/'):
        os.makedirs(save_dir + 'activations/')

    # set the parameters for the dataset
    root_dir = 'StimulusImages/SG_train_double_sf/'  # directory containing stimulus images
    num_images = 20

    # set the parameters for the model
    num_classes = 2
    batch_size = 1
    num_epochs = 1
    learning_rate = 0.0001

    stimulus_noise_sd = 0.02

    data_transforms = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        GaussianNoise(0, stimulus_noise_sd),
    ])

    train_dirs = [[root_dir+f'sep_{sep}',0,0.05,sep] for sep in [0.5]+[float(x) for x in range(1, 181)]]


    trials = range(1,21)

    model = AlexNet()
    alexnet = torchvision.models.alexnet(pretrained=True)
    copy_weights_alexnet(model, alexnet)
    model = model.to(device)
    model.eval()

    def getActivation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(getActivation('c1'))
    model.pool1.register_forward_hook(getActivation('p1'))
    model.conv2.register_forward_hook(getActivation('c2'))
    model.pool2.register_forward_hook(getActivation('p2'))
    model.conv3.register_forward_hook(getActivation('c3'))
    model.conv4.register_forward_hook(getActivation('c4'))
    model.conv5.register_forward_hook(getActivation('c5'))
    model.pool3.register_forward_hook(getActivation('p3'))

    for dir in train_dirs:
        train_root_dir = dir[0]
        train_ref_ori = dir[1]
        train_sf = dir[2]
        train_sep = dir[3]

        for trial in trials:
            test_count = 0
            count = 0
            
            activation = {}

            train_grating_dataset = GratingDataset(train_root_dir, transform=data_transforms, num_images=num_images)
            train_dataloader = DataLoader(train_grating_dataset, batch_size=1, shuffle=False, num_workers=4)

            activations_all_CW = []
            activations_all_CCW = []

            for (train_i, train_data) in enumerate(train_dataloader):
                activations_CW = []
                activations_CCW = []
                train_inputs, train_labels, train_seps = train_data
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)
                train_outputs = model(train_inputs, train_sep)

                if train_labels.item() == 1:
                    activations_CW.append(np.array(F.relu(activation["c1"]).cpu().numpy().flatten()))
                    activations_CW.append(np.array(activation["p1"].cpu().numpy().flatten()))
                    activations_CW.append(np.array(F.relu(activation["c2"]).cpu().numpy().flatten()))
                    activations_CW.append(np.array(activation["p2"].cpu().numpy().flatten()))
                    activations_CW.append(np.array(F.relu(activation["c3"]).cpu().numpy().flatten()))
                    activations_CW.append(np.array(F.relu(activation["c4"]).cpu().numpy().flatten()))
                    activations_CW.append(np.array(F.relu(activation["c5"]).cpu().numpy().flatten()))
                    activations_CW.append(np.array(activation["p3"].cpu().numpy().flatten()))
                elif train_labels.item() == 0:
                    activations_CCW.append(np.array(F.relu(activation["c1"]).cpu().numpy().flatten()))
                    activations_CCW.append(np.array(activation["p1"].cpu().numpy().flatten()))
                    activations_CCW.append(np.array(F.relu(activation["c2"]).cpu().numpy().flatten()))
                    activations_CCW.append(np.array(activation["p2"].cpu().numpy().flatten()))
                    activations_CCW.append(np.array(F.relu(activation["c3"]).cpu().numpy().flatten()))
                    activations_CCW.append(np.array(F.relu(activation["c4"]).cpu().numpy().flatten()))
                    activations_CCW.append(np.array(F.relu(activation["c5"]).cpu().numpy().flatten()))
                    activations_CCW.append(np.array(activation["p3"].cpu().numpy().flatten()))

                if train_labels.item() == 1:
                    activations_all_CW.append(np.concatenate(activations_CW))
                elif train_labels.item() == 0:
                    activations_all_CCW.append(np.concatenate(activations_CCW))

            print('Saving activations for trial ' + str(trial) + '...')
            np.save(save_dir+f'activations_all_CW_sep_{train_sep}_sf_{train_sf}_trial_' + str(trial) + '.npy', activations_all_CW)
            np.save(save_dir+f'activations_all_CCW_sep_{train_sep}_sf_{train_sf}_trial_' + str(trial) + '.npy', activations_all_CCW)