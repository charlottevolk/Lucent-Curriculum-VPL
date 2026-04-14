# train the alexnet model (from test_alexnet.py) on the grating dataset
# load the model and the dataset

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
    import copy
    import random
    import math
    #import cv2
    import PIL
    from PIL import Image
    import scipy
    from scipy import ndimage
    from dataset_activations import GratingDataset
    from transforms import GaussianNoise
    from activation_models.activation_alexnet import AlexNet
    from activation_models.activation_noskip_alexnet import AlexNet as AlexNet_noskip
    from statistics import mean
    from copy_weights import copy_weights_alexnet, copy_weights_resnet18, copy_weights_vgg11, copy_weights_vgg16, copy_weights_densenet121, copy_weights_convnext
    from copy_weights import copy_weights_squeezenet, copy_weights_mobilenetv2, copy_weights_mobilenetv3, copy_weights_efficientnet, copy_weights_mnasnet, copy_weights_googlenet
    from activation_models.activation_resnet18_skip import ResNet18 as ResNet18_skip
    from activation_models.activation_VGG11_skip import VGG11 as VGG11_skip
    from activation_models.activation_VGG16_skip import VGG16 as VGG16_skip
    from activation_models.activation_densenet121_skip import DenseNet121 as DenseNet121_skip
    from activation_models.activation_efficientnet_skip import EfficientNet as EfficientNet_skip
    from activation_models.activation_convnext_skip import ConvNeXt as ConvNeXt_skip
    from activation_models.activation_googlenet_skip import GoogLeNet as GoogLeNet_skip
    from activation_models.activation_mnasnet_skip import MNASNet as MNASNet_skip
    from activation_models.activation_squeezenet_skip import SqueezeNet as SqueezeNet_skip
    from activation_models.activation_mobilenetv2_skip import MobileNetV2 as MobileNetV2_skip
    from activation_models.activation_mobilenetv3_skip import MobileNetV3 as MobileNetV3_skip

    # set the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_dir = 'saved_outputs/'
    model_names = ['AlexNet']

    # set the parameters for the dataset
    root_dir = 'StimulusImages/imagenet_100_1000/' # Put the shared 1000 images from the COCO dataset here
    num_images = 100

    # set the parameters for the model
    num_classes = 2
    batch_size = 1
    num_epochs = 1
    learning_rate = 0.0001

    stimulus_noise_sd = 0.0 # no noise for collecting activations

    # set the transforms
    data_transforms = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        GaussianNoise(0, stimulus_noise_sd), # STANDARD DEVIATION OF GAUSSIAN NOISE
    ])
    
    train_dir = [root_dir,0,0.05,0.5]

    for model_name in model_names:
        print("Model name: ", model_name)
        
        save_dir = base_dir+f'collecting_activations_imagenet_100_{model_name}/'
        if not os.path.exists(save_dir+'activations/'):
            os.makedirs(save_dir+'activations/')
        
        trial_nums = range(1,21)

        for trial_num in trial_nums:

            test_count = 0
            count = 0

            model = []
            seps = []

            def getActivation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            
            activation = {}
            
            test_losses = []
            train_losses = []
            train_accuracy = []
            test_accuracy = []

            #For new models
            if model_name == 'AlexNet':
                model.append(AlexNet())
                alexnet = torchvision.models.alexnet(pretrained=True)
                copy_weights_alexnet(model[count], alexnet)
            elif model_name == 'alexnet_noskip':
                model.append(AlexNet_noskip())
                alexnet = torchvision.models.alexnet(pretrained=True)
                copy_weights_alexnet(model[count], alexnet)
            elif model_name == 'ResNet18':
                model.append(ResNet18())
                resnet18 = torchvision.models.resnet18(pretrained=True)
                copy_weights_resnet18(model[count], resnet18)
            elif model_name == 'ResNet18_skip':
                model.append(ResNet18_skip())
                resnet18_skip = torchvision.models.resnet18(pretrained=True)
                copy_weights_resnet18(model[count], resnet18_skip)
            elif model_name == 'VGG11':
                model.append(VGG11())
                vgg11 = torchvision.models.vgg11(pretrained=True)
                copy_weights_vgg11(model[count], vgg11)
            elif model_name == 'VGG11_skip':
                model.append(VGG11_skip())
                vgg11_skip = torchvision.models.vgg11(pretrained=True)
                copy_weights_vgg11(model[count], vgg11_skip)
            elif model_name == 'VGG16':
                model.append(VGG16())
                vgg16 = torchvision.models.vgg16(pretrained=True)
                copy_weights_vgg16(model[count], vgg16)
            elif model_name == 'VGG16_skip':
                model.append(VGG16_skip())
                vgg16_skip = torchvision.models.vgg16(pretrained=True)
                copy_weights_vgg16(model[count], vgg16_skip)
            elif model_name == 'EfficientNet_skip':
                model.append(EfficientNet_skip())
                efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
                copy_weights_efficientnet(model[count], efficientnet)
            elif model_name == 'SqueezeNet_skip':
                model.append(SqueezeNet_skip())
                squeezenet = torchvision.models.squeezenet1_1(pretrained=True)
                copy_weights_squeezenet(model[count], squeezenet)
            elif model_name == 'MobileNetV2_skip':
                model.append(MobileNetV2_skip())
                mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
                copy_weights_mobilenetv2(model[count], mobilenetv2)
            elif model_name == 'MobileNetV3_skip':
                model.append(MobileNetV3_skip())
                mobilenetv3 = torchvision.models.mobilenet_v3_small(pretrained=True)
                copy_weights_mobilenetv3(model[count], mobilenetv3)
            elif model_name == 'MNASNet_skip':
                model.append(MNASNet_skip())
                mnasnet = torchvision.models.mnasnet1_0(pretrained=True)
                copy_weights_mnasnet(model[count], mnasnet)
            elif model_name == 'DenseNet121':
                model.append(DenseNet121())
                densenet121 = torchvision.models.densenet121(pretrained=True)
                copy_weights_densenet121(model[count], densenet121)
            elif model_name == 'DenseNet121_skip':
                model.append(DenseNet121_skip())
                densenet121_skip = torchvision.models.densenet121(pretrained=True)
                copy_weights_densenet121(model[count], densenet121_skip)
            elif model_name == 'ConvNeXt':
                model.append(ConvNeXt())
                convnext = torchvision.models.convnext_tiny(pretrained=True)
                copy_weights_convnext(model[count], convnext)
            elif model_name == 'ConvNeXt_skip':
                model.append(ConvNeXt_skip())
                convnext = torchvision.models.convnext_tiny(pretrained=True)
                copy_weights_convnext(model[count], convnext)
            elif model_name == 'GoogLeNet_skip':
                model.append(GoogLeNet_skip())
                googlenet = torchvision.models.googlenet(
                    weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1,
                    aux_logits=True,
                )
                copy_weights_googlenet(model[count], googlenet)
            elif model_name == 'SqueezeNet':
                model.append(SqueezeNet())
                squeezenet = torchvision.models.squeezenet1_1(pretrained=True)
                copy_weights_squeezenet(model[count], squeezenet)
            elif model_name == 'MobileNetV2':
                model.append(MobileNetV2())
                mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
                copy_weights_mobilenetv2(model[count], mobilenetv2)
            model[count] = model[count].to(device)

            # Register hooks based on model architecture
            if model_name == 'AlexNet':
                # AlexNet: register hooks for all skip connection layers
                model[count].conv1.register_forward_hook(getActivation("c1"))
                model[count].pool1.register_forward_hook(getActivation("p1"))
                model[count].conv2.register_forward_hook(getActivation("c2"))
                model[count].pool2.register_forward_hook(getActivation("p2"))
                model[count].conv3.register_forward_hook(getActivation("c3"))
                model[count].conv4.register_forward_hook(getActivation("c4"))
                model[count].conv5.register_forward_hook(getActivation("c5"))
                model[count].pool3.register_forward_hook(getActivation("p3"))
            elif model_name == 'alexnet_noskip':
                # AlexNet no-skip: register hook for final pooling layer before fc1
                model[count].pool3.register_forward_hook(getActivation("p3"))
            elif model_name == 'ResNet18':
                # ResNet18: prefer avgpool (input to fc); fall back to layer4 for variants without avgpool.
                if hasattr(model[count], "avgpool"):
                    model[count].avgpool.register_forward_hook(getActivation("avgpool"))
                else:
                    model[count].layer4.register_forward_hook(getActivation("layer4"))
            elif model_name == 'ResNet18_skip':
                # ResNet18 with skip connections: register hooks for all skip connection layers
                model[count].conv1.register_forward_hook(getActivation("conv1"))
                model[count].maxpool.register_forward_hook(getActivation("maxpool"))
                model[count].layer1.register_forward_hook(getActivation("layer1"))
                model[count].layer2.register_forward_hook(getActivation("layer2"))
                model[count].layer3.register_forward_hook(getActivation("layer3"))
                model[count].layer4.register_forward_hook(getActivation("layer4"))
            elif model_name == 'VGG11':
                # VGG11: register hook for final pooling layer before fc1
                model[count].pool5.register_forward_hook(getActivation("pool5"))
            elif model_name == 'VGG11_skip':
                # VGG11 with skip connections: register hooks for all skip connection layers
                model[count].conv1.register_forward_hook(getActivation("c1"))
                model[count].pool1.register_forward_hook(getActivation("p1"))
                model[count].conv2.register_forward_hook(getActivation("c2"))
                model[count].pool2.register_forward_hook(getActivation("p2"))
                model[count].conv3_1.register_forward_hook(getActivation("c3_1"))
                model[count].conv3_2.register_forward_hook(getActivation("c3_2"))
                model[count].pool3.register_forward_hook(getActivation("p3"))
                model[count].conv4_1.register_forward_hook(getActivation("c4_1"))
                model[count].conv4_2.register_forward_hook(getActivation("c4_2"))
                model[count].pool4.register_forward_hook(getActivation("p4"))
                model[count].conv5_1.register_forward_hook(getActivation("c5_1"))
                model[count].conv5_2.register_forward_hook(getActivation("c5_2"))
                model[count].pool5.register_forward_hook(getActivation("p5"))
            elif model_name == 'VGG16':
                # VGG16: register hook for fc2 output (input to fc3 readout layer)
                model[count].fc2.register_forward_hook(getActivation("fc2"))
            elif model_name == 'VGG16_skip':
                # VGG16 with skip connections: register hooks for all skip connection layers
                model[count].conv1_1.register_forward_hook(getActivation("c1_1"))
                model[count].conv1_2.register_forward_hook(getActivation("c1_2"))
                model[count].pool1.register_forward_hook(getActivation("p1"))
                model[count].conv2_1.register_forward_hook(getActivation("c2_1"))
                model[count].conv2_2.register_forward_hook(getActivation("c2_2"))
                model[count].pool2.register_forward_hook(getActivation("p2"))
                model[count].conv3_1.register_forward_hook(getActivation("c3_1"))
                model[count].conv3_2.register_forward_hook(getActivation("c3_2"))
                model[count].conv3_3.register_forward_hook(getActivation("c3_3"))
                model[count].pool3.register_forward_hook(getActivation("p3"))
                model[count].conv4_1.register_forward_hook(getActivation("c4_1"))
                model[count].conv4_2.register_forward_hook(getActivation("c4_2"))
                model[count].conv4_3.register_forward_hook(getActivation("c4_3"))
                model[count].pool4.register_forward_hook(getActivation("p4"))
                model[count].conv5_1.register_forward_hook(getActivation("c5_1"))
                model[count].conv5_2.register_forward_hook(getActivation("c5_2"))
                model[count].conv5_3.register_forward_hook(getActivation("c5_3"))
                model[count].pool5.register_forward_hook(getActivation("p5"))
            elif model_name == 'DenseNet121':
                # DenseNet121: register hook for avgpool output (input to fc layer)
                model[count].avgpool.register_forward_hook(getActivation("avgpool"))
            elif model_name == 'ConvNeXt':
                # ConvNeXt: register hook for avgpool output (input to classifier)
                model[count].avgpool.register_forward_hook(getActivation("avgpool"))
            elif model_name == 'SqueezeNet':
                # SqueezeNet: register hook for final pooling layer before classifier
                model[count].avgpool.register_forward_hook(getActivation("avgpool"))
            elif model_name == 'MobileNetV2':
                # MobileNetV2: register hook for final pooling layer before classifier
                model[count].avgpool.register_forward_hook(getActivation("avgpool"))
            elif model_name == 'EfficientNet_skip':
                # EfficientNet with skip connections: stem + each stage block + head
                model[count].stem.register_forward_hook(getActivation("stem"))
                for i, block in enumerate(model[count].blocks):
                    block.register_forward_hook(getActivation(f"block{i+1}"))
                model[count].head.register_forward_hook(getActivation("head"))
            elif model_name == 'SqueezeNet_skip':
                # SqueezeNet with skip connections: activations from listed feature indices
                for i in model[count].skip_indices:
                    model[count].features[i].register_forward_hook(getActivation(f"f{i}"))
            elif model_name == 'MobileNetV2_skip':
                # MobileNetV2 with skip connections: each feature block
                for i, block in enumerate(model[count].blocks):
                    block.register_forward_hook(getActivation(f"block{i+1}"))
            elif model_name == 'MobileNetV3_skip':
                # MobileNetV3 with skip connections: each feature block
                for i, block in enumerate(model[count].blocks):
                    block.register_forward_hook(getActivation(f"block{i+1}"))
            elif model_name == 'DenseNet121_skip':
                # DenseNet121 with skip connections: conv0 + each dense/transition block + norm5
                model[count].features.conv0.register_forward_hook(getActivation("conv0"))
                for i, block in enumerate(model[count].blocks):
                    block.register_forward_hook(getActivation(f"block{i+1}"))
                model[count].features.norm5.register_forward_hook(getActivation("norm5"))
            elif model_name == 'ConvNeXt_skip':
                # ConvNeXt with skip connections: each stage/downsample module output
                for i, layer in enumerate(model[count].features):
                    layer.register_forward_hook(getActivation(f"layer{i+1}"))
            elif model_name == 'GoogLeNet_skip':
                # GoogLeNet with skip connections: key conv/inception modules
                model[count].backbone.conv1.register_forward_hook(getActivation("conv1"))
                model[count].backbone.conv2.register_forward_hook(getActivation("conv2"))
                model[count].backbone.conv3.register_forward_hook(getActivation("conv3"))
                model[count].backbone.inception3a.register_forward_hook(getActivation("inception3a"))
                model[count].backbone.inception3b.register_forward_hook(getActivation("inception3b"))
                model[count].backbone.inception4a.register_forward_hook(getActivation("inception4a"))
                model[count].backbone.inception4b.register_forward_hook(getActivation("inception4b"))
                model[count].backbone.inception4c.register_forward_hook(getActivation("inception4c"))
                model[count].backbone.inception4d.register_forward_hook(getActivation("inception4d"))
                model[count].backbone.inception4e.register_forward_hook(getActivation("inception4e"))
                model[count].backbone.inception5a.register_forward_hook(getActivation("inception5a"))
                model[count].backbone.inception5b.register_forward_hook(getActivation("inception5b"))
            elif model_name == 'MNASNet_skip':
                # MNASNet with skip connections: stem + each block + head
                model[count].stem.register_forward_hook(getActivation("stem"))
                for i, block in enumerate(model[count].blocks):
                    block.register_forward_hook(getActivation(f"block{i+1}"))
                model[count].head.register_forward_hook(getActivation("head"))

            # Get parameters
            train_root_dir = train_dir[0]
            train_ref_ori = train_dir[1]
            train_sf = train_dir[2]
            train_sep = train_dir[3]

            #plt.imshow(train_ref_img.permute(1,2,0))

            # load the dataset
            train_grating_dataset = GratingDataset(train_root_dir, transform=data_transforms, num_images=num_images)

            # create the dataloader
            train_dataloader = DataLoader(train_grating_dataset, batch_size=1, shuffle=True, num_workers=4)

            iter_count = 0

            # train the model

            activations_all = []
                
            for epoch in range(num_epochs):
                for (train_i, train_data) in enumerate(train_dataloader):
                    activations = []

                    # set model to eval mode
                    model[count].eval()

                    train_inputs, train_labels = train_data
                    train_inputs = train_inputs.to(device)
                    train_labels = train_labels.to(device)

                    train_outputs = model[count](train_inputs, train_sep)

                    # Collect activations based on model architecture
                    if model_name == 'AlexNet':
                        # AlexNet: concatenate all skip connection activations (matches fc1 input)
                        activations.append(np.array(torch.tensor(activation["c1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c4"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c5"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p3"].cpu().numpy()).flatten()))
                    elif model_name == 'alexnet_noskip':
                        # AlexNet no-skip: just the final pooling layer
                        activations.append(np.array(torch.tensor(activation["p3"].cpu().numpy()).flatten()))
                    elif model_name == 'ResNet18':
                        # ResNet18: avgpool for standard model, layer4 for variants without avgpool.
                        if "avgpool" in activation:
                            activations.append(np.array(torch.tensor(activation["avgpool"].cpu().numpy()).flatten()))
                        else:
                            activations.append(np.array(torch.tensor(activation["layer4"].cpu().numpy()).flatten()))
                    elif model_name == 'ResNet18_skip':
                        # ResNet18 with skip connections: concatenate all skip connection activations (matches fc input)
                        activations.append(np.array(torch.tensor(activation["conv1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["maxpool"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["layer1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["layer2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["layer3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["layer4"].cpu().numpy()).flatten()))
                    elif model_name == 'VGG11':
                        # VGG11: activations from final pooling layer before fc1 (512*7*7=25088-dim input to fc1 readout layer)
                        activations.append(np.array(torch.tensor(activation["pool5"].cpu().numpy()).flatten()))
                    elif model_name == 'VGG11_skip':
                        # VGG11 with skip connections: concatenate all skip connection activations (matches fc1 input)
                        activations.append(np.array(torch.tensor(activation["c1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c3_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c3_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c4_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c4_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p4"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c5_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c5_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p5"].cpu().numpy()).flatten()))
                    elif model_name == 'VGG16':
                        # VGG16: activations from fc2 (4096-dim input to fc3 readout layer)
                        activations.append(np.array(torch.tensor(activation["fc2"].cpu().numpy()).flatten()))
                    elif model_name == 'VGG16_skip':
                        # VGG16 with skip connections: concatenate all skip connection activations (matches fc1 input)
                        activations.append(np.array(torch.tensor(activation["c1_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c1_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c2_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c2_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c3_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c3_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c3_3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c4_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c4_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c4_3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p4"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c5_1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c5_2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["c5_3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["p5"].cpu().numpy()).flatten()))
                    elif model_name == 'DenseNet121':
                        # DenseNet121: activations from avgpool (1024-dim input to fc layer)
                        activations.append(np.array(torch.tensor(activation["avgpool"].cpu().numpy()).flatten()))
                    elif model_name == 'ConvNeXt':
                        # ConvNeXt: activations from avgpool (768-dim for tiny variant input to classifier)
                        activations.append(np.array(torch.tensor(activation["avgpool"].cpu().numpy()).flatten()))
                    elif model_name == 'SqueezeNet':
                        # SqueezeNet: activations from avgpool (512-dim input to classifier)
                        activations.append(np.array(torch.tensor(activation["avgpool"].cpu().numpy()).flatten()))
                    elif model_name == 'MobileNetV2':
                        # MobileNetV2: activations from avgpool (1280-dim input to classifier)
                        activations.append(np.array(torch.tensor(activation["avgpool"].cpu().numpy()).flatten()))
                    elif model_name == 'EfficientNet_skip':
                        # EfficientNet with skip connections: stem + each stage block + head
                        activations.append(np.array(torch.tensor(activation["stem"].cpu().numpy()).flatten()))
                        for i in range(len(model[count].blocks)):
                            activations.append(np.array(torch.tensor(activation[f"block{i+1}"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["head"].cpu().numpy()).flatten()))
                    elif model_name == 'SqueezeNet_skip':
                        # SqueezeNet with skip connections: selected feature indices
                        for i in model[count].skip_indices:
                            activations.append(np.array(torch.tensor(activation[f"f{i}"].cpu().numpy()).flatten()))
                    elif model_name == 'MobileNetV2_skip':
                        # MobileNetV2 with skip connections: every feature block output
                        for i in range(len(model[count].blocks)):
                            activations.append(np.array(torch.tensor(activation[f"block{i+1}"].cpu().numpy()).flatten()))
                    elif model_name == 'MobileNetV3_skip':
                        # MobileNetV3 with skip connections: every feature block output
                        for i in range(len(model[count].blocks)):
                            activations.append(np.array(torch.tensor(activation[f"block{i+1}"].cpu().numpy()).flatten()))
                    elif model_name == 'DenseNet121_skip':
                        # DenseNet121 with skip connections: conv0 + each block + final norm
                        activations.append(np.array(torch.tensor(activation["conv0"].cpu().numpy()).flatten()))
                        for i in range(len(model[count].blocks)):
                            activations.append(np.array(torch.tensor(activation[f"block{i+1}"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["norm5"].cpu().numpy()).flatten()))
                    elif model_name == 'ConvNeXt_skip':
                        # ConvNeXt with skip connections: each stage/downsample output
                        for i in range(len(model[count].features)):
                            activations.append(np.array(torch.tensor(activation[f"layer{i+1}"].cpu().numpy()).flatten()))
                    elif model_name == 'GoogLeNet_skip':
                        # GoogLeNet with skip connections: key conv/inception module outputs
                        activations.append(np.array(torch.tensor(activation["conv1"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["conv2"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["conv3"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception3a"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception3b"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception4a"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception4b"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception4c"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception4d"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception4e"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception5a"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["inception5b"].cpu().numpy()).flatten()))
                    elif model_name == 'MNASNet_skip':
                        # MNASNet with skip connections: stem + each block + head
                        activations.append(np.array(torch.tensor(activation["stem"].cpu().numpy()).flatten()))
                        for i in range(len(model[count].blocks)):
                            activations.append(np.array(torch.tensor(activation[f"block{i+1}"].cpu().numpy()).flatten()))
                        activations.append(np.array(torch.tensor(activation["head"].cpu().numpy()).flatten()))
                    
                    iter_count += 1
                    activations_all.append(np.concatenate(activations))
                
            
            print('Finished Training & Testing')

            np.save(save_dir+'activations/100_imgs_all_activations_no_noise_1000_imagenet_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_model_trial_'+str(trial_num)+'_batch_size_1.npy', activations_all, allow_pickle=True)
