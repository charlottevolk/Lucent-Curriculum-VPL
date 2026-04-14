"""
Functions for copying pretrained weights from torchvision models to custom architectures.
"""

def copy_weights_alexnet(model, alexnet):
    """Copy weights from pretrained AlexNet to custom AlexNet model."""
    model.conv1.weight.data = alexnet.features[0].weight.data
    model.conv1.bias.data = alexnet.features[0].bias.data
    model.conv2.weight.data = alexnet.features[3].weight.data
    model.conv2.bias.data = alexnet.features[3].bias.data
    model.conv3.weight.data = alexnet.features[6].weight.data
    model.conv3.bias.data = alexnet.features[6].bias.data
    model.conv4.weight.data = alexnet.features[8].weight.data
    model.conv4.bias.data = alexnet.features[8].bias.data
    model.conv5.weight.data = alexnet.features[10].weight.data
    model.conv5.bias.data = alexnet.features[10].bias.data


def copy_weights_resnet18(model, resnet):
    """Copy weights from pretrained ResNet18 to custom ResNet18 model with BatchNorm."""
    # Copy initial conv layer
    model.conv1.weight.data = resnet.conv1.weight.data
    # Copy BatchNorm weights
    model.bn1.weight.data = resnet.bn1.weight.data
    model.bn1.bias.data = resnet.bn1.bias.data
    model.bn1.running_mean.data = resnet.bn1.running_mean.data
    model.bn1.running_var.data = resnet.bn1.running_var.data
    
    # Copy layer1 (2 BasicBlocks)
    for i in range(2):
        model.layer1[i].conv1.weight.data = resnet.layer1[i].conv1.weight.data
        model.layer1[i].bn1.weight.data = resnet.layer1[i].bn1.weight.data
        model.layer1[i].bn1.bias.data = resnet.layer1[i].bn1.bias.data
        model.layer1[i].bn1.running_mean.data = resnet.layer1[i].bn1.running_mean.data
        model.layer1[i].bn1.running_var.data = resnet.layer1[i].bn1.running_var.data
        model.layer1[i].conv2.weight.data = resnet.layer1[i].conv2.weight.data
        model.layer1[i].bn2.weight.data = resnet.layer1[i].bn2.weight.data
        model.layer1[i].bn2.bias.data = resnet.layer1[i].bn2.bias.data
        model.layer1[i].bn2.running_mean.data = resnet.layer1[i].bn2.running_mean.data
        model.layer1[i].bn2.running_var.data = resnet.layer1[i].bn2.running_var.data
        # Copy downsample if it exists
        if resnet.layer1[i].downsample is not None and len(resnet.layer1[i].downsample) > 0:
            model.layer1[i].downsample[0].weight.data = resnet.layer1[i].downsample[0].weight.data
            model.layer1[i].downsample[1].weight.data = resnet.layer1[i].downsample[1].weight.data
            model.layer1[i].downsample[1].bias.data = resnet.layer1[i].downsample[1].bias.data
            model.layer1[i].downsample[1].running_mean.data = resnet.layer1[i].downsample[1].running_mean.data
            model.layer1[i].downsample[1].running_var.data = resnet.layer1[i].downsample[1].running_var.data
    
    # Copy layer2 (2 BasicBlocks)
    for i in range(2):
        model.layer2[i].conv1.weight.data = resnet.layer2[i].conv1.weight.data
        model.layer2[i].bn1.weight.data = resnet.layer2[i].bn1.weight.data
        model.layer2[i].bn1.bias.data = resnet.layer2[i].bn1.bias.data
        model.layer2[i].bn1.running_mean.data = resnet.layer2[i].bn1.running_mean.data
        model.layer2[i].bn1.running_var.data = resnet.layer2[i].bn1.running_var.data
        model.layer2[i].conv2.weight.data = resnet.layer2[i].conv2.weight.data
        model.layer2[i].bn2.weight.data = resnet.layer2[i].bn2.weight.data
        model.layer2[i].bn2.bias.data = resnet.layer2[i].bn2.bias.data
        model.layer2[i].bn2.running_mean.data = resnet.layer2[i].bn2.running_mean.data
        model.layer2[i].bn2.running_var.data = resnet.layer2[i].bn2.running_var.data
        # Copy downsample if it exists
        if resnet.layer2[i].downsample is not None and len(resnet.layer2[i].downsample) > 0:
            model.layer2[i].downsample[0].weight.data = resnet.layer2[i].downsample[0].weight.data
            model.layer2[i].downsample[1].weight.data = resnet.layer2[i].downsample[1].weight.data
            model.layer2[i].downsample[1].bias.data = resnet.layer2[i].downsample[1].bias.data
            model.layer2[i].downsample[1].running_mean.data = resnet.layer2[i].downsample[1].running_mean.data
            model.layer2[i].downsample[1].running_var.data = resnet.layer2[i].downsample[1].running_var.data
    
    # Copy layer3 (2 BasicBlocks)
    for i in range(2):
        model.layer3[i].conv1.weight.data = resnet.layer3[i].conv1.weight.data
        model.layer3[i].bn1.weight.data = resnet.layer3[i].bn1.weight.data
        model.layer3[i].bn1.bias.data = resnet.layer3[i].bn1.bias.data
        model.layer3[i].bn1.running_mean.data = resnet.layer3[i].bn1.running_mean.data
        model.layer3[i].bn1.running_var.data = resnet.layer3[i].bn1.running_var.data
        model.layer3[i].conv2.weight.data = resnet.layer3[i].conv2.weight.data
        model.layer3[i].bn2.weight.data = resnet.layer3[i].bn2.weight.data
        model.layer3[i].bn2.bias.data = resnet.layer3[i].bn2.bias.data
        model.layer3[i].bn2.running_mean.data = resnet.layer3[i].bn2.running_mean.data
        model.layer3[i].bn2.running_var.data = resnet.layer3[i].bn2.running_var.data
        # Copy downsample if it exists
        if resnet.layer3[i].downsample is not None and len(resnet.layer3[i].downsample) > 0:
            model.layer3[i].downsample[0].weight.data = resnet.layer3[i].downsample[0].weight.data
            model.layer3[i].downsample[1].weight.data = resnet.layer3[i].downsample[1].weight.data
            model.layer3[i].downsample[1].bias.data = resnet.layer3[i].downsample[1].bias.data
            model.layer3[i].downsample[1].running_mean.data = resnet.layer3[i].downsample[1].running_mean.data
            model.layer3[i].downsample[1].running_var.data = resnet.layer3[i].downsample[1].running_var.data
    
    # Copy layer4 (2 BasicBlocks)
    for i in range(2):
        model.layer4[i].conv1.weight.data = resnet.layer4[i].conv1.weight.data
        model.layer4[i].bn1.weight.data = resnet.layer4[i].bn1.weight.data
        model.layer4[i].bn1.bias.data = resnet.layer4[i].bn1.bias.data
        model.layer4[i].bn1.running_mean.data = resnet.layer4[i].bn1.running_mean.data
        model.layer4[i].bn1.running_var.data = resnet.layer4[i].bn1.running_var.data
        model.layer4[i].conv2.weight.data = resnet.layer4[i].conv2.weight.data
        model.layer4[i].bn2.weight.data = resnet.layer4[i].bn2.weight.data
        model.layer4[i].bn2.bias.data = resnet.layer4[i].bn2.bias.data
        model.layer4[i].bn2.running_mean.data = resnet.layer4[i].bn2.running_mean.data
        model.layer4[i].bn2.running_var.data = resnet.layer4[i].bn2.running_var.data
        # Copy downsample if it exists
        if resnet.layer4[i].downsample is not None and len(resnet.layer4[i].downsample) > 0:
            model.layer4[i].downsample[0].weight.data = resnet.layer4[i].downsample[0].weight.data
            model.layer4[i].downsample[1].weight.data = resnet.layer4[i].downsample[1].weight.data
            model.layer4[i].downsample[1].bias.data = resnet.layer4[i].downsample[1].bias.data
            model.layer4[i].downsample[1].running_mean.data = resnet.layer4[i].downsample[1].running_mean.data
            model.layer4[i].downsample[1].running_var.data = resnet.layer4[i].downsample[1].running_var.data


def copy_weights_vgg16(model, vgg):
    """Copy weights from pretrained VGG16 to custom VGG16 model."""
    # VGG16 features are stored in a sequential container
    # Map to custom model's explicit layers
    model.conv1_1.weight.data = vgg.features[0].weight.data
    model.conv1_1.bias.data = vgg.features[0].bias.data
    model.conv1_2.weight.data = vgg.features[2].weight.data
    model.conv1_2.bias.data = vgg.features[2].bias.data
    
    model.conv2_1.weight.data = vgg.features[5].weight.data
    model.conv2_1.bias.data = vgg.features[5].bias.data
    model.conv2_2.weight.data = vgg.features[7].weight.data
    model.conv2_2.bias.data = vgg.features[7].bias.data
    
    model.conv3_1.weight.data = vgg.features[10].weight.data
    model.conv3_1.bias.data = vgg.features[10].bias.data
    model.conv3_2.weight.data = vgg.features[12].weight.data
    model.conv3_2.bias.data = vgg.features[12].bias.data
    model.conv3_3.weight.data = vgg.features[14].weight.data
    model.conv3_3.bias.data = vgg.features[14].bias.data
    
    model.conv4_1.weight.data = vgg.features[17].weight.data
    model.conv4_1.bias.data = vgg.features[17].bias.data
    model.conv4_2.weight.data = vgg.features[19].weight.data
    model.conv4_2.bias.data = vgg.features[19].bias.data
    model.conv4_3.weight.data = vgg.features[21].weight.data
    model.conv4_3.bias.data = vgg.features[21].bias.data
    
    model.conv5_1.weight.data = vgg.features[24].weight.data
    model.conv5_1.bias.data = vgg.features[24].bias.data
    model.conv5_2.weight.data = vgg.features[26].weight.data
    model.conv5_2.bias.data = vgg.features[26].bias.data
    model.conv5_3.weight.data = vgg.features[28].weight.data
    model.conv5_3.bias.data = vgg.features[28].bias.data


def copy_weights_vgg11(model, vgg):
    """Copy weights from pretrained VGG11 to custom VGG11 model."""
    # VGG11 features are stored in a sequential container
    # VGG11 architecture: 1-1-2-2-2 (8 conv layers total)
    
    # Block 1 - single conv
    model.conv1.weight.data = vgg.features[0].weight.data
    model.conv1.bias.data = vgg.features[0].bias.data
    
    # Block 2 - single conv
    model.conv2.weight.data = vgg.features[3].weight.data
    model.conv2.bias.data = vgg.features[3].bias.data
    
    # Block 3 - two convs
    model.conv3_1.weight.data = vgg.features[6].weight.data
    model.conv3_1.bias.data = vgg.features[6].bias.data
    model.conv3_2.weight.data = vgg.features[8].weight.data
    model.conv3_2.bias.data = vgg.features[8].bias.data
    
    # Block 4 - two convs
    model.conv4_1.weight.data = vgg.features[11].weight.data
    model.conv4_1.bias.data = vgg.features[11].bias.data
    model.conv4_2.weight.data = vgg.features[13].weight.data
    model.conv4_2.bias.data = vgg.features[13].bias.data
    
    # Block 5 - two convs
    model.conv5_1.weight.data = vgg.features[16].weight.data
    model.conv5_1.bias.data = vgg.features[16].bias.data
    model.conv5_2.weight.data = vgg.features[18].weight.data
    model.conv5_2.bias.data = vgg.features[18].bias.data


def copy_weights_densenet121(model, densenet):
    """Copy weights from pretrained DenseNet121 to custom DenseNet121 model."""
    # Copy all features (conv layers, batch norms, dense blocks, transitions)
    # DenseNet uses a Sequential container for features, which we can copy directly
    for name, module in densenet.features.named_children():
        if hasattr(model.features, name):
            target_module = getattr(model.features, name)
            # Copy weights recursively for each module
            _copy_module_weights(target_module, module)


def copy_weights_convnext(model, convnext):
    """Copy weights from pretrained ConvNeXt to custom ConvNeXt model."""
    import torch.nn as nn
    
    # Copy features using recursive module copy
    _copy_module_weights(model.features, convnext.features)
    
    # Copy classifier (LayerNorm2d, Flatten, Linear)
    # The classifier[0] is LayerNorm2d, classifier[2] is Linear (classifier[1] is Flatten with no params)
    if hasattr(model, 'classifier') and len(model.classifier) > 0:
        # Copy final LayerNorm
        if isinstance(model.classifier[0], nn.Module):
            _copy_module_weights(model.classifier[0], model.classifier[0])
        # Note: We don't copy classifier[2] (Linear) weights as they're initialized to zero in custom model


def copy_weights_efficientnet(model, efficientnet):
    """Copy weights from pretrained EfficientNet to custom EfficientNet model."""
    # Copy all features (conv layers, batch norms, blocks)
    for name, module in efficientnet.features.named_children():
        if hasattr(model.backbone.features, name):
            target_module = getattr(model.backbone.features, name)
            _copy_module_weights(target_module, module)
    # No need to copy classifier since it's replaced in custom model


def copy_weights_mnasnet(model, mnasnet):
    """Copy weights from pretrained MNASNet to custom MNASNet model."""
    for name, module in mnasnet.layers.named_children():
        if hasattr(model.backbone.layers, name):
            target_module = getattr(model.backbone.layers, name)
            _copy_module_weights(target_module, module)
    # No need to copy classifier since it's replaced in custom model


def copy_weights_mobilenetv2(model, mobilenetv2):
    """Copy weights from pretrained MobileNetV2 to custom MobileNetV2 model."""
    for name, module in mobilenetv2.features.named_children():
        if hasattr(model.backbone.features, name):
            target_module = getattr(model.backbone.features, name)
            _copy_module_weights(target_module, module)
    # No need to copy classifier since it's replaced in custom model


def copy_weights_mobilenetv3(model, mobilenetv3):
    """Copy weights from pretrained MobileNetV3 to custom MobileNetV3 model."""
    for name, module in mobilenetv3.features.named_children():
        if hasattr(model.backbone.features, name):
            target_module = getattr(model.backbone.features, name)
            _copy_module_weights(target_module, module)
    # No need to copy classifier since it's replaced in custom model


def copy_weights_squeezenet(model, squeezenet):
    # """Copy weights from pretrained SqueezeNet to custom SqueezeNet model."""
    # # Copy all features (conv layers, fire modules, etc.)
    # print("PRETRAINED FEATURES LAYOUT:")
    # print(squeezenet.features)
    # print("-------------------------")
    # print("CUSTOM FEATURES LAYOUT:")
    # print(model.features)
    for name, module in squeezenet.features.named_children():
        if hasattr(model.features, name):
            target_module = getattr(model.features, name)
            _copy_module_weights(target_module, module)
    # No need to copy classifier since it's replaced in custom model
    # model.features.load_state_dict(squeezenet.features.state_dict())


def copy_weights_googlenet(model, googlenet):
    """Copy weights from pretrained GoogLeNet to custom GoogLeNet model."""
    if hasattr(model, 'backbone'):
        _copy_module_weights(model.backbone, googlenet)
    else:
        _copy_module_weights(model, googlenet)


def _copy_module_weights(target, source):
    """Recursively copy weights from source module to target module."""
    import torch.nn as nn
    
    if isinstance(source, nn.Conv2d) and isinstance(target, nn.Conv2d):
        target.weight.data.copy_(source.weight.data)
        if source.bias is not None and target.bias is not None:
            target.bias.data.copy_(source.bias.data)
    
    elif isinstance(source, nn.BatchNorm2d) and isinstance(target, nn.BatchNorm2d):
        target.weight.data.copy_(source.weight.data)
        target.bias.data.copy_(source.bias.data)
        target.running_mean.data.copy_(source.running_mean.data)
        target.running_var.data.copy_(source.running_var.data)
        # FIX: Added num_batches_tracked for proper momentum tracking
        target.num_batches_tracked.data.copy_(source.num_batches_tracked.data)
    
    elif isinstance(source, nn.LayerNorm) and isinstance(target, nn.LayerNorm):
        target.weight.data.copy_(source.weight.data)
        target.bias.data.copy_(source.bias.data)
    
    # Handle LayerNorm2d (custom class) - check by class name since they're from different modules
    elif type(source).__name__ == 'LayerNorm2d' and type(target).__name__ == 'LayerNorm2d':
        target.weight.data.copy_(source.weight.data)
        target.bias.data.copy_(source.bias.data)
    
    elif isinstance(source, nn.Linear) and isinstance(target, nn.Linear):
        target.weight.data.copy_(source.weight.data)
        if source.bias is not None and target.bias is not None:
            target.bias.data.copy_(source.bias.data)
    
    # FIX: Removed the `elif isinstance(source, nn.Parameter)` block. 
    # named_children() only yields nn.Module objects, never raw nn.Parameters, 
    # making that block unreachable dead code.
    
    # Recursively copy for container modules
    elif hasattr(source, '_modules'):
        for name, child_module in source.named_children():
            if hasattr(target, name):
                target_child = getattr(target, name)
                _copy_module_weights(target_child, child_module)