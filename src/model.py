# import torch.nn as nn
# from torchvision import models

# def get_model():
#     model = models.resnet18(pretrained=True)

#     for param in model.parameters():
#         param.requires_grad = False

#     model.fc = nn.Linear(model.fc.in_features, 2)
#     return model
import torch.nn as nn
from torchvision import models

def get_model():
    """
    Builds the ResNet18 model.
    Compatible with both new and old versions of torchvision.
    """
    try:
        # MODERN WAY (torchvision v0.13+)
        # We try to use the 'weights' parameter first
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except AttributeError:
        # LEGACY WAY (Backup for older versions)
        # If the above fails, we use the old 'pretrained' parameter
        print("Note: Using legacy 'pretrained' method (older torchvision detected)")
        model = models.resnet18(pretrained=True)

    # Freeze the early layers (The "Eyes")
    # We don't want to retrain the parts that already know how to see lines/shapes
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer (The "Brain")
    # We change the output to 2 classes (Real vs AI)
    # This new layer is NOT frozen, so it will learn from your data
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    return model